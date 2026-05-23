"""
API principal para detecção de objetos.

Endpoints:
- /detection/fused  [PRINCIPAL] Pipeline semântico em 4 etapas (Qwen→YOLOE guiado→juiz→seleção)
- /detection        Qwen autônomo (legado)
- /detection/sys_prompt  Qwen com prompt customizado (legado)
- /detection/sequential  YOLOE → Qwen sequencial (legado)
- /health

Arquitetura /detection/fused — pipeline semântico:

  Etapa 1 (paralelo):
    a) Qwen analisa a imagem completa → lista objetos centrais + quantidades
       Ex: [{"label": "carregador", "count": 1}, {"label": "notebook", "count": 1}]
       Qwen NÃO gera bboxes nesta etapa — apenas interpretação semântica.
    b) YOLOE detecta todos os candidatos com bboxes precisas (labels ignorados).

  Etapa 2:
    Para cada candidato do YOLOE (filtrado por área), Qwen julga em paralelo:
    "Este crop corresponde a algum dos objetos centrais identificados?"
    → {"match": "carregador", "score": 0.9} ou {"match": null, "score": 0.0}

  Etapa 3 — Seleção:
    Para cada objeto central esperado (com sua contagem):
    - Filtra candidatos julgados como correspondentes
    - Seleciona os N melhores por score (respeitando a contagem)
    - Resultado: bbox precisa do YOLOE + label semântico do Qwen
"""

import json
import math
import os
import asyncio
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from system_instruction import (
    SCENE_INSTRUCTION, JUDGE_INSTRUCTION,
    SYSTEM_INSTRUCTION, SEQUENTIAL_INSTRUCTION,
)
from models import DetectedObject, BBox
from services.yoloe import run_yoloe, run_yoloe_prompted, get_yoloe
from services.qwen import (
    parse_scene_response, parse_judge_response,
    parse_qwen_response, parse_qwen_refine_response,
)
from services.fusion import serialize_detections
from utils.image import resize_base64_image, b64_to_pil, pil_to_b64

app = FastAPI()


# ──────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("Inicializando... Carregando modelo YOLO...")
    try:
        get_yoloe()
        print("Modelo YOLO carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo YOLO: {e}")
        raise

VLLM_URL   = os.getenv("VLLM_URL",   "http://vllm:8000") + "/v1/chat/completions"
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class Prompt(BaseModel):
    image: str

class PromptSys(BaseModel):
    image: str
    prompt: str


# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────

_MIN_AREA = 0.005   # < 0.5% da imagem → ruído
_MAX_AREA = 0.75    # > 75% da imagem → fundo
_CROP_PADDING = 0.05


# ──────────────────────────────────────────────
# Geometria
# ──────────────────────────────────────────────

def _bbox_area(bbox: BBox) -> float:
    return max(0.0, (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1))


def _centrality(bbox: BBox) -> float:
    """1.0 = centro da imagem, 0.0 = canto."""
    cx = (bbox.x1 + bbox.x2) / 2
    cy = (bbox.y1 + bbox.y2) / 2
    return max(0.0, 1.0 - 2 * max(abs(cx - 0.5), abs(cy - 0.5)))


def _crop_object(img: Image.Image, bbox: BBox) -> Image.Image:
    """Crop com padding, sem ultrapassar os limites da imagem."""
    w, h = img.size
    pad_x = (bbox.x2 - bbox.x1) * _CROP_PADDING
    pad_y = (bbox.y2 - bbox.y1) * _CROP_PADDING
    x1 = max(0.0, bbox.x1 - pad_x)
    y1 = max(0.0, bbox.y1 - pad_y)
    x2 = min(1.0, bbox.x2 + pad_x)
    y2 = min(1.0, bbox.y2 + pad_y)
    return img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))


# ──────────────────────────────────────────────
# Chamadas ao vLLM
# ──────────────────────────────────────────────

async def _call_scene(client: httpx.AsyncClient, image_b64: str) -> dict:
    """
    Etapa 1a: Qwen identifica objetos centrais + quantidades na imagem completa.
    Não gera bboxes — apenas interpretação semântica.
    """
    response = await client.post(
        VLLM_URL,
        json={
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": SCENE_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Look at this image. List the central foreground objects "
                                "and how many of each you can see. Ignore background elements."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": 256,
        },
    )
    result = response.json()
    print(f"[Scene] status={response.status_code} | {result}")
    return result


async def _call_judge(
    client: httpx.AsyncClient,
    crop_b64: str,
    central_objects: list[dict],
    candidate_idx: int,
) -> dict:
    """
    Etapa 2: Qwen julga se o crop de um candidato do YOLOE corresponde a algum
    dos objetos centrais identificados na Etapa 1.
    """
    expected_list = ", ".join(
        f'"{o["label"]}" (×{o["count"]})' for o in central_objects
    )
    response = await client.post(
        VLLM_URL,
        json={
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": JUDGE_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Expected central objects in the scene: [{expected_list}]\n\n"
                                "Does this crop show one of those objects? "
                                "The object must be the dominant element — "
                                "not just background or a supporting surface."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": 64,
        },
    )
    result = response.json()
    print(f"[Judge #{candidate_idx}] status={response.status_code} | {result}")
    return result


async def _call_vllm_legacy(
    client: httpx.AsyncClient, image_b64: str, system: str
) -> dict:
    """Chamada genérica legada ao Qwen (endpoints /detection e /detection/sys_prompt)."""
    response = await client.post(
        VLLM_URL,
        json={
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image and detect foreground objects."},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                },
            ],
        },
    )
    result = response.json()
    print(f"vLLM status={response.status_code} | response={result}")
    return result


def _bbox_prominence(bbox: BBox) -> float:
    area = _bbox_area(bbox)
    area_score = min(1.0, math.sqrt(area) * 2)
    cx = (bbox.x1 + bbox.x2) / 2
    cy = (bbox.y1 + bbox.y2) / 2
    centrality = max(0.0, 1.0 - 2 * max(abs(cx - 0.5), abs(cy - 0.5)))
    return round(0.6 * area_score + 0.4 * centrality, 4)


async def _call_vllm_sequential(
    client: httpx.AsyncClient,
    img: Image.Image,
    yoloe_objects: list[DetectedObject],
) -> dict:
    import json as _json
    filtered = [
        (i, obj) for i, obj in enumerate(yoloe_objects)
        if _bbox_area(obj.bbox) >= _MIN_AREA
    ] or list(enumerate(yoloe_objects))
    clean_b64 = pil_to_b64(img)
    payload = _json.dumps(
        [{"index": i, "label": obj.label,
          "conf": round(obj.yoloe_conf or obj.score, 4),
          "bbox_norm": {"x1": round(obj.bbox.x1, 4), "y1": round(obj.bbox.y1, 4),
                        "x2": round(obj.bbox.x2, 4), "y2": round(obj.bbox.y2, 4)},
          "area": round(_bbox_area(obj.bbox), 4),
          "prominence": _bbox_prominence(obj.bbox)}
         for i, obj in filtered],
        ensure_ascii=False,
    )
    response = await client.post(
        VLLM_URL,
        json={
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": SEQUENTIAL_INSTRUCTION},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": f"YOLOE detections:\n{payload}\n\nRename in Brazilian Portuguese."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}"}},
                ]},
            ],
        },
    )
    return response.json()


# ──────────────────────────────────────────────
# Seleção de melhores matches
# ──────────────────────────────────────────────

def _iou(a: BBox, b: BBox) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / (_bbox_area(a) + _bbox_area(b) - inter)


def _select_best_matches(
    central_objects: list[dict],
    candidates: list[DetectedObject],
    judgments: list[tuple[str | None, float]],
    iou_dedup: float = 0.25,
) -> list[DetectedObject]:
    """
    Para cada objeto central esperado, seleciona os N melhores candidatos
    que o Qwen julgou como correspondentes (N = count do objeto central).

    Ranking: score × (0.6 + 0.4 × centralidade).
    Dedup intra-label: se dois candidatos selecionados tiverem IoU > iou_dedup,
    mantém apenas o com maior ranking (o primeiro — já está ordenado).
    """
    # Agrupa candidatos aprovados por label
    approved: dict[str, list[tuple[float, DetectedObject]]] = {}
    for (match_label, score), candidate in zip(judgments, candidates):
        if match_label is None:
            continue
        approved.setdefault(match_label, []).append((score, candidate))

    result: list[DetectedObject] = []
    for central in central_objects:
        label = central["label"]
        count = central["count"]

        group = approved.get(label, [])
        if not group:
            print(f"Nenhum candidato aprovado para '{label}'")
            continue

        # Ordena por score × centralidade (melhor primeiro)
        ranked = sorted(
            group,
            key=lambda t: t[0] * (0.6 + 0.4 * _centrality(t[1].bbox)),
            reverse=True,
        )

        selected: list[DetectedObject] = []
        for score, candidate in ranked:
            if len(selected) >= count:
                break
            # Dedup: descarta se sobrepõe demais com já selecionado
            overlap = any(_iou(candidate.bbox, s.bbox) > iou_dedup for s in selected)
            if overlap:
                print(f"Dedup intra-label: '{label}' bbox descartado por sobreposição")
                continue
            selected.append(candidate)
            result.append(DetectedObject(
                label=label,
                score=round(score, 4),
                bbox=candidate.bbox,
                source="fused",
                yoloe_conf=candidate.yoloe_conf,
            ))
            print(f"Selecionado: '{label}' score={score:.2f} bbox={candidate.bbox}")

    return result


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.post("/detection/fused")
async def detection_fused(body: Prompt) -> dict:
    """
    Pipeline semântico principal — Qwen coordena, YOLOE guiado refina.

    Etapa 1 — Qwen analisa a cena:
      Identifica objetos centrais + quantidades + gera prompts em inglês.
      Ex: {"label": "carregador", "count": 1,
           "prompts": ["charger", "wall charger", "power adapter", ...]}

    Etapa 2 — YOLOE com text prompts:
      Usa os prompts ingleses do Qwen para guiar a detecção → bboxes precisas
      e semanticamente alinhadas com o que o Qwen identificou.

    Etapa 3 — Julgamento paralelo:
      Para cada crop do YOLOE, Qwen julga: corresponde a um objeto central?

    Etapa 4 — Seleção:
      Para cada objeto central (respeitando count), seleciona os N melhores
      candidatos aprovados, com dedup por IoU.
    """
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)

    loop = asyncio.get_event_loop()

    # ── Etapa 1: Qwen identifica cena + gera prompts ──
    async with httpx.AsyncClient(timeout=120.0) as client:
        scene_response = await _call_scene(client, image_b64)

    central_objects = parse_scene_response(scene_response)
    print(f"Objetos centrais: {central_objects}")

    if not central_objects:
        print("Qwen não identificou objetos centrais na cena.")
        return serialize_detections([], too_many=False)

    # Coleta todos os prompts em inglês gerados pelo Qwen
    all_prompts = []
    for obj in central_objects:
        for p in obj.get("prompts", []):
            if p not in all_prompts:
                all_prompts.append(p)

    print(f"Prompts para YOLOE: {all_prompts}")

    # ── Etapa 2: YOLOE guiado pelos prompts do Qwen ──
    yoloe_objects = await loop.run_in_executor(
        None, run_yoloe_prompted, img, all_prompts
    )
    print(f"YOLOE (prompted): {len(yoloe_objects)} candidato(s)")

    if not yoloe_objects:
        print("YOLOE não detectou candidatos com os prompts fornecidos.")
        return serialize_detections([], too_many=False)

    # Filtra por área
    candidates = [
        o for o in yoloe_objects
        if _MIN_AREA <= _bbox_area(o.bbox) <= _MAX_AREA
    ] or yoloe_objects

    print(f"Candidatos após filtro de área: {len(candidates)}")

    # ── Etapa 3: Qwen julga cada candidato em paralelo ──
    async with httpx.AsyncClient(timeout=120.0) as client:
        judge_tasks = [
            _call_judge(client, pil_to_b64(_crop_object(img, c.bbox)), central_objects, i)
            for i, c in enumerate(candidates)
        ]
        judge_responses = await asyncio.gather(*judge_tasks)

    judgments = [parse_judge_response(r) for r in judge_responses]

    approved_count = sum(1 for m, _ in judgments if m is not None)
    print(f"Candidatos aprovados pelo Qwen: {approved_count}/{len(candidates)}")

    # ── Etapa 4: Seleção dos melhores matches ──
    final = _select_best_matches(central_objects, candidates, judgments)

    print(f"Resultado final: {len(final)} objeto(s)")
    return serialize_detections(final)


@app.post("/detection")
async def detection(body: Prompt) -> dict:
    """Qwen autônomo — labels + bboxes gerados pelo Qwen, sem YOLOE."""
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        return serialize_detections(
            parse_qwen_response(
                await _call_vllm_legacy(client, image_b64, SYSTEM_INSTRUCTION)
            ),
            too_many=False,
        )


@app.post("/detection/sys_prompt")
async def detection_sys(body: PromptSys) -> dict:
    """Qwen com prompt customizado."""
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        return serialize_detections(
            parse_qwen_response(
                await _call_vllm_legacy(client, image_b64, body.prompt)
            ),
            too_many=False,
        )


@app.post("/detection/sequential")
async def detection_sequential(body: Prompt) -> dict:
    """Pipeline legado — YOLOE detecta tudo → Qwen refina e renomeia."""
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)
    loop = asyncio.get_event_loop()
    yoloe_objects = await loop.run_in_executor(None, run_yoloe, img)
    if not yoloe_objects:
        return serialize_detections([], too_many=False)
    async with httpx.AsyncClient(timeout=120.0) as client:
        vllm_response = await _call_vllm_sequential(client, img, yoloe_objects)
    return serialize_detections(parse_qwen_refine_response(vllm_response, yoloe_objects))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
