"""
API principal para detecção de objetos.

Endpoints disponíveis:
- /detection/fused   [PRINCIPAL] YOLOE → filtro de contenção → crop → Qwen paralelo
- /detection         Qwen autônomo (sem YOLOE)
- /detection/sys_prompt  Qwen com prompt customizado
- /detection/sequential  Pipeline legado YOLOE → Qwen
- /health

Arquitetura do pipeline /detection/fused:
  1. YOLOE detecta bboxes.
  2. Filtra por área mínima (ruído) e máxima (fundo muito grande).
  3. Regra de contenção: se bbox A está ≥70% dentro de bbox B e é menor,
     A é sub-parte de B (ex: pino de carregador dentro do carregador) → descartado.
  4. Crop de cada candidato com padding.
  5. Todos os crops enviados ao Qwen em paralelo via asyncio.gather.
  6. Qwen classifica cada crop individualmente:
     - score < 0.6 → descartado (fundo, sub-parte, múltiplos objetos no crop)
  7. Deduplicação pós-Qwen: IoU > 0.3 entre resultados → fica o de maior score.
  8. Retorna top 5 por score.
"""

import math
import os
import asyncio
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from system_instruction import CROP_INSTRUCTION, SYSTEM_INSTRUCTION, SEQUENTIAL_INSTRUCTION
from models import DetectedObject, BBox
from services.yoloe import run_yoloe, get_yoloe
from services.qwen import parse_crop_response, parse_qwen_response, parse_qwen_refine_response
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

_MIN_AREA      = 0.005   # < 0.5% da imagem → ruído
_MAX_AREA      = 0.75    # > 75% da imagem → provavelmente fundo
_CROP_PADDING  = 0.05    # padding ao redor do crop
_CONTAINMENT_T = 0.70    # A está ≥70% dentro de B → A é sub-parte de B
_IOU_DEDUP_T   = 0.30    # IoU pós-Qwen: > 0.3 → mantém o de maior score


# ──────────────────────────────────────────────
# Funções auxiliares de geometria
# ──────────────────────────────────────────────

def _bbox_area(bbox: BBox) -> float:
    return max(0.0, (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1))


def _intersection_area(a: BBox, b: BBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _containment(a: BBox, b: BBox) -> float:
    """Fração da área de A que está dentro de B (0–1)."""
    area_a = _bbox_area(a)
    if area_a == 0:
        return 0.0
    return _intersection_area(a, b) / area_a


def _filter_contained(objects: list[DetectedObject]) -> list[DetectedObject]:
    """
    Remove bboxes que são sub-partes de outras bboxes maiores.

    Regra: se A está ≥70% contido em B e area(A) < area(B),
    A é provavelmente um componente de B (ex: pino do carregador dentro do carregador)
    e deve ser descartado antes de ir ao Qwen.
    """
    to_remove = set()
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j or j in to_remove:
                continue
            if _containment(a.bbox, b.bbox) >= _CONTAINMENT_T and \
               _bbox_area(a.bbox) < _bbox_area(b.bbox):
                to_remove.add(i)
                print(f"Contenção: objeto {i} (área={_bbox_area(a.bbox):.3f}) "
                      f"descartado por estar dentro de {j} (área={_bbox_area(b.bbox):.3f})")
                break
    return [obj for i, obj in enumerate(objects) if i not in to_remove]


def _deduplicate_results(objects: list[DetectedObject]) -> list[DetectedObject]:
    """
    Remove detecções sobrepostas após a classificação pelo Qwen.

    Se dois resultados têm IoU > 0.3, mantém apenas o de maior score.
    (Garante que a mesa com carregador sobrepostos, o carregador preciso vença.)
    """
    sorted_objs = sorted(objects, key=lambda o: o.score, reverse=True)
    kept: list[DetectedObject] = []
    for obj in sorted_objs:
        overlap = any(obj.bbox.iou(k.bbox) > _IOU_DEDUP_T for k in kept)
        if not overlap:
            kept.append(obj)
        else:
            print(f"Dedup pós-Qwen: '{obj.label}' (score={obj.score:.2f}) "
                  f"descartado por sobreposição com resultado de maior score")
    return kept


# ──────────────────────────────────────────────
# Crop e chamada ao Qwen
# ──────────────────────────────────────────────

def _crop_object(img: Image.Image, bbox: BBox) -> Image.Image:
    """Recorta a região da bbox com padding, sem sair dos limites da imagem."""
    w, h = img.size
    pad_x = (bbox.x2 - bbox.x1) * _CROP_PADDING
    pad_y = (bbox.y2 - bbox.y1) * _CROP_PADDING
    x1 = max(0.0, bbox.x1 - pad_x)
    y1 = max(0.0, bbox.y1 - pad_y)
    x2 = min(1.0, bbox.x2 + pad_x)
    y2 = min(1.0, bbox.y2 + pad_y)
    return img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))


async def _classify_crop(
    client: httpx.AsyncClient,
    img: Image.Image,
    obj: DetectedObject,
) -> DetectedObject | None:
    """
    Envia o crop de um objeto ao Qwen para classificação individual.

    Descarta se:
    - score < 60% (fundo, sub-parte, ou múltiplos objetos no crop)
    - erro na resposta
    """
    crop = _crop_object(img, obj.bbox)
    crop_b64 = pil_to_b64(crop)

    try:
        response = await client.post(
            VLLM_URL,
            json={
                "model": VLLM_MODEL,
                "messages": [
                    {"role": "system", "content": CROP_INSTRUCTION},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
                            {"type": "text", "text": "O que é este objeto?"},
                        ],
                    },
                ],
                "max_tokens": 64,
            },
        )
        result = response.json()
    except Exception as e:
        print(f"Erro ao chamar vLLM para crop: {e}")
        return None

    label, score = parse_crop_response(result)
    if not label:
        print(f"Crop descartado: score={score:.2f}, bbox={obj.bbox}")
        return None

    print(f"Crop classificado: '{label}' score={score:.2f}")
    return DetectedObject(
        label=label,
        score=score,
        bbox=obj.bbox,
        source="fused",
        yoloe_conf=obj.yoloe_conf,
    )


async def _call_vllm(client: httpx.AsyncClient, image_b64: str, system: str) -> dict:
    response = await client.post(
        VLLM_URL,
        json={
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": "Descreva essa imagem"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]},
            ],
        },
    )
    return response.json()


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
        if _bbox_area(obj.bbox) >= 0.005
    ] or list(enumerate(yoloe_objects))

    clean_b64 = pil_to_b64(img)
    yoloe_payload = _json.dumps(
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
                    {"type": "text", "text": (
                        f"YOLOE detections:\n{yoloe_payload}\n\n"
                        "Select the most prominent foreground objects and rename in Brazilian Portuguese."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}"}},
                ]},
            ],
        },
    )
    return response.json()


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.post("/detection/fused")
async def detection_fused(body: Prompt) -> dict:
    """
    Pipeline principal: YOLOE → contenção → crop → Qwen paralelo → dedup.

    Filtros aplicados:
    1. Área: 0.5% ≤ área ≤ 75% da imagem
    2. Contenção: bbox ≥70% dentro de outra maior → sub-parte → descartado
    3. Qwen score < 60% → fundo, sub-parte ou múltiplos objetos no crop → descartado
    4. IoU pós-Qwen > 0.3 → mantém o de maior score
    """
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)

    loop = asyncio.get_event_loop()
    yoloe_objects = await loop.run_in_executor(None, run_yoloe, img)

    if not yoloe_objects:
        return serialize_detections([], too_many=False)

    # Filtro 1: área
    candidates = [
        obj for obj in yoloe_objects
        if _MIN_AREA <= _bbox_area(obj.bbox) <= _MAX_AREA
    ] or yoloe_objects

    # Filtro 2: contenção (remove sub-partes)
    candidates = _filter_contained(candidates)

    print(f"YOLOE: {len(yoloe_objects)} → após filtros: {len(candidates)} candidatos")

    if not candidates:
        return serialize_detections([], too_many=False)

    # Filtro 3: Qwen em paralelo por crop
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [_classify_crop(client, img, obj) for obj in candidates]
        results = await asyncio.gather(*tasks)

    classified = [r for r in results if r is not None]

    # Filtro 4: deduplicação pós-Qwen por IoU
    final = _deduplicate_results(classified)
    final = final[:5]

    print(f"Resultado final: {len(final)} objeto(s)")
    return serialize_detections(final)


@app.post("/detection")
async def detection(body: Prompt) -> dict:
    """Qwen autônomo sem YOLOE."""
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        qwen_response = await _call_vllm(client, image_b64, SYSTEM_INSTRUCTION)
        return serialize_detections(parse_qwen_response(qwen_response), too_many=False)


@app.post("/detection/sys_prompt")
async def detection_sys(body: PromptSys) -> dict:
    """Qwen com prompt customizado."""
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        qwen_response = await _call_vllm(client, image_b64, body.prompt)
        return serialize_detections(parse_qwen_response(qwen_response), too_many=False)


@app.post("/detection/sequential")
async def detection_sequential(body: Prompt) -> dict:
    """Pipeline legado YOLOE → Qwen refina lista completa."""
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)
    loop = asyncio.get_event_loop()
    yoloe_objects = await loop.run_in_executor(None, run_yoloe, img)
    if not yoloe_objects:
        return serialize_detections([], too_many=False)
    async with httpx.AsyncClient(timeout=120.0) as client:
        vllm_response = await _call_vllm_sequential(client, img, yoloe_objects)
    return serialize_detections(
        parse_qwen_refine_response(vllm_response, yoloe_objects)
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
