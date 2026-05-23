"""
API principal para detecção de objetos.

Endpoints:
- /detection/fused  [PRINCIPAL] Qwen + YOLOE-zero paralelo → fusão IoU → verificação de crop
- /detection        Qwen autônomo
- /detection/sys_prompt  Qwen com prompt customizado
- /detection/sequential  Pipeline legado
- /health

Arquitetura /detection/fused:

  Etapa 1 (paralelo):
    a) Qwen detecta objetos + gera bboxes na imagem completa.
    b) YOLOE-zero detecta candidatos com bboxes precisas.

  Etapa 2 — Filtros pré-fusão:
    - Área: remove objetos < 1.5% (ruído/fundo) ou > 75% (cena inteira).
    - Centralidade + área combinadas: objetos muito pequenos E periféricos
      provavelmente não são o foco do usuário.

  Etapa 3 — Fusão IoU (Qwen coordena, YOLOE refina):
    - IoU >= 0.15: label/score do Qwen + bbox precisa do YOLOE (source="fused").
    - Sem match: label/score do Qwen + bbox do próprio Qwen (source="qwen").

  Etapa 4 — Verificação final por crop:
    Para cada objeto fundido, envia o crop ao Qwen:
    - Qwen confirma/corrige o label e retorna score de confiança.
    - score < 0.6 → objeto descartado.

  Etapa 5 — Remove sub-partes:
    Descarta bboxes cujo interior (>60%) está contido num bbox maior.
    Ex: "teclado" dentro de "laptop" → descartado.

  Etapa 6 — Ranking final:
    score × (0.6 + 0.4 × centralidade) → top 5.
"""

import json
import math
import os
import asyncio
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from system_instruction import SYSTEM_INSTRUCTION, VERIFY_INSTRUCTION, SEQUENTIAL_INSTRUCTION
from models import DetectedObject, BBox
from services.yoloe import run_yoloe, get_yoloe
from services.qwen import parse_qwen_response, parse_verify_response, parse_qwen_refine_response
from services.fusion import fuse_by_iou, serialize_detections
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
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")


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

_MIN_FOREGROUND_AREA = 0.015   # 1.5 % — abaixo disso é ruído/fundo
_MAX_FOREGROUND_AREA = 0.75    # 75 % — acima disso é a cena inteira
_SMALL_AREA          = 0.03    # 3 % — pequeno mas não microscópico
_SMALL_CENTRALITY    = 0.25    # muito periférico (canto)
_CROP_PADDING        = 0.05
_IOI_THRESHOLD       = 0.15    # IoU mínimo para match Qwen↔YOLOE
_SUBPART_CONTAINMENT = 0.60    # >60% de A dentro de B → A é sub-parte de B


# ──────────────────────────────────────────────
# Geometria
# ──────────────────────────────────────────────

def _bbox_area(bbox: BBox) -> float:
    return max(0.0, (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1))


def _centrality(bbox: BBox) -> float:
    """1.0 = centro, 0.0 = canto."""
    cx = (bbox.x1 + bbox.x2) / 2
    cy = (bbox.y1 + bbox.y2) / 2
    return max(0.0, 1.0 - 2 * max(abs(cx - 0.5), abs(cy - 0.5)))


def _iou(a: BBox, b: BBox) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / (_bbox_area(a) + _bbox_area(b) - inter)


def _containment(a: BBox, b: BBox) -> float:
    """Fração da área de A que está dentro de B."""
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = _bbox_area(a)
    return inter / area_a if area_a > 0 else 0.0


def _is_foreground(obj: DetectedObject) -> bool:
    """
    Retorna True se o objeto deve ser considerado foreground relevante.
    Descarta:
      - Objetos muito pequenos (< 1.5% da imagem) → ruído/detalhe de fundo
      - Objetos que cobrem quase tudo (> 75%) → fundo/cena inteira
      - Objetos pequenos (< 3%) E muito periféricos → foco secundário improvável
    """
    area = _bbox_area(obj.bbox)
    if area < _MIN_FOREGROUND_AREA:
        return False
    if area > _MAX_FOREGROUND_AREA:
        return False
    if area < _SMALL_AREA and _centrality(obj.bbox) < _SMALL_CENTRALITY:
        return False
    return True


def _remove_subparts(objects: list[DetectedObject]) -> list[DetectedObject]:
    """
    Remove detecções que são sub-partes de outros objetos na mesma cena.

    Regra: containment(A em B) > 0.60 E area(A) < area(B) → descarta A.
    Exemplo: "teclado" bbox dentro do "laptop" bbox → descarta "teclado".
    """
    to_remove: set[int] = set()
    for i, a in enumerate(objects):
        if i in to_remove:
            continue
        for j, b in enumerate(objects):
            if i == j or j in to_remove:
                continue
            if (_containment(a.bbox, b.bbox) > _SUBPART_CONTAINMENT
                    and _bbox_area(a.bbox) < _bbox_area(b.bbox)):
                to_remove.add(i)
                print(f"Sub-parte: '{a.label}' ({_bbox_area(a.bbox):.3f}) "
                      f"descartado — está dentro de '{b.label}' ({_bbox_area(b.bbox):.3f})")
                break
    return [o for i, o in enumerate(objects) if i not in to_remove]


def _crop_object(img: Image.Image, bbox: BBox) -> Image.Image:
    """Crop com padding de 5%, sem ultrapassar os limites da imagem."""
    w, h = img.size
    pad_x = (bbox.x2 - bbox.x1) * _CROP_PADDING
    pad_y = (bbox.y2 - bbox.y1) * _CROP_PADDING
    x1 = max(0.0, bbox.x1 - pad_x)
    y1 = max(0.0, bbox.y1 - pad_y)
    x2 = min(1.0, bbox.x2 + pad_x)
    y2 = min(1.0, bbox.y2 + pad_y)
    return img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))


def _bbox_prominence(bbox: BBox) -> float:
    area = _bbox_area(bbox)
    area_score = min(1.0, math.sqrt(area) * 2)
    cx = (bbox.x1 + bbox.x2) / 2
    cy = (bbox.y1 + bbox.y2) / 2
    centrality = max(0.0, 1.0 - 2 * max(abs(cx - 0.5), abs(cy - 0.5)))
    return round(0.6 * area_score + 0.4 * centrality, 4)


# ──────────────────────────────────────────────
# Chamadas ao vLLM
# ──────────────────────────────────────────────

async def _call_qwen(client: httpx.AsyncClient, image_b64: str, system: str) -> dict:
    """Chamada ao Qwen com imagem completa."""
    response = await client.post(
        VLLM_URL,
        json={
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Detect all visible foreground objects with tight bounding boxes.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        },
    )
    result = response.json()
    print(f"[Qwen] status={response.status_code}")
    return result


async def _call_verify(
    client: httpx.AsyncClient,
    crop_b64: str,
    hint_label: str,
) -> dict:
    """
    Etapa 4: Qwen verifica o label de um crop já fundido.
    Recebe o crop isolado + hint do label atual para contexto.
    """
    response = await client.post(
        VLLM_URL,
        json={
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": VERIFY_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f'This object was initially labeled "{hint_label}". '
                                "Verify if that is correct, or correct it. "
                                "What is this object?"
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
    print(f"[Verify] status={response.status_code} | hint='{hint_label}'")
    return result


async def _call_vllm_sequential(
    client: httpx.AsyncClient,
    img: Image.Image,
    yoloe_objects: list[DetectedObject],
) -> dict:
    import json as _json
    filtered = [
        (i, obj) for i, obj in enumerate(yoloe_objects)
        if _bbox_area(obj.bbox) >= _MIN_FOREGROUND_AREA
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
# Endpoints
# ──────────────────────────────────────────────

@app.post("/detection/fused")
async def detection_fused(body: Prompt) -> dict:
    """
    Pipeline principal: Qwen + YOLOE-zero paralelo → fusão IoU → verificação de crop.
    """
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)
    img_size = img.size  # (width, height) para normalizar coords pixel

    loop = asyncio.get_event_loop()

    # ── Etapa 1: Qwen e YOLOE em paralelo ──
    async with httpx.AsyncClient(timeout=120.0) as client:
        qwen_task  = _call_qwen(client, image_b64, SYSTEM_INSTRUCTION)
        yoloe_task = loop.run_in_executor(None, run_yoloe, img)
        qwen_response, yoloe_objects = await asyncio.gather(qwen_task, yoloe_task)

    qwen_objects = parse_qwen_response(qwen_response, img_size=img_size)
    print(f"Qwen:  {len(qwen_objects)} objeto(s)")
    print(f"YOLOE: {len(yoloe_objects)} candidato(s)")

    if not qwen_objects:
        return serialize_detections([], too_many=False)

    # ── Etapa 2: Filtros de foreground ──
    qwen_filtered = [o for o in qwen_objects if _is_foreground(o)]
    if not qwen_filtered:
        print("Todos os objetos do Qwen foram filtrados — usando todos")
        qwen_filtered = qwen_objects

    yoloe_filtered = [o for o in yoloe_objects if _is_foreground(o)]
    print(f"Após filtro foreground — Qwen: {len(qwen_filtered)}, YOLOE: {len(yoloe_filtered)}")

    # ── Etapa 3: Fusão IoU ──
    fused = fuse_by_iou(qwen_filtered, yoloe_filtered, iou_threshold=_IOI_THRESHOLD)
    print(f"Pós-fusão: {len(fused)} objeto(s)")

    # ── Etapa 4: Verificação de crop pelo Qwen (paralelo) ──
    async with httpx.AsyncClient(timeout=120.0) as client:
        verify_tasks = [
            _call_verify(client, pil_to_b64(_crop_object(img, obj.bbox)), obj.label)
            for obj in fused
        ]
        verify_responses = await asyncio.gather(*verify_tasks)

    verified: list[DetectedObject] = []
    for obj, vresp in zip(fused, verify_responses):
        label, score = parse_verify_response(vresp)
        if not label:
            print(f"Descartado na verificação: '{obj.label}' (score baixo)")
            continue
        verified.append(DetectedObject(
            label=label,
            score=score,
            bbox=obj.bbox,
            source=obj.source,
            yoloe_conf=obj.yoloe_conf,
        ))

    print(f"Após verificação: {len(verified)} objeto(s)")

    # ── Etapa 5: Remove sub-partes ──
    verified = _remove_subparts(verified)
    print(f"Após remoção de sub-partes: {len(verified)} objeto(s)")

    # ── Etapa 6: Ranking final ──
    final = sorted(
        verified,
        key=lambda o: o.score * (0.6 + 0.4 * _centrality(o.bbox)),
        reverse=True,
    )[:5]

    print(f"Resultado final: {len(final)} objeto(s)")
    return serialize_detections(final)


@app.post("/detection")
async def detection(body: Prompt) -> dict:
    """Qwen autônomo — labels + bboxes gerados pelo Qwen, sem YOLOE."""
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)
    async with httpx.AsyncClient(timeout=120.0) as client:
        return serialize_detections(
            parse_qwen_response(
                await _call_qwen(client, image_b64, SYSTEM_INSTRUCTION),
                img_size=img.size,
            ),
            too_many=False,
        )


@app.post("/detection/sys_prompt")
async def detection_sys(body: PromptSys) -> dict:
    """Qwen com prompt customizado."""
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)
    async with httpx.AsyncClient(timeout=120.0) as client:
        return serialize_detections(
            parse_qwen_response(
                await _call_qwen(client, image_b64, body.prompt),
                img_size=img.size,
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
