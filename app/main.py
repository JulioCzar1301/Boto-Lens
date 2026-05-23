"""
API principal para detecção de objetos.

Endpoints:
- /detection/fused  [PRINCIPAL] Qwen + YOLOE em paralelo → fusão IoU → dedup → ranking
- /detection        Qwen autônomo (apenas labels + bboxes do Qwen, sem YOLOE)
- /detection/sys_prompt  Qwen com prompt customizado
- /detection/sequential  Pipeline legado (YOLOE → Qwen)
- /health

Arquitetura /detection/fused:
  1. Qwen e YOLOE rodam simultaneamente e de forma independente.
  2. Qwen analisa a imagem completa → labels corretos em português + bboxes aproximadas.
  3. YOLOE detecta bboxes precisas (labels genéricos/errados ignorados).
  4. Fusão por IoU (fuse_by_iou):
       - IoU >= 0.15 entre Qwen e YOLOE → usa label/score do Qwen + bbox precisa do YOLOE.
       - Sem match de YOLOE → usa label + bbox do próprio Qwen como fallback.
  5. Deduplicação pós-fusão: contenção → mantém menor bbox; IoU > 0.20 → mantém maior score.
  6. Ranking final por score × (0.6 + 0.4 × centralidade).
  7. Retorna top 5.
"""

import math
import os
import asyncio
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from system_instruction import SYSTEM_INSTRUCTION, SEQUENTIAL_INSTRUCTION, CROP_INSTRUCTION
from models import DetectedObject, BBox
from services.yoloe import run_yoloe, get_yoloe
from services.qwen import parse_qwen_response, parse_qwen_refine_response, parse_crop_response
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

_MIN_AREA     = 0.005   # < 0.5 % da imagem → ruído
_MAX_AREA     = 0.75    # > 75 % da imagem → fundo
_CROP_PADDING = 0.05


# ──────────────────────────────────────────────
# Geometria
# ──────────────────────────────────────────────

def _bbox_area(bbox: BBox) -> float:
    return max(0.0, (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1))


def _intersection_area(a: BBox, b: BBox) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _containment(a: BBox, b: BBox) -> float:
    """Fração da área de A que está dentro de B."""
    area_a = _bbox_area(a)
    return 0.0 if area_a == 0 else _intersection_area(a, b) / area_a


def _centrality(bbox: BBox) -> float:
    """1.0 = centro da imagem, 0.0 = canto."""
    cx = (bbox.x1 + bbox.x2) / 2
    cy = (bbox.y1 + bbox.y2) / 2
    return max(0.0, 1.0 - 2 * max(abs(cx - 0.5), abs(cy - 0.5)))


def _deduplicate(objects: list[DetectedObject]) -> list[DetectedObject]:
    """
    Deduplicação pós-fusão:
    - Contenção > 50 %: mantém o bbox MENOR (mais preciso).
    - IoU > 0.20 sem contenção clara: mantém o de maior score.
    """
    sorted_objs = sorted(objects, key=lambda o: o.score, reverse=True)
    kept: list[DetectedObject] = []

    for obj in sorted_objs:
        discard = False
        to_remove: list[int] = []

        for i, k in enumerate(kept):
            obj_area = _bbox_area(obj.bbox)
            k_area   = _bbox_area(k.bbox)
            obj_in_k = _containment(obj.bbox, k.bbox)
            k_in_obj = _containment(k.bbox, obj.bbox)
            iou      = obj.bbox.iou(k.bbox)

            has_containment = obj_in_k > 0.5 or k_in_obj > 0.5

            if has_containment or iou > 0.20:
                if has_containment:
                    if obj_area < k_area:
                        to_remove.append(i)
                        print(f"Dedup contenção: '{k.label}' (area={k_area:.3f}) removido"
                              f" → mantém '{obj.label}' (area={obj_area:.3f})")
                    else:
                        discard = True
                        print(f"Dedup contenção: '{obj.label}' (area={obj_area:.3f}) descartado"
                              f" → mantém '{k.label}' (area={k_area:.3f})")
                        break
                else:
                    if obj.score > k.score:
                        to_remove.append(i)
                        print(f"Dedup IoU: '{k.label}' (score={k.score:.2f}) removido"
                              f" → mantém '{obj.label}' (score={obj.score:.2f})")
                    else:
                        discard = True
                        print(f"Dedup IoU: '{obj.label}' (score={obj.score:.2f}) descartado")
                        break

        if discard:
            continue
        for i in sorted(to_remove, reverse=True):
            kept.pop(i)
        kept.append(obj)

    return kept


def _rank(objects: list[DetectedObject], top_k: int = 5) -> list[DetectedObject]:
    """Ranking por score × bônus de centralidade, retorna top_k."""
    return sorted(
        objects,
        key=lambda o: o.score * (0.6 + 0.4 * _centrality(o.bbox)),
        reverse=True,
    )[:top_k]


# ──────────────────────────────────────────────
# Helpers de chamada ao vLLM
# ──────────────────────────────────────────────

async def _call_vllm(client: httpx.AsyncClient, image_b64: str, system: str) -> dict:
    """Chamada genérica ao Qwen com imagem completa."""
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
                            "text": (
                                "Analyze this image carefully. "
                                "For each foreground object, estimate a TIGHT bounding box "
                                "that fits the object precisely — do not include background."
                            ),
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
# Endpoints
# ──────────────────────────────────────────────

@app.post("/detection/fused")
async def detection_fused(body: Prompt) -> dict:
    """
    Pipeline principal — Qwen + YOLOE em paralelo com fusão IoU.

    1. Qwen e YOLOE rodam simultaneamente.
    2. Qwen → labels corretos + bboxes aproximadas (prompt melhorado para precisão).
    3. YOLOE → bboxes precisas (labels ignorados).
    4. Fusão por IoU: se IoU >= 0.15, usa label/score do Qwen + bbox do YOLOE.
       Caso contrário, mantém bbox do próprio Qwen (fallback).
    5. Filtra objetos com área < 0.5% ou > 75%.
    6. Deduplicação por contenção/IoU.
    7. Ranking por score × centralidade → top 5.
    """
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)

    loop = asyncio.get_event_loop()

    # Etapas 1–3: YOLOE e Qwen em paralelo
    async with httpx.AsyncClient(timeout=120.0) as client:
        yoloe_task = loop.run_in_executor(None, run_yoloe, img)
        qwen_task  = _call_vllm(client, image_b64, SYSTEM_INSTRUCTION)
        yoloe_objects, qwen_response = await asyncio.gather(yoloe_task, qwen_task)

    print(f"YOLOE: {len(yoloe_objects)} detecção(ões)")
    qwen_objects = parse_qwen_response(qwen_response)
    print(f"Qwen:  {len(qwen_objects)} detecção(ões)")

    if not qwen_objects:
        return serialize_detections([], too_many=False)

    # Etapa 4: fusão por IoU — YOLOE afina os bboxes do Qwen
    fused = fuse_by_iou(qwen_objects, yoloe_objects)

    # Etapa 5: filtro de área
    fused = [o for o in fused if _MIN_AREA <= _bbox_area(o.bbox) <= _MAX_AREA] or fused

    # Etapas 6–7: dedup + ranking
    deduped = _deduplicate(fused)
    final   = _rank(deduped)

    print(f"Resultado final: {len(final)} objeto(s)")
    return serialize_detections(final)


@app.post("/detection")
async def detection(body: Prompt) -> dict:
    """Qwen autônomo — labels + bboxes gerados pelo Qwen, sem YOLOE."""
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        return serialize_detections(
            parse_qwen_response(await _call_vllm(client, image_b64, SYSTEM_INSTRUCTION)),
            too_many=False,
        )


@app.post("/detection/sys_prompt")
async def detection_sys(body: PromptSys) -> dict:
    """Qwen com prompt customizado."""
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        return serialize_detections(
            parse_qwen_response(await _call_vllm(client, image_b64, body.prompt)),
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
