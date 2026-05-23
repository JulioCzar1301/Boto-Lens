"""
API principal para detecção de objetos.

Endpoints disponíveis:
- /detection/fused   [PRINCIPAL] YOLOE detecta bboxes → crop por objeto → Qwen classifica em paralelo
- /detection         Qwen autônomo (sem YOLOE)
- /detection/sys_prompt  Qwen com prompt customizado
- /detection/sequential  Pipeline legado YOLOE → Qwen (mantido para comparação)
- /health

Arquitetura do pipeline /detection/fused:
  1. YOLOE-zero detecta todos os objetos → bboxes precisas.
  2. Filtra objetos por área mínima (ruído) e máxima (fundo muito grande).
  3. Para cada bbox, faz um crop da imagem original com padding.
  4. Todos os crops são enviados ao Qwen em paralelo via asyncio.gather.
  5. Qwen classifica cada crop individualmente → label em PT-BR + score próprio.
  6. Crops com score < 60% são descartados (background ou incerteza alta).
  7. Retorna os até 5 objetos com maior score.
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
# Constantes de filtragem
# ──────────────────────────────────────────────

# Objetos menores que 0.5% da imagem → ruído
_MIN_AREA = 0.005
# Objetos maiores que 75% da imagem → provavelmente fundo (parede, chão, mesa inteira)
_MAX_AREA = 0.75
# Padding ao redor do crop (fração do tamanho da bbox)
_CROP_PADDING = 0.05


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _bbox_area(bbox: BBox) -> float:
    return max(0.0, (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1))


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
    Retorna None se o score for < 60% (background ou incerteza alta).
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
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"},
                            },
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
        print(f"Crop descartado (score={score:.2f}): bbox={obj.bbox}")
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
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Descreva essa imagem"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                },
            ],
        },
    )
    result = response.json()
    print(f"vLLM Response Status: {response.status_code}")
    return result


_MIN_AREA_FRACTION = 0.005


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
        if _bbox_area(obj.bbox) >= _MIN_AREA_FRACTION
    ] or list(enumerate(yoloe_objects))

    clean_b64 = pil_to_b64(img)
    yoloe_payload = _json.dumps(
        [{
            "index": i,
            "label": obj.label,
            "conf": round(obj.yoloe_conf or obj.score, 4),
            "bbox_norm": {"x1": round(obj.bbox.x1, 4), "y1": round(obj.bbox.y1, 4),
                          "x2": round(obj.bbox.x2, 4), "y2": round(obj.bbox.y2, 4)},
            "area": round(_bbox_area(obj.bbox), 4),
            "prominence": _bbox_prominence(obj.bbox),
        } for i, obj in filtered],
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
                        "Original photo (no annotations).\n\n"
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
    Pipeline principal: YOLOE → crop por objeto → Qwen classifica em paralelo.

    1. YOLOE detecta bboxes (labels ignorados).
    2. Filtra por área mínima (ruído) e máxima (fundo).
    3. Crop de cada objeto com padding.
    4. Todos os crops enviados ao Qwen em paralelo.
    5. Qwen dá label em PT-BR + score próprio por crop.
    6. Score < 60% → descartado.
    7. Retorna os top 5 por score.
    """
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)

    # Etapa 1: YOLOE detecta bboxes
    loop = asyncio.get_event_loop()
    yoloe_objects = await loop.run_in_executor(None, run_yoloe, img)

    if not yoloe_objects:
        return serialize_detections([], too_many=False)

    # Etapa 2: filtra por área (remove ruído e objetos gigantes de fundo)
    candidates = [
        obj for obj in yoloe_objects
        if _MIN_AREA <= _bbox_area(obj.bbox) <= _MAX_AREA
    ]

    if not candidates:
        candidates = yoloe_objects  # fallback sem filtro

    print(f"YOLOE: {len(yoloe_objects)} detecções → {len(candidates)} candidatos após filtro de área")

    # Etapa 3+4: crop + classificação Qwen em paralelo
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [_classify_crop(client, img, obj) for obj in candidates]
        results = await asyncio.gather(*tasks)

    # Etapa 5: filtra None (score < 60%) e pega top 5
    final = [r for r in results if r is not None]
    final = sorted(final, key=lambda o: o.score, reverse=True)[:5]

    print(f"Resultado final: {len(final)} objeto(s) com score >= 60%")
    return serialize_detections(final)


@app.post("/detection")
async def detection(body: Prompt) -> dict:
    """Qwen autônomo (sem YOLOE) — gera bbox + label direto."""
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        qwen_response = await _call_vllm(client, image_b64, SYSTEM_INSTRUCTION)
        qwen_objects = parse_qwen_response(qwen_response)
        return serialize_detections(qwen_objects, too_many=False)


@app.post("/detection/sys_prompt")
async def detection_sys(body: PromptSys) -> dict:
    """Qwen com prompt customizado."""
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        qwen_response = await _call_vllm(client, image_b64, body.prompt)
        qwen_objects = parse_qwen_response(qwen_response)
        return serialize_detections(qwen_objects, too_many=False)


@app.post("/detection/sequential")
async def detection_sequential(body: Prompt) -> dict:
    """Pipeline legado: YOLOE → Qwen refina lista completa. Mantido para comparação."""
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)
    loop = asyncio.get_event_loop()
    yoloe_objects = await loop.run_in_executor(None, run_yoloe, img)

    if not yoloe_objects:
        return serialize_detections([], too_many=False)

    async with httpx.AsyncClient(timeout=120.0) as client:
        vllm_response = await _call_vllm_sequential(client, img, yoloe_objects)

    refined = parse_qwen_refine_response(vllm_response, yoloe_objects)
    return serialize_detections(refined)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
