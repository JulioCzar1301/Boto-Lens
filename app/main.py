"""
API principal para detecção de objetos.

Endpoints disponíveis:
- /detection: Apenas Qwen (detecção autônoma, sem YOLOE)
- /detection/sys_prompt: Qwen com prompt customizado
- /detection/fused: Pipeline paralelo — YOLOE + Qwen rodam ao mesmo tempo, fusão por IoU
- /detection/sequential: Pipeline legado — YOLOE detecta tudo -> Qwen refina e nomeia
- /health: Health check

Arquitetura do pipeline /detection/fused (paralelo):
  1. YOLOE e Qwen rodam simultaneamente e de forma independente.
  2. Qwen analisa a imagem livremente e gera labels corretos + bboxes aproximadas.
  3. YOLOE gera bboxes precisas para os objetos que consegue detectar.
  4. Fusão por IoU:
     - Qwen + YOLOE concordam (IoU >= threshold) -> label do Qwen + bbox do YOLOE.
     - Qwen detectou algo que YOLOE não viu -> label + bbox do Qwen (fallback).
  Isso garante que objetos fora do vocabulário do YOLOE (ex: carregador) não sejam perdidos.
"""

import math
import os
import asyncio
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from system_instruction import SYSTEM_INSTRUCTION, SEQUENTIAL_INSTRUCTION
from models import DetectedObject, BBox
from services.yoloe import run_yoloe, get_yoloe
from services.qwen import parse_qwen_response, parse_qwen_refine_response
from services.fusion import fuse_by_iou, serialize_detections
from utils.image import (
    resize_base64_image,
    b64_to_pil,
    pil_to_b64,
)

app = FastAPI()


# ----------------------------------------------
# Startup Events
# ----------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Carrega o modelo YOLO durante a inicialização da API."""
    print("Inicializando... Carregando modelo YOLO...")
    try:
        get_yoloe()
        print("Modelo YOLO carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo YOLO: {e}")
        raise

VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000") + "/v1/chat/completions"


# ----------------------------------------------
# Schemas (Pydantic)
# ----------------------------------------------

class Prompt(BaseModel):
    """Payload para detecção sem prompt customizado."""
    image: str  # Base64 da imagem


class PromptSys(BaseModel):
    """Payload para detecção com prompt customizado."""
    image: str  # Base64 da imagem
    prompt: str


# ----------------------------------------------
# Helpers
# ----------------------------------------------

async def _call_vllm(client: httpx.AsyncClient, image_b64: str, system: str) -> dict:
    """
    Realiza chamada para o vLLM (Qwen) com a imagem e instrução de sistema.

    Args:
        client: Cliente HTTP assíncrono.
        image_b64: Imagem em base64.
        system: Instrução de sistema para o modelo.

    Returns:
        dict: Resposta JSON do vLLM.
    """
    response = await client.post(
        VLLM_URL,
        json={
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Descreva essa imagem"},
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
    print(f"vLLM Response Status: {response.status_code}")
    print(f"vLLM Response: {result}")
    return result


# Fração mínima da área da imagem para um objeto ser enviado ao Qwen.
_MIN_AREA_FRACTION = 0.005


def _bbox_area(bbox: BBox) -> float:
    """Retorna a área normalizada da bounding box (0-1)."""
    return max(0.0, (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1))


def _bbox_prominence(bbox: BBox) -> float:
    """
    Calcula a proeminência espacial do objeto: combina tamanho relativo e centralidade.
    """
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
    """
    Etapa 2 do pipeline sequencial YOLOE -> Qwen (arquitetura clean-image + JSON).
    """
    import json as _json

    filtered: list[tuple[int, DetectedObject]] = [
        (i, obj)
        for i, obj in enumerate(yoloe_objects)
        if _bbox_area(obj.bbox) >= _MIN_AREA_FRACTION
    ]
    if not filtered:
        filtered = list(enumerate(yoloe_objects))

    clean_b64 = pil_to_b64(img)
    yoloe_payload = _json.dumps(
        [
            {
                "index": original_idx,
                "label": obj.label,
                "conf": round(obj.yoloe_conf or obj.score, 4),
                "bbox_norm": {
                    "x1": round(obj.bbox.x1, 4),
                    "y1": round(obj.bbox.y1, 4),
                    "x2": round(obj.bbox.x2, 4),
                    "y2": round(obj.bbox.y2, 4),
                },
                "area": round(_bbox_area(obj.bbox), 4),
                "prominence": _bbox_prominence(obj.bbox),
            }
            for original_idx, obj in filtered
        ],
        ensure_ascii=False,
    )

    response = await client.post(
        VLLM_URL,
        json={
            "model": "Qwen/Qwen2.5-VL-32B-Instruct",
            "messages": [
                {"role": "system", "content": SEQUENTIAL_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "The image below is the original photo with NO annotations drawn on it.\n\n"
                                "The JSON list below contains objects detected by YOLOE-zero. "
                                "Each entry includes 'bbox_norm' with normalized coordinates (0-1) "
                                "that tell you exactly where each object is located in the image. "
                                "Use these coordinates to locate each object visually in the clean photo.\n\n"
                                f"YOLOE detections:\n{yoloe_payload}\n\n"
                                "Select the most visually significant objects. "
                                "Rename them correctly in Brazilian Portuguese."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}"},
                        },
                    ],
                },
            ],
        },
    )
    result = response.json()
    print(f"vLLM Sequential Response Status: {response.status_code}")
    print(f"vLLM Sequential Response: {result}")
    return result


# ----------------------------------------------
# Endpoints
# ----------------------------------------------

@app.post("/detection")
async def detection(body: Prompt) -> dict:
    """Endpoint que retorna a resposta bruta do Qwen (sem fusão)."""
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        qwen_response = await _call_vllm(client, image_b64, SYSTEM_INSTRUCTION)
        qwen_objects = parse_qwen_response(qwen_response)
        return serialize_detections(qwen_objects, too_many=False)


@app.post("/detection/sys_prompt")
async def detection_sys(body: PromptSys) -> dict:
    """Endpoint que permite enviar um prompt customizado para o Qwen."""
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        qwen_response = await _call_vllm(client, image_b64, body.prompt)
        qwen_objects = parse_qwen_response(qwen_response)
        return serialize_detections(qwen_objects, too_many=False)


@app.post("/detection/fused")
async def detection_fused(body: Prompt) -> dict:
    """
    Endpoint principal - pipeline paralelo YOLOE + Qwen com fusão por IoU.

    Fluxo:
        1. YOLOE e Qwen rodam em paralelo e de forma independente.
        2. Qwen analisa a imagem livremente -> gera labels corretos + bboxes aproximadas.
        3. YOLOE gera bboxes precisas (mesmo que com labels errados do COCO).
        4. Fusão por IoU:
           - Qwen + YOLOE concordam (IoU >= threshold) -> label do Qwen + bbox do YOLOE.
           - Qwen detectou algo que YOLOE não viu -> label + bbox do Qwen (fallback).
        Assim, objetos fora do vocabulário do YOLOE (ex: carregador) não são perdidos.
    """
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)
    loop = asyncio.get_event_loop()

    async with httpx.AsyncClient(timeout=120.0) as client:
        yoloe_task = loop.run_in_executor(None, run_yoloe, img)
        qwen_task = _call_vllm(client, image_b64, SYSTEM_INSTRUCTION)
        yoloe_objects, qwen_response = await asyncio.gather(yoloe_task, qwen_task)

    print(f"YOLOE: {len(yoloe_objects)} objeto(s) detectado(s)")
    qwen_objects = parse_qwen_response(qwen_response)
    print(f"Qwen: {len(qwen_objects)} objeto(s) detectado(s)")

    if not qwen_objects:
        return serialize_detections([], too_many=False)

    # Fusão por IoU: YOLOE afina as bboxes do Qwen quando possível
    fused = fuse_by_iou(qwen_objects, yoloe_objects)
    return serialize_detections(fused)


@app.post("/detection/sequential")
async def detection_sequential(body: Prompt) -> dict:
    """
    Pipeline legado - sequencial YOLOE -> Qwen.
    Limitação: se YOLOE não detectar um objeto, Qwen não o vê.
    Mantido para fins de comparação.
    """
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)
    loop = asyncio.get_event_loop()
    yoloe_objects = await loop.run_in_executor(None, run_yoloe, img)

    if not yoloe_objects:
        return serialize_detections([], too_many=False)

    print(f"YOLOE: {len(yoloe_objects)} objeto(s) detectado(s)")

    async with httpx.AsyncClient(timeout=120.0) as client:
        vllm_response = await _call_vllm_sequential(client, img, yoloe_objects)

    refined_objects = parse_qwen_refine_response(vllm_response, yoloe_objects)
    return serialize_detections(refined_objects)


@app.get("/health")
async def health() -> dict:
    """Endpoint de health check."""
    return {"status": "ok"}
