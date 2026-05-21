"""
API principal para detecção de objetos.

Endpoints disponíveis:
- /detection: Apenas Qwen (detecção autônoma, sem YOLOE)
- /detection/sys_prompt: Qwen com prompt customizado
- /detection/fused: Pipeline sequencial — YOLOE detecta tudo → Qwen refina e nomeia
- /health: Health check

Arquitetura do pipeline /detection/fused:
  1. YOLOE-zero detecta todos os objetos → bboxes precisas + labels brutos.
  2. Os objetos são filtrados por área mínima (remove ruídos muito pequenos).
  3. A imagem é anotada com bounding boxes NUMERADAS (índice original de cada objeto).
  4. O Qwen recebe a imagem anotada + JSON com {index, label, conf, bbox_norm, area,
     prominence} e seleciona os objetos mais significantes, renomeando-os em português.
  5. As coordenadas finais são SEMPRE as do YOLOE (preservadas pelo yoloe_index).
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
from services.fusion import serialize_detections
from utils.image import (
    resize_base64_image,
    b64_to_pil,
    draw_yoloe_detections,
    draw_indexed_detections,
    pil_to_b64,
)

app = FastAPI()


# ──────────────────────────────────────────────
# Startup Events
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Carrega o modelo YOLO durante a inicialização da API."""
    print("🚀 Inicializando... Carregando modelo YOLO...")
    try:
        get_yoloe()  # Carrega o modelo YOLO na memória
        print("✅ Modelo YOLO carregado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo YOLO: {e}")
        raise

VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000") + "/v1/chat/completions"


# ──────────────────────────────────────────────
# Schemas (Pydantic)
# ──────────────────────────────────────────────

class Prompt(BaseModel):
    """Payload para detecção sem prompt customizado."""
    image: str  # Base64 da imagem


class PromptSys(BaseModel):
    """Payload para detecção com prompt customizado."""
    image: str  # Base64 da imagem
    prompt: str


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

async def _call_vllm(client: httpx.AsyncClient, image_b64: str, system: str) -> dict:
    """
    Realiza chamada para o vLLM (Qwen) com a imagem e instrução de sistema.

    Args:
        client (httpx.AsyncClient): Cliente HTTP assíncrono.
        image_b64 (str): Imagem em base64.
        system (str): Instrução de sistema para o modelo.

    Returns:
        dict: Resposta JSON do vLLM.
    """
    response = await client.post(
        VLLM_URL,
        json={
            "model": "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit",
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
    print(f"🔹 vLLM Response Status: {response.status_code}")
    print(f"🔹 vLLM Response: {result}")
    return result


# Fração mínima da área da imagem para um objeto ser enviado ao Qwen.
# Objetos menores que 0.5 % da imagem são considerados ruído e descartados.
_MIN_AREA_FRACTION = 0.005


def _bbox_area(bbox: BBox) -> float:
    """Retorna a área normalizada da bounding box (0–1)."""
    return max(0.0, (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1))


def _bbox_prominence(bbox: BBox) -> float:
    """
    Calcula a proeminência espacial do objeto: combina tamanho relativo e centralidade.

    Fórmula: 0.6 * area_score + 0.4 * centrality
    - area_score: raiz quadrada da área normalizada, escalada para [0, 1].
    - centrality: 1 quando o centro do objeto coincide com o centro da imagem; 0 nos cantos.
    """
    area = _bbox_area(bbox)
    area_score = min(1.0, math.sqrt(area) * 2)  # sqrt(0.25)*2 = 1.0 para objetos grandes

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
    Etapa 2 do pipeline YOLOE → Qwen.

    Fluxo interno:
      1. Filtra objetos com área < _MIN_AREA_FRACTION (remove ruído).
      2. Anota a imagem com bboxes NUMERADAS usando os índices originais.
      3. Monta o payload JSON: {index, label, conf, bbox_norm, area, prominence}.
      4. Envia imagem anotada + JSON ao Qwen e retorna a resposta bruta.

    Args:
        client (httpx.AsyncClient): Cliente HTTP assíncrono.
        img (Image.Image): Imagem PIL original (sem anotações).
        yoloe_objects (list[DetectedObject]): Todas as detecções do YOLOE.

    Returns:
        dict: Resposta JSON do vLLM.
    """
    import json as _json

    # 1. Filtra por área mínima (mantendo o índice original para o Qwen referenciar)
    filtered: list[tuple[int, DetectedObject]] = [
        (i, obj)
        for i, obj in enumerate(yoloe_objects)
        if _bbox_area(obj.bbox) >= _MIN_AREA_FRACTION
    ]

    if not filtered:
        # Fallback: sem filtro de área se todos os objetos forem pequenos
        filtered = list(enumerate(yoloe_objects))

    # 2. Anota a imagem com bboxes numeradas
    annotated_img = draw_indexed_detections(img, filtered)
    annotated_b64 = pil_to_b64(annotated_img)

    # 3. Monta o payload JSON com contexto espacial enriquecido
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

    # 4. Chama o Qwen com a imagem anotada e o JSON de detecções
    response = await client.post(
        VLLM_URL,
        json={
            "model": "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit",
            "messages": [
                {"role": "system", "content": SEQUENTIAL_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "The image has numbered bounding boxes drawn on it. "
                                "Each number corresponds to the 'index' field in the JSON list below.\n\n"
                                f"YOLOE detections:\n{yoloe_payload}\n\n"
                                "Analyze the annotated image together with the JSON. "
                                "Select the most visually significant and interactive objects "
                                "that a child would want to interact with. "
                                "Rename them correctly in Brazilian Portuguese."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{annotated_b64}"},
                        },
                    ],
                },
            ],
        },
    )
    result = response.json()
    print(f"🔹 vLLM Sequential Response Status: {response.status_code}")
    print(f"🔹 vLLM Sequential Response: {result}")
    return result


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.post("/detection")
async def detection(body: Prompt) -> dict:
    """
    Endpoint que retorna a resposta bruta do Qwen (sem fusão).

    Args:
        body (Prompt): Imagem em base64.

    Returns:
        dict: Resposta original do vLLM.
    """
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        qwen_response = await _call_vllm(client, image_b64, SYSTEM_INSTRUCTION)
        qwen_objects = parse_qwen_response(qwen_response)
        return serialize_detections(qwen_objects, too_many=False)



@app.post("/detection/sys_prompt")
async def detection_sys(body: PromptSys) -> dict:
    """
    Endpoint que permite enviar um prompt customizado para o Qwen.

    Args:
        body (PromptSys): Imagem em base64 e prompt customizado.

    Returns:
        dict: Resposta do vLLM com o prompt informado.
    """
    
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        qwen_response = await _call_vllm(client, image_b64, body.prompt)
        qwen_objects = parse_qwen_response(qwen_response)
        return serialize_detections(qwen_objects, too_many=False)



@app.post("/detection/fused")
async def detection_fused(body: Prompt) -> dict:
    """
    Endpoint principal — pipeline sequencial YOLOE → Qwen.

    Fluxo:
        1. YOLOE detecta todos os objetos na imagem, retornando bboxes precisas
           e labels brutos (possivelmente em inglês ou mal nomeados).
        2. Objetos muito pequenos (área < 0.5 % da imagem) são descartados como ruído.
        3. A imagem é anotada com bboxes NUMERADAS (índice original de cada objeto)
           usando cores distintas para cada box.
        4. O Qwen recebe: imagem anotada + JSON {index, label, conf, bbox_norm,
           area, prominence}. Ele seleciona os objetos mais significantes
           (não necessariamente os de maior confiança) e os renomeia em português.
        5. As coordenadas retornadas são SEMPRE as do YOLOE (preservadas via yoloe_index).

    Args:
        body (Prompt): Imagem em base64.

    Returns:
        dict: Lista de objetos refinados com label em português e bbox do YOLOE.
    """
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)

    # Etapa 1: YOLOE detecta todos os objetos
    loop = asyncio.get_event_loop()
    yoloe_objects = await loop.run_in_executor(None, run_yoloe, img)

    if not yoloe_objects:
        return serialize_detections([], too_many=False)

    print(f"🔷 YOLOE: {len(yoloe_objects)} objeto(s) detectado(s)")
    print(yoloe_objects)

    # Etapas 2-4: filtra por área, anota a imagem com boxes numeradas e chama o Qwen
    async with httpx.AsyncClient(timeout=120.0) as client:
        vllm_response = await _call_vllm_sequential(client, img, yoloe_objects)
        print(vllm_response)

    # Etapa 5: monta o resultado final com bboxes do YOLOE preservadas
    refined_objects = parse_qwen_refine_response(vllm_response, yoloe_objects)
    return serialize_detections(refined_objects)


@app.get("/health")
async def health() -> dict:
    """Endpoint de health check."""
    return {"status": "ok"}