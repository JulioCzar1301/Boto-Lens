"""
API principal para detecção de objetos utilizando fusão entre Qwen (vLLM) e YOLOE.

Endpoints disponíveis:
- /detection: Apenas Qwen (resposta bruta)
- /detection/sys_prompt: Qwen com prompt customizado
- /detection/fused: Fusão Qwen + YOLOE
- /health: Health check
"""

import os
import asyncio
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from system_instruction import SYSTEM_INSTRUCTION
from models import DetectedObject
from services.yoloe import run_yoloe
from services.qwen import parse_qwen_response
from services.fusion import fuse_by_iou, serialize_detections
from utils.image import resize_base64_image, b64_to_pil

app = FastAPI()

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
    return response.json()


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
    async with httpx.AsyncClient(timeout=120.0) as client:
        return await _call_vllm(client, body.image, body.prompt)


@app.post("/detection/fused")
async def detection_fused(body: Prompt) -> dict:
    """
    Endpoint principal com fusão entre Qwen e YOLOE.

    Executa Qwen e YOLOE em paralelo, combina os resultados via IoU
    e retorna os objetos fusionados.

    Args:
        body (Prompt): Imagem em base64.

    Returns:
        dict: Lista de objetos detectados após fusão.
    """
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)

    async with httpx.AsyncClient(timeout=120.0) as client:
        qwen_task = asyncio.create_task(
            _call_vllm(client, image_b64, SYSTEM_INSTRUCTION)
        )
        yoloe_task = asyncio.get_event_loop().run_in_executor(None, run_yoloe, img)

        vllm_response, yoloe_objects = await asyncio.gather(qwen_task, yoloe_task)

    # Verifica se o Qwen indicou excesso de objetos
    if vllm_response.get("too_many_objects"):
        return serialize_detections([], too_many=True)

    qwen_objects = parse_qwen_response(vllm_response)
    fused_objects = fuse_by_iou(qwen_objects, yoloe_objects)
    return serialize_detections(fused_objects)


@app.get("/health")
async def health() -> dict:
    """Endpoint de health check."""
    return {"status": "ok"}