"""
Serviço para comunicação e parsing das respostas do modelo Qwen (vLLM).

Funções de chamada HTTP:
- call_qwen:              Envia imagem completa ao Qwen para detecção + bboxes.
- call_verify:            Envia crop ao Qwen para verificação/correção de label.
- call_vllm_sequential:   Modo legado — YOLOE detecta, Qwen renomeia.

Funções de parsing:
- parse_qwen_response:        Qwen autônomo — bbox + label (normaliza coords pixel→[0,1]).
- parse_verify_response:      Verificação final de crop — label + score de confiança.
- parse_qwen_refine_response: Legado — modo sequencial.
"""

import json
import re
import logging
import httpx
from PIL import Image

from models import DetectedObject, BBox
from system_instruction import SYSTEM_INSTRUCTION, VERIFY_INSTRUCTION, SEQUENTIAL_INSTRUCTION
from config import VLLM_URL, VLLM_MODEL, MIN_FOREGROUND_AREA
from utils.geometry import bbox_area, bbox_prominence

log = logging.getLogger(__name__)

VERIFY_SCORE_THRESHOLD = 0.5  # Crops abaixo desse score são descartados na verificação


# ─────────────────────────────────────────────────────────────────────────────
# Chamadas HTTP ao vLLM
# ─────────────────────────────────────────────────────────────────────────────

async def call_qwen(client: httpx.AsyncClient, image_b64: str, system: str) -> dict:
    """Envia a imagem completa ao Qwen e retorna o JSON de detecções."""
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
    log.info(f"[Qwen] status={response.status_code}")
    try:
        raw = result["choices"][0]["message"]["content"]
        log.debug(f"[Qwen raw] {raw[:500]}")
    except Exception:
        pass
    return result


async def call_verify(
    client: httpx.AsyncClient,
    crop_b64: str,
    hint_label: str,
) -> dict:
    """
    Etapa 4: envia o crop de um objeto fundido ao Qwen para verificação/correção de label.

    Args:
        client:     Cliente HTTP assíncrono.
        crop_b64:   Crop do objeto em base64.
        hint_label: Label atual do objeto (contexto para o modelo).

    Returns:
        Resposta bruta do vLLM com {"label": ..., "score": ...}.
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
    log.info(f"[Verify] status={response.status_code} | hint='{hint_label}'")
    return result


async def call_vllm_sequential(
    client: httpx.AsyncClient,
    img: Image.Image,
    yoloe_objects: list[DetectedObject],
) -> dict:
    """
    Modo sequencial legado: envia as detecções do YOLOE ao Qwen para renomeação
    em português e seleção dos top-5 objetos mais proeminentes.
    """
    from utils.image import pil_to_b64

    filtered = [
        (i, obj) for i, obj in enumerate(yoloe_objects)
        if bbox_area(obj.bbox) >= MIN_FOREGROUND_AREA
    ] or list(enumerate(yoloe_objects))

    clean_b64 = pil_to_b64(img)
    payload = json.dumps(
        [
            {
                "index": i,
                "label": obj.label,
                "conf": round(obj.yoloe_conf or obj.score, 4),
                "bbox_norm": {
                    "x1": round(obj.bbox.x1, 4),
                    "y1": round(obj.bbox.y1, 4),
                    "x2": round(obj.bbox.x2, 4),
                    "y2": round(obj.bbox.y2, 4),
                },
                "area": round(bbox_area(obj.bbox), 4),
                "prominence": bbox_prominence(obj.bbox),
            }
            for i, obj in filtered
        ],
        ensure_ascii=False,
    )

    response = await client.post(
        VLLM_URL,
        json={
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": SEQUENTIAL_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"YOLOE detections:\n{payload}\n\nRename in Brazilian Portuguese.",
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
    return response.json()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> str:
    """Remove markdown code fences e whitespace extra."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Etapa 1: parse da cena — lista de objetos centrais + quantidades
# ─────────────────────────────────────────────────────────────────────────────

def _try_fix_json(raw: str) -> str:
    """
    Repara JSON malformado comum do Qwen2.5-VL/7B.
    Estratégia robusta: para qualquer bloco "bbox_norm": {...},
    extrai os 4 primeiros números e reconstrói o objeto corretamente,
    independente do formato das chaves.
    """
    def fix_bbox(m: re.Match) -> str:
        content = m.group(1)
        # Remove strings entre aspas (nomes de chaves como "x1", "x1=", etc.)
        content = re.sub(r'"[^"]*"\s*:?\s*', '', content)
        nums = re.findall(r"[\d.]+", content)
        if len(nums) >= 4:
            return f'"bbox_norm": {{"x1": {nums[0]}, "y1": {nums[1]}, "x2": {nums[2]}, "y2": {nums[3]}}}'
        return m.group(0)  # não altera se não encontrar 4 números

    raw = re.sub(r'"bbox_norm":\s*\{([^}]*)\}', fix_bbox, raw, flags=re.DOTALL)
    # Trailing comma antes de } ou ]
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    return raw


def _parse_json_robust(raw: str) -> dict | None:
    """Tenta parsear JSON, com reparo automático em caso de falha."""
    raw = _extract_json(raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fixed = _try_fix_json(raw)
    try:
        result = json.loads(fixed)
        log.info("JSON reparado com sucesso")
        return result
    except json.JSONDecodeError as e:
        log.warning(f"JSON inválido mesmo após reparo: {e}\n{raw[:300]}")
        return None


def _normalize_coord(v: float, dim: int) -> float:
    """Normaliza: se valor > 1.0, assume pixel e divide pela dimensão."""
    return v / dim if v > 1.0 else v


def _make_bbox(x1r, y1r, x2r, y2r, img_w: int, img_h: int) -> BBox | None:
    """
    Cria BBox normalizada a partir de coords raw (pixel ou [0,1]).
    Retorna None se a bbox for degenerada.
    """
    is_pixel = any(v > 1.0 for v in [x1r, y1r, x2r, y2r])
    if is_pixel:
        x1 = _normalize_coord(x1r, img_w)
        y1 = _norma