"""
Serviço para parsing da resposta do modelo Qwen (vLLM).

Converte a resposta JSON bruta do Qwen em objetos DetectedObject estruturados.
"""

import json
import logging
from models import DetectedObject, BBox

log = logging.getLogger(__name__)


def parse_qwen_response(vllm_response: dict) -> list[DetectedObject]:
    """
    Extrai e valida a lista de objetos detectados a partir da resposta do vLLM (Qwen).

    Trata erros de parsing, blocos markdown e o campo 'too_many_objects'.

    Args:
        vllm_response (dict): Resposta bruta da API do vLLM.

    Returns:
        list[DetectedObject]: Lista de objetos detectados (vazia em caso de erro ou limite excedido).
    """
    try:
        raw = vllm_response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        log.warning(f"Resposta vLLM inesperada: {e}")
        return []

    # Remove blocos markdown (```json ... ```) se presentes
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"JSON inválido do Qwen: {e}\n{raw[:300]}")
        return []

    # Verifica se o modelo indicou muitos objetos
    if data.get("too_many_objects"):
        log.info("Qwen: too_many_objects=true")
        return []

    objects = []
    for obj in data.get("objects", []):
        bb = obj.get("bbox_norm", {})
        try:
            bbox = BBox(
                x1=float(bb["x1"]),
                y1=float(bb["y1"]),
                x2=float(bb["x2"]),
                y2=float(bb["y2"]),
            )
        except (KeyError, TypeError, ValueError):
            log.warning(f"Bounding box inválida ignorada: {obj}")
            continue

        objects.append(DetectedObject(
            label=obj.get("label", "objeto"),
            score=float(obj.get("score", 0.5)),
            bbox=bbox,
            source="qwen",
        ))

    log.info(f"Qwen: {len(objects)} objeto(s) detectado(s)")
    return objects