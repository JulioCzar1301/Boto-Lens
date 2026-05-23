"""
Serviço para parsing das respostas do modelo Qwen (vLLM).

Funções:
- parse_crop_response: classifica um crop individual (nova arquitetura).
- parse_qwen_response: modo autônomo legado (Qwen gera bbox + label).
- parse_qwen_refine_response: modo sequencial legado.
"""

import json
import logging
from models import DetectedObject, BBox

log = logging.getLogger(__name__)

SCORE_THRESHOLD = 0.6  # Crops com score abaixo disso são descartados


def _strip_markdown(raw: str) -> str:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def parse_crop_response(vllm_response: dict) -> tuple[str, float]:
    """
    Interpreta a resposta do Qwen para classificação de um crop individual.

    Returns:
        (label, score) — label em português e score de confiança (0–1).
        Retorna ("", 0.0) em caso de erro.
    """
    try:
        raw = vllm_response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        log.error(f"Resposta vLLM inesperada (crop): {e}")
        return "", 0.0

    raw = _strip_markdown(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"JSON inválido do Qwen (crop): {e}\n{raw[:200]}")
        return "", 0.0

    label = data.get("label", "").strip()
    score = float(data.get("score", 0.0))

    if not label or score < SCORE_THRESHOLD:
        return "", score

    return label, score


def parse_qwen_response(vllm_response: dict) -> list[DetectedObject]:
    """Modo autônomo legado — Qwen gera bbox + label."""
    print("Resposta bruta do vLLM (Qwen):", json.dumps(vllm_response)[:500])
    try:
        raw = vllm_response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        log.error(f"Resposta vLLM inesperada: {e}")
        return []

    raw = _strip_markdown(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"JSON inválido do Qwen: {e}\n{raw[:300]}")
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

        score = float(obj.get("score", 0.5))
        if score < 0.2:
            continue

        objects.append(DetectedObject(
            label=obj.get("label", "objeto"),
            score=score,
            bbox=bbox,
            source="qwen",
        ))

    return sorted(objects, key=lambda o: o.score, reverse=True)[:5]


def parse_qwen_refine_response(
    vllm_response: dict,
    yoloe_objects: list[DetectedObject],
) -> list[DetectedObject]:
    """Modo sequencial legado."""
    try:
        raw = vllm_response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        log.error(f"Resposta vLLM inesperada (refine): {e}")
        return []

    raw = _strip_markdown(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"JSON inválido do Qwen (refine): {e}\n{raw[:300]}")
        return []

    objects = []
    for obj in data.get("objects", []):
        idx = obj.get("yoloe_index")
        if idx is None or not (0 <= idx < len(yoloe_objects)):
            continue

        yobj = yoloe_objects[idx]
        score = float(obj.get("score", yobj.score))
        if score < 0.2:
            continue

        objects.append(DetectedObject(
            label=obj.get("label", yobj.label),
            score=score,
            bbox=yobj.bbox,
            source="sequential",
            yoloe_conf=yobj.yoloe_conf,
        ))

    return sorted(objects, key=lambda o: o.score, reverse=True)[:5]
