"""
Serviço para parsing da resposta do modelo Qwen (vLLM).

Funções disponíveis:
- parse_qwen_response: modo autônomo (Qwen gera bbox + label).
- parse_qwen_refine_response: modo sequencial (Qwen seleciona do YOLOE e renomeia;
  coordenadas são preservadas do YOLOE via yoloe_index).
"""

import json
import logging
from models import DetectedObject, BBox

log = logging.getLogger(__name__)


def _strip_markdown(raw: str) -> str:
    """Remove blocos markdown (```json ... ```) se presentes."""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def parse_qwen_response(vllm_response: dict) -> list[DetectedObject]:
    """
    Extrai e valida a lista de objetos detectados a partir da resposta do vLLM (Qwen).

    Args:
        vllm_response (dict): Resposta bruta da API do vLLM.

    Returns:
        list[DetectedObject]: Lista de até 5 objetos detectados (vazia em caso de erro).
    """
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

    # Garante no máximo 5, ordenados por score
    objects = sorted(objects, key=lambda o: o.score, reverse=True)[:5]
    log.info(f"Qwen: {len(objects)} objeto(s) detectado(s)")
    return objects


def parse_qwen_refine_response(
    vllm_response: dict,
    yoloe_objects: list[DetectedObject],
) -> list[DetectedObject]:
    """
    Interpreta a resposta do Qwen no modo sequencial (refinamento sobre YOLOE).

    O Qwen retorna yoloe_index + label + score (sem bbox). As coordenadas são
    buscadas diretamente de yoloe_objects usando o índice informado.

    Args:
        vllm_response (dict): Resposta bruta da API do vLLM.
        yoloe_objects (list[DetectedObject]): Detecções originais do YOLOE.

    Returns:
        list[DetectedObject]: Lista refinada com label do Qwen e bbox do YOLOE.
    """
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
            log.warning(f"yoloe_index inválido ignorado: {obj}")
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

    # Garante no máximo 5, ordenados por score
    objects = sorted(objects, key=lambda o: o.score, reverse=True)[:5]
    log.info(f"Qwen (refine): {len(objects)} objeto(s) selecionado(s) do YOLOE")
    return objects
