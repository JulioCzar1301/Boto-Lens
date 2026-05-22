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


def parse_qwen_response(vllm_response: dict) -> list[DetectedObject]:
    """
    Extrai e valida a lista de objetos detectados a partir da resposta do vLLM (Qwen).

    Trata erros de parsing, blocos markdown e o campo 'too_many_objects'.

    Args:
        vllm_response (dict): Resposta bruta da API do vLLM.

    Returns:
        list[DetectedObject]: Lista de objetos detectados (vazia em caso de erro ou limite excedido).
    """
    print("Resposta bruta do vLLM (Qwen):", json.dumps(vllm_response)[:500])  # Log para debug')
    try:
        raw = vllm_response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        log.error(f"Resposta vLLM inesperada: {e}")
        log.error(f"Resposta completa: {json.dumps(vllm_response)}")
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
        yoloe_objects (list[DetectedObject]): Detecções originais do YOLOE
            (usadas para recuperar a bbox pelo índice).

    Returns:
        list[DetectedObject]: Lista refinada com label do Qwen e bbox do YOLOE.
    """
    try:
        raw = vllm_response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        log.error(f"Resposta vLLM inesperada (refine): {e}")
        log.error(f"Resposta completa: {json.dumps(vllm_response)}")
        return []

    # Remove blocos markdown se presentes
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"JSON inválido do Qwen (refine): {e}\n{raw[:300]}")
        return []

    if data.get("too_many_objects"):
        log.info("Qwen (refine): too_many_objects=true")
        return []

    objects = []
    for obj in data.get("objects", []):
        idx = obj.get("yoloe_index")
        if idx is None or not (0 <= idx < len(yoloe_objects)):
            log.warning(f"yoloe_index inválido ignorado: {obj}")
            continue

        yobj = yoloe_objects[idx]
        label = obj.get("label", yobj.label)
        score = float(obj.get("score", yobj.score))

        if score < 0.2:
            continue

        objects.append(DetectedObject(
            label=label,
            score=score,
            bbox=yobj.bbox,          # coordenadas preservadas do YOLOE
            source="sequential",
            yoloe_conf=yobj.yoloe_conf,
        ))

    log.info(f"Qwen (refine): {len(objects)} objeto(s) selecionado(s) do YOLOE")
    return objects
