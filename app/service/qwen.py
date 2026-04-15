import json
import logging

log = logging.getLogger(__name__)

def parse_qwen_response(vllm_response: dict) -> list[DetectedObject]:
    """
    Recebe o JSON bruto retornado pelo vLLM e extrai a lista de DetectedObject.
    Retorna lista vazia em caso de erro ou too_many_objects=true.
    """
    try:
        raw = vllm_response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        log.warning(f"Resposta vLLM inesperada: {e}")
        return []

    # Remove blocos markdown caso o modelo ignore a instrução
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
            log.warning(f"bbox inválida ignorada: {obj}")
            continue

        objects.append(DetectedObject(
            label=obj.get("label", "objeto"),
            score=float(obj.get("score", 0.5)),
            bbox=bbox,
            source="qwen",
        ))

    log.info(f"Qwen: {len(objects)} objeto(s)")
    return objects