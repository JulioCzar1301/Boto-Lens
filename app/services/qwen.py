"""
Serviço para parsing das respostas do modelo Qwen (vLLM).

Funções:
- parse_scene_response:  Etapa 1 — Qwen identifica objetos centrais + quantidades.
- parse_judge_response:  Etapa 3 — Qwen julga se um crop do YOLOE corresponde a um objeto central.
- parse_qwen_response:   Legado — Qwen autônomo (gera bbox + label).
- parse_qwen_refine_response: Legado — modo sequencial.
"""

import json
import logging
from models import DetectedObject, BBox

log = logging.getLogger(__name__)

JUDGE_SCORE_THRESHOLD = 0.6  # Score mínimo para o Qwen aceitar um candidato


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

def parse_scene_response(vllm_response: dict) -> list[dict]:
    """
    Interpreta a resposta do Qwen da Etapa 1 (identificação de cena).

    Returns:
        Lista de dicts:
        [{"label": "carregador", "count": 1, "prompts": ["charger", "wall charger", ...]}, ...]
        Vazia se não houver objetos ou em caso de erro.
    """
    try:
        raw = vllm_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        log.error(f"parse_scene_response: resposta inesperada — {e}")
        return []

    raw = _extract_json(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"parse_scene_response: JSON inválido — {e}\n{raw[:300]}")
        return []

    result = []
    for obj in data.get("objects", []):
        label = str(obj.get("label", "")).strip()
        count = int(obj.get("count", 1))
        prompts = [str(p).strip() for p in obj.get("prompts", []) if str(p).strip()]
        if not label or count <= 0:
            continue
        # Fallback: se Qwen não gerou prompts, usa o label em inglês simples
        if not prompts:
            prompts = [label]
        result.append({"label": label, "count": count, "prompts": prompts})

    log.info(f"Cena identificada: {result}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 3: parse do julgamento — Qwen valida se crop corresponde a objeto central
# ─────────────────────────────────────────────────────────────────────────────

def parse_judge_response(vllm_response: dict) -> tuple[str | None, float]:
    """
    Interpreta a resposta do Qwen da Etapa 3 (julgamento de candidato).

    Returns:
        (match_label, score) — label correspondente e confiança.
        (None, 0.0) se não há match ou score < threshold.
    """
    try:
        raw = vllm_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        log.error(f"parse_judge_response: resposta inesperada — {e}")
        return None, 0.0

    raw = _extract_json(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"parse_judge_response: JSON inválido — {e}\n{raw[:200]}")
        return None, 0.0

    match = data.get("match")
    score = float(data.get("score", 0.0))

    if not match or score < JUDGE_SCORE_THRESHOLD:
        return None, score

    return str(match).strip(), score


# ─────────────────────────────────────────────────────────────────────────────
# Legado — /detection (Qwen autônomo)
# ─────────────────────────────────────────────────────────────────────────────

def parse_qwen_response(vllm_response: dict) -> list[DetectedObject]:
    """Modo autônomo legado — Qwen gera bbox + label."""
    try:
        raw = vllm_response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        log.error(f"parse_qwen_response: resposta inesperada — {e}")
        return []

    raw = _extract_json(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"parse_qwen_response: JSON inválido — {e}\n{raw[:300]}")
        return []

    objects = []
    for obj in data.get("objects", []):
        bb = obj.get("bbox_norm", {})
        try:
            bbox = BBox(
                x1=float(bb["x1"]), y1=float(bb["y1"]),
                x2=float(bb["x2"]), y2=float(bb["y2"]),
            )
        except (KeyError, TypeError, ValueError):
            log.warning(f"Bbox inválida ignorada: {obj}")
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


# ─────────────────────────────────────────────────────────────────────────────
# Legado — /detection/sequential
# ─────────────────────────────────────────────────────────────────────────────

def parse_qwen_refine_response(
    vllm_response: dict,
    yoloe_objects: list[DetectedObject],
) -> list[DetectedObject]:
    """Modo sequencial legado."""
    try:
        raw = vllm_response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        log.error(f"parse_qwen_refine_response: resposta inesperada — {e}")
        return []

    raw = _extract_json(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"parse_qwen_refine_response: JSON inválido — {e}\n{raw[:300]}")
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
