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

def _normalize_coord(v: float, dim: int) -> float:
    """Normaliza coordenada: se > 1.0, assume pixels e divide pela dimensão."""
    return v / dim if v > 1.0 else v


def _try_fix_json(raw: str) -> str:
    """
    Tenta reparar JSON malformado comum do Qwen2.5-VL:
    - Remove vírgulas duplas / valores extras (ex: "x2": 195, 260)
    - Remove trailing commas antes de } ou ]
    """
    import re
    # Remove valores extras após vírgula dentro de bbox: "x2": 195, 260, → "x2": 195,
    raw = re.sub(r'("x[12]":\s*[\d.]+),\s*[\d.]+\s*,', r'\1,', raw)
    raw = re.sub(r'("y[12]":\s*[\d.]+),\s*[\d.]+\s*,', r'\1,', raw)
    # Remove trailing commas antes de } ou ]
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    return raw


def parse_qwen_response(
    vllm_response: dict,
    img_size: tuple[int, int] | None = None,
) -> list[DetectedObject]:
    """
    Modo autônomo — Qwen gera bbox + label.

    Args:
        vllm_response: Resposta bruta do vLLM.
        img_size: (width, height) da imagem enviada ao Qwen.
                  Necessário para normalizar coords em pixel para [0,1].
                  Se None, assume que coords já são normalizadas.
    """
    try:
        raw = vllm_response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        log.error(f"parse_qwen_response: resposta inesperada — {e}")
        return []

    raw = _extract_json(raw)

    # Tentativa 1: parse direto
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Tentativa 2: repara JSON e tenta de novo
    if data is None:
        fixed = _try_fix_json(raw)
        try:
            data = json.loads(fixed)
            log.info("parse_qwen_response: JSON reparado com sucesso")
        except json.JSONDecodeError as e:
            log.warning(f"parse_qwen_response: JSON inválido mesmo após reparo — {e}\n{raw[:300]}")
            return []

    img_w, img_h = img_size if img_size else (1, 1)

    objects = []
    for obj in data.get("objects", []):
        bb = obj.get("bbox_norm", {})
        try:
            x1_raw = float(bb["x1"])
            y1_raw = float(bb["y1"])
            x2_raw = float(bb["x2"])
            y2_raw = float(bb["y2"])
        except (KeyError, TypeError, ValueError):
            log.warning(f"Bbox inválida ignorada: {obj}")
            continue

        # Detecta se são coords em pixel (qualquer valor > 1.0) e normaliza
        if any(v > 1.0 for v in [x1_raw, y1_raw, x2_raw, y2_raw]):
            x1 = _normalize_coord(x1_raw, img_w)
            y1 = _normalize_coord(y1_raw, img_h)
            x2 = _normalize_coord(x2_raw, img_w)
            y2 = _normalize_coord(y2_raw, img_h)
            log.info(f"Coords em pixel detectadas e normalizadas: ({x1_raw},{y1_raw},{x2_raw},{y2_raw}) → ({x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f})")
        else:
            x1, y1, x2, y2 = x1_raw, y1_raw, x2_raw, y2_raw

        # Garante x1<x2 e y1<y2 e valores em [0,1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)

        if x2 <= x1 or y2 <= y1:
            log.warning(f"Bbox degenerada ignorada: {obj}")
            continue

        score = float(obj.get("score", 0.5))
        if score < 0.2:
            continue

        objects.append(DetectedObject(
            label=obj.get("label", "objeto"),
            score=score,
            bbox=bbox,  # _make_bbox já retorna BBox normalizada
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
