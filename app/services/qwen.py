"""
Serviço para parsing das respostas do modelo Qwen (vLLM).

Funções principais:
- parse_qwen_response:   Qwen autônomo — bbox + label (normaliza coords pixel→[0,1]).
- parse_verify_response: Verificação final de crop — label + score de confiança.
- parse_qwen_refine_response: Legado — modo sequencial.
"""

import json
import re
import logging
from models import DetectedObject, BBox

log = logging.getLogger(__name__)

VERIFY_SCORE_THRESHOLD = 0.5  # Crops abaixo desse score são descartados na verificação


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
    Repara JSON malformado comum do Qwen2.5-VL/7B:
    - "bbox_norm": {"x1": 23, 274, 206, 578}  → chaves faltando em y1/x2/y2
    - "bbox_norm": {107, 244, 212, 352}        → todas as chaves faltando
    - "x2": 195, 260,                          → valor extra após coord
    - trailing commas antes de } ou ]
    """
    # Caso 1: bbox com x1 mas sem chaves para y1/x2/y2
    # ex: "bbox_norm": {"x1": 23, 274, 206, 578}
    raw = re.sub(
        r'"bbox_norm":\s*\{\s*"x1":\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\s*\}',
        r'"bbox_norm": {"x1": \1, "y1": \2, "x2": \3, "y2": \4}',
        raw,
    )
    # Caso 2: bbox sem nenhuma chave
    # ex: "bbox_norm": {107, 244, 212, 352}
    raw = re.sub(
        r'"bbox_norm":\s*\{\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\s*\}',
        r'"bbox_norm": {"x1": \1, "y1": \2, "x2": \3, "y2": \4}',
        raw,
    )
    # Valor numérico extra após coordenada (ex: "x2": 195, 260,)
    raw = re.sub(r'("x[12]":\s*[\d.]+),\s*[\d.]+\s*,', r'\1,', raw)
    raw = re.sub(r'("y[12]":\s*[\d.]+),\s*[\d.]+\s*,', r'\1,', raw)
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
        y1 = _normalize_coord(y1r, img_h)
        x2 = _normalize_coord(x2r, img_w)
        y2 = _normalize_coord(y2r, img_h)
        log.debug(f"Coords pixel → norm: ({x1r},{y1r},{x2r},{y2r}) → ({x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f})")
    else:
        x1, y1, x2, y2 = x1r, y1r, x2r, y2r

    # Garante x1<x2, y1<y2, [0,1]
    x1, x2 = sorted([max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))])
    y1, y2 = sorted([max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))])

    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


# ─────────────────────────────────────────────────────────────────────────────
# parse_qwen_response — bbox + label (Qwen autônomo)
# ─────────────────────────────────────────────────────────────────────────────

def parse_qwen_response(
    vllm_response: dict,
    img_size: tuple[int, int] | None = None,
) -> list[DetectedObject]:
    """
    Interpreta a resposta do Qwen (detecção autônoma com bbox + label).

    Args:
        vllm_response: Resposta bruta do vLLM.
        img_size: (width, height) da imagem enviada. Necessário para normalizar
                  coordenadas em pixel (Qwen2.5-VL retorna pixels, não [0,1]).
    """
    try:
        raw = vllm_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        log.error(f"parse_qwen_response: resposta inesperada — {e}")
        return []

    data = _parse_json_robust(raw)
    if data is None:
        return []

    img_w, img_h = img_size if img_size else (1, 1)

    objects = []
    for obj in data.get("objects", []):
        bb = obj.get("bbox_norm", {})
        try:
            bbox = _make_bbox(
                float(bb["x1"]), float(bb["y1"]),
                float(bb["x2"]), float(bb["y2"]),
                img_w, img_h,
            )
        except (KeyError, TypeError, ValueError):
            log.warning(f"Bbox inválida ignorada: {obj}")
            continue

        if bbox is None:
            log.warning(f"Bbox degenerada ignorada: {obj}")
            continue

        score = float(obj.get("score", 0.5))
        if score < 0.2:
            continue

        objects.append(DetectedObject(
            label=str(obj.get("label", "objeto")).strip(),
            score=score,
            bbox=bbox,  # _make_bbox já retorna BBox normalizada
            source="qwen",
        ))

    return sorted(objects, key=lambda o: o.score, reverse=True)[:5]


# ─────────────────────────────────────────────────────────────────────────────
# parse_verify_response — verificação final de crop
# ─────────────────────────────────────────────────────────────────────────────

def parse_verify_response(vllm_response: dict) -> tuple[str, float]:
    """
    Interpreta a resposta do Qwen na etapa de verificação final de crop.

    Returns:
        (label, score) — label em português e confiança.
        ("", 0.0) se score < threshold ou em caso de erro.
    """
    try:
        raw = vllm_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        log.error(f"parse_verify_response: resposta inesperada — {e}")
        return "", 0.0

    data = _parse_json_robust(raw)
    if data is None:
        return "", 0.0

    label = str(data.get("label", "")).strip()
    score = float(data.get("score", 0.0))

    if not label or score < VERIFY_SCORE_THRESHOLD:
        log.info(f"Verificação: descartado (score={score:.2f}, label='{label}')")
        return "", score

    return label, score


# ─────────────────────────────────────────────────────────────────────────────
# Legado — /detection/sequential
# ─────────────────────────────────────────────────────────────────────────────

def parse_qwen_refine_response(
    vllm_response: dict,
    yoloe_objects: list[DetectedObject],
) -> list[DetectedObject]:
    """Modo sequencial legado."""
    try:
        raw = vllm_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        log.error(f"parse_qwen_refine_response: resposta inesperada — {e}")
        return []

    data = _parse_json_robust(raw)
    if data is None:
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
            label=str(obj.get("label", yobj.label)).strip(),
            score=score,
            bbox=yobj.bbox,
            source="sequential",
            yoloe_conf=yobj.yoloe_conf,
        ))

    return sorted(objects, key=lambda o: o.score, reverse=True)[:5]
