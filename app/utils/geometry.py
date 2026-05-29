"""
Funções geométricas para manipulação de bounding boxes.

Utilizadas no pipeline de detecção para calcular área, centralidade,
IoU, contenção e filtragem de objetos de foreground.
"""

import math
from PIL import Image

from models import BBox, DetectedObject
from config import (
    MIN_FOREGROUND_AREA,
    MAX_FOREGROUND_AREA,
    SMALL_AREA,
    SMALL_CENTRALITY,
    SUBPART_CONTAINMENT,
    CROP_PADDING,
)


def bbox_area(bbox: BBox) -> float:
    """Área da bounding box em coordenadas normalizadas."""
    return max(0.0, (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1))


def centrality(bbox: BBox) -> float:
    """1.0 = centro exato, 0.0 = canto da imagem."""
    cx = (bbox.x1 + bbox.x2) / 2
    cy = (bbox.y1 + bbox.y2) / 2
    return max(0.0, 1.0 - 2 * max(abs(cx - 0.5), abs(cy - 0.5)))


def iou(a: BBox, b: BBox) -> float:
    """Intersection over Union entre dois bounding boxes."""
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / (bbox_area(a) + bbox_area(b) - inter)


def containment(a: BBox, b: BBox) -> float:
    """Fração da área de A que está contida em B."""
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = bbox_area(a)
    return inter / area_a if area_a > 0 else 0.0


def bbox_prominence(bbox: BBox) -> float:
    """
    Score de proeminência combinando área e centralidade.
    Usado no ranking final: score × (0.6 + 0.4 × centralidade).
    """
    area = bbox_area(bbox)
    area_score = min(1.0, math.sqrt(area) * 2)
    cx = (bbox.x1 + bbox.x2) / 2
    cy = (bbox.y1 + bbox.y2) / 2
    cent = max(0.0, 1.0 - 2 * max(abs(cx - 0.5), abs(cy - 0.5)))
    return round(0.6 * area_score + 0.4 * cent, 4)


def is_foreground(obj: DetectedObject) -> bool:
    """
    Retorna True se o objeto deve ser considerado foreground relevante.

    Descarta:
      - Muito pequenos (< 1.5%) → ruído/detalhe de fundo
      - Muito grandes (> 75%)   → fundo/cena inteira
      - Pequenos (< 3%) e periféricos → foco secundário improvável
    """
    area = bbox_area(obj.bbox)
    if area < MIN_FOREGROUND_AREA:
        return False
    if area > MAX_FOREGROUND_AREA:
        return False
    if area < SMALL_AREA and centrality(obj.bbox) < SMALL_CENTRALITY:
        return False
    return True


def remove_subparts(objects: list[DetectedObject]) -> list[DetectedObject]:
    """
    Remove detecções que são sub-partes de outros objetos.

    Regra: containment(A em B) > 60% E area(A) < area(B) → descarta A.
    Exemplo: bbox "teclado" dentro de bbox "notebook" → descarta "teclado".
    """
    to_remove: set[int] = set()
    for i, a in enumerate(objects):
        if i in to_remove:
            continue
        for j, b in enumerate(objects):
            if i == j or j in to_remove:
                continue
            if (containment(a.bbox, b.bbox) > SUBPART_CONTAINMENT
                    and bbox_area(a.bbox) < bbox_area(b.bbox)):
                to_remove.add(i)
                print(
                    f"Sub-parte: '{a.label}' ({bbox_area(a.bbox):.3f}) "
                    f"descartado — está dentro de '{b.label}' ({bbox_area(b.bbox):.3f})"
                )
                break
    return [o for i, o in enumerate(objects) if i not in to_remove]


def crop_object(img: Image.Image, bbox: BBox) -> Image.Image:
    """Recorta o objeto da imagem com padding de 5%, respeitando os limites."""
    w, h = img.size
    pad_x = (bbox.x2 - bbox.x1) * CROP_PADDING
    pad_y = (bbox.y2 - bbox.y1) * CROP_PADDING
    x1 = max(0.0, bbox.x1 - pad_x)
    y1 = max(0.0, bbox.y1 - pad_y)
    x2 = min(1.0, bbox.x2 + pad_x)
    y2 = min(1.0, bbox.y2 + pad_y)
    return img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
