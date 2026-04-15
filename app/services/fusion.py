"""
Serviço de fusão de detecções entre Qwen e YOLOE.

Utiliza IoU (Intersection over Union) para combinar bounding boxes precisas do YOLOE
com os labels e scores semânticos do Qwen.
"""

import logging
from models import DetectedObject

log = logging.getLogger(__name__)

IOU_THRESHOLD = 0.15  # Limiar mínimo de IoU para considerar um match


def fuse_by_iou(
    qwen_objects: list[DetectedObject],
    yoloe_objects: list[DetectedObject],
    iou_threshold: float = IOU_THRESHOLD,
) -> list[DetectedObject]:
    """
    Combina detecções do Qwen com as do YOLOE baseado em IoU.

    Regras:
        - Se IoU >= threshold: usa label/score do Qwen + bbox do YOLOE (source="fused")
        - Caso contrário: mantém o objeto do Qwen intacto (source="qwen")

    Args:
        qwen_objects (list[DetectedObject]): Lista de objetos detectados pelo Qwen.
        yoloe_objects (list[DetectedObject]): Lista de objetos detectados pelo YOLOE.
        iou_threshold (float): Limiar mínimo de IoU para considerar um match.

    Returns:
        list[DetectedObject]: Lista de objetos após fusão.
    """
    fused = []
    used = set()  # Índices dos objetos do YOLOE já utilizados

    for qobj in qwen_objects:
        best_iou = 0.0
        best_idx = -1

        # Encontra o objeto do YOLOE com maior IoU em relação ao Qwen atual
        for i, yobj in enumerate(yoloe_objects):
            iou = qobj.bbox.iou(yobj.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        # Se o melhor match atende ao limiar e ainda não foi usado
        if best_iou >= iou_threshold and best_idx not in used:
            yobj = yoloe_objects[best_idx]
            used.add(best_idx)
            fused.append(DetectedObject(
                label=qobj.label,
                score=qobj.score,
                bbox=yobj.bbox,
                source="fused",
                yoloe_conf=yobj.yoloe_conf,
            ))
            log.debug(
                f"Fusão: '{qobj.label}' ↔ '{yobj.label}' (IoU={best_iou:.3f})"
            )
        else:
            fused.append(qobj)
            log.debug(
                f"Sem match para '{qobj.label}' (melhor IoU={best_iou:.3f})"
            )

    return fused


def serialize_detections(objects: list[DetectedObject], too_many: bool = False) -> dict:
    """
    Serializa a lista de objetos detectados para o formato de resposta da API.

    Args:
        objects (list[DetectedObject]): Lista de objetos detectados.
        too_many (bool): Indica se muitos objetos foram detectados.

    Returns:
        dict: Dicionário no formato esperado pela API.
    """
    return {
        "objects": [
            {
                "label": o.label,
                "score": round(o.score, 4),
                "bbox_norm": {
                    "x1": round(o.bbox.x1, 4),
                    "y1": round(o.bbox.y1, 4),
                    "x2": round(o.bbox.x2, 4),
                    "y2": round(o.bbox.y2, 4),
                },
                "source": o.source,
                "yoloe_conf": round(o.yoloe_conf, 4) if o.yoloe_conf is not None else None,
            }
            for o in objects
        ],
        "too_many_objects": too_many,
    }