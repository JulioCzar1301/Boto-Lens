"""
Serviço para execução do modelo YOLOE (prompt-free) em imagens.

Responsável por carregar o modelo e realizar inferências,
retornando objetos detectados com bounding boxes normalizadas.
"""

import logging
from PIL import Image
from ultralytics import YOLOE

from models import DetectedObject, BBox

log = logging.getLogger(__name__)

# Constantes de configuração do YOLOE
YOLOE_MODEL = "yoloe-11l-seg-pf.pt"
YOLOE_CONF = 0.25
YOLOE_MAX_DET = 50

# Singleton do modelo YOLOE
_yoloe_model = None


def get_yoloe() -> YOLOE:
    """
    Retorna a instância singleton do modelo YOLOE.

    Carrega o modelo na primeira chamada e reutiliza nas chamadas seguintes.

    Returns:
        YOLOE: Instância do modelo carregado.
    """
    global _yoloe_model
    if _yoloe_model is None:
        log.info(f"Carregando YOLOE ({YOLOE_MODEL})...")
        _yoloe_model = YOLOE(YOLOE_MODEL)
    return _yoloe_model


def run_yoloe(img: Image.Image) -> list[DetectedObject]:
    """
    Executa o YOLOE prompt-free na imagem fornecida.

    Realiza a detecção de objetos e retorna uma lista de DetectedObject
    com bounding boxes normalizadas (coordenadas entre 0 e 1).

    Args:
        img (Image.Image): Imagem PIL no formato RGB.

    Returns:
        list[DetectedObject]: Lista de objetos detectados pelo YOLOE.
    """
    model = get_yoloe()
    width, height = img.size

    results = model.predict(
        source=img,
        conf=YOLOE_CONF,
        max_det=YOLOE_MAX_DET,
        verbose=False,
    )

    objects = []
    if not results:
        return objects

    for box in results[0].boxes:
        xyxy = box.xyxy[0].tolist()
        objects.append(DetectedObject(
            label=model.names.get(int(box.cls[0]), str(int(box.cls[0]))),
            score=float(box.conf[0]),
            bbox=BBox(
                x1=xyxy[0] / width,
                y1=xyxy[1] / height,
                x2=xyxy[2] / width,
                y2=xyxy[3] / height,
            ),
            source="yoloe",
            yoloe_conf=float(box.conf[0]),
        ))

    log.info(f"YOLOE: {len(objects)} objeto(s) detectado(s)")
    return objects