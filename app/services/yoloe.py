"""
Serviço para execução do modelo YOLOE em imagens.

Modos:
- run_yoloe(img):                  Prompt-free (COCO classes, sem guia textual).
- run_yoloe_prompted(img, prompts): Text-prompted — YOLOE guiado por nomes em inglês
                                    gerados pelo Qwen. Mais preciso para objetos específicos.
"""

import logging
from PIL import Image
from ultralytics import YOLOE

from models import DetectedObject, BBox

log = logging.getLogger(__name__)

YOLOE_MODEL  = "yoloe-11l-seg-pf.pt"
YOLOE_CONF   = 0.15   # limiar menor com prompts — YOLOE já é guiado
YOLOE_MAX_DET = 20


_yoloe_model = None


def get_yoloe() -> YOLOE:
    """Retorna a instância singleton do modelo YOLOE."""
    global _yoloe_model
    if _yoloe_model is None:
        log.info(f"Carregando YOLOE ({YOLOE_MODEL})...")
        _yoloe_model = YOLOE(YOLOE_MODEL)
    return _yoloe_model


def _boxes_to_objects(
    results,
    label_map: dict[int, str],
    width: int,
    height: int,
) -> list[DetectedObject]:
    """Converte resultados do YOLOE em DetectedObject com bboxes normalizadas."""
    objects = []
    if not results:
        return objects
    for box in results[0].boxes:
        cls_idx = int(box.cls[0])
        label = label_map.get(cls_idx, str(cls_idx))
        xyxy = box.xyxy[0].tolist()
        objects.append(DetectedObject(
            label=label,
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
    return objects


def run_yoloe_prompted(img: Image.Image, prompts: list[str]) -> list[DetectedObject]:
    """
    Executa YOLOE com text prompts gerados pelo Qwen.

    O modelo usa os nomes em inglês para guiar a detecção, produzindo bboxes
    mais precisas para os objetos específicos identificados na cena.

    Args:
        img (Image.Image): Imagem PIL RGB.
        prompts (list[str]): Lista de nomes em inglês (ex: ["charger", "wall charger"]).

    Returns:
        list[DetectedObject]: Objetos detectados com bboxes normalizadas.
                              O label de cada objeto é o prompt que gerou a detecção.
    """
    if not prompts:
        log.warning("run_yoloe_prompted: nenhum prompt fornecido, usando prompt-free")
        return run_yoloe(img)

    model = get_yoloe()
    width, height = img.size

    try:
        # Configura os text prompts no modelo
        text_pe = model.get_text_pe(prompts)
        model.set_classes(prompts, text_pe)
        log.info(f"YOLOE text prompts: {prompts}")
    except Exception as e:
        log.warning(f"Text prompts não suportados neste modelo, usando prompt-free: {e}")
        return run_yoloe(img)

    results = model.predict(
        source=img,
        conf=YOLOE_CONF,
        max_det=YOLOE_MAX_DET,
        verbose=False,
    )

    # O label_map é os próprios prompts indexados por cls
    label_map = {i: p for i, p in enumerate(prompts)}
    objects = _boxes_to_objects(results, label_map, width, height)
    log.info(f"YOLOE (prompted): {len(objects)} objeto(s) detectado(s)")
    return objects


def run_yoloe(img: Image.Image) -> list[DetectedObject]:
    """
    Executa o YOLOE prompt-free (COCO classes).
    Mantido para o endpoint /detection/sequential.
    """
    model = get_yoloe()
    width, height = img.size

    results = model.predict(
        source=img,
        conf=0.25,
        max_det=50,
        verbose=False,
    )

    label_map = {i: name for i, name in model.names.items()}
    objects = _boxes_to_objects(results, label_map, width, height)
    log.info(f"YOLOE (prompt-free): {len(objects)} objeto(s) detectado(s)")
    return objects
