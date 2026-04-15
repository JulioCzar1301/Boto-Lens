# services/yoloe.py
from models import DetectedObject
from ultralytics import YOLOE
import logging

log = logging.getLogger(__name__)

YOLOE_MODEL   = "yoloe-11l-seg-pf.pt"
YOLOE_CONF    = 0.25
YOLOE_MAX_DET = 50

_yoloe_model = None

def get_yoloe() -> YOLOE:
    global _yoloe_model
    if _yoloe_model is None:
        log.info(f"Carregando YOLOE ({YOLOE_MODEL})...")
        _yoloe_model = YOLOE(YOLOE_MODEL)
    return _yoloe_model


def run_yoloe(img: Image.Image) -> list[DetectedObject]:
    """Roda YOLOE prompt-free e retorna lista de DetectedObject com bbox normalizada."""
    model = get_yoloe()
    w, h  = img.size

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
                x1=xyxy[0] / w,
                y1=xyxy[1] / h,
                x2=xyxy[2] / w,
                y2=xyxy[3] / h,
            ),
            source="yoloe",
            yoloe_conf=float(box.conf[0]),
        ))

    log.info(f"YOLOE: {len(objects)} objeto(s)")
    return objects
