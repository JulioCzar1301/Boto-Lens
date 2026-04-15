import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Optional

from PIL import Image
from ultralytics import YOLOE

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configurações YOLOE
# ──────────────────────────────────────────────

IMAGE_MAX_SIDE     = 1080
IMAGE_JPEG_QUALITY = 82



# ──────────────────────────────────────────────
# Imagem
# ──────────────────────────────────────────────

def resize_base64_image(
    b64: str,
    max_side: int = IMAGE_MAX_SIDE,
    quality: int = IMAGE_JPEG_QUALITY,
) -> str:
    """Redimensiona uma imagem base64 mantendo proporção e recodifica como JPEG."""
    if "," in b64:
        b64 = b64.split(",")[1]

    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_pil(b64: str) -> Image.Image:
    """Converte base64 (com ou sem prefixo data URL) para PIL.Image."""
    if "," in b64:
        b64 = b64.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


