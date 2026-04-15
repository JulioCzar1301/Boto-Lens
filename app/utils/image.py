"""
Utilitários para manipulação de imagens em base64 e PIL.

Fornece funções para redimensionamento, conversão entre formatos
e preparação de imagens para envio aos modelos.
"""

import base64
import io
import logging
from PIL import Image

log = logging.getLogger(__name__)

# Constantes para processamento de imagens
IMAGE_MAX_SIDE = 1024
IMAGE_JPEG_QUALITY = 82


def resize_base64_image(
    b64: str,
    max_side: int = IMAGE_MAX_SIDE,
    quality: int = IMAGE_JPEG_QUALITY,
) -> str:
    """
    Redimensiona uma imagem em base64 mantendo a proporção original.

    Se a imagem exceder o tamanho máximo em qualquer dimensão,
    ela é redimensionada proporcionalmente. O resultado é codificado
    como JPEG em base64.

    Args:
        b64 (str): String base64 da imagem (com ou sem prefixo data URL).
        max_side (int): Tamanho máximo para o maior lado da imagem.
        quality (int): Qualidade do JPEG (1-100, padrão 82).

    Returns:
        str: Imagem redimensionada em base64 (sem prefixo).
    """
    # Remove prefixo data URL se presente
    if "," in b64:
        b64 = b64.split(",")[1]

    # Decodifica e converte para RGB
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    # Redimensiona se necessário
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))

    # Codifica como JPEG em base64
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_pil(b64: str) -> Image.Image:
    """
    Converte uma string base64 para objeto PIL Image.

    Aceita strings com ou sem prefixo data URL.

    Args:
        b64 (str): String base64 da imagem.

    Returns:
        Image.Image: Objeto PIL Image no formato RGB.
    """
    if "," in b64:
        b64 = b64.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")