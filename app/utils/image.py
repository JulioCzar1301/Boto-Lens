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
IMAGE_MAX_SIDE = 768  # Reduzido de 1024 para economizar tokens
IMAGE_JPEG_QUALITY = 60  # Reduzido de 82 para maior compressão


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


def pil_to_b64(
    img: Image.Image,
    quality: int = IMAGE_JPEG_QUALITY,
) -> str:
    """
    Converte um objeto PIL Image para string base64.

    Args:
        img (Image.Image): Objeto PIL Image.
        quality (int): Qualidade do JPEG (1-100).

    Returns:
        str: Imagem em base64 (sem prefixo).
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def draw_yoloe_detections(
    img: Image.Image,
    detections: list,
    line_width: int = 3,
    font_size: int = 14,
) -> Image.Image:
    """
    Desenha bounding boxes e labels do YOLOE sobre a imagem.

    Args:
        img (Image.Image): Imagem PIL no formato RGB.
        detections (list): Lista de DetectedObject com labels e bboxes normalizadas.
        line_width (int): Largura da linha dos bounding boxes.
        font_size (int): Tamanho da fonte dos labels.

    Returns:
        Image.Image: Imagem com os bounding boxes desenhados.
    """
    from PIL import ImageDraw, ImageFont

    # Cria uma cópia para não alterar a imagem original
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    width, height = img.size

    # Define cores para os boxes (verde brilhante para visibilidade)
    box_color = (0, 255, 0)  # Verde
    text_bg_color = (0, 0, 0)  # Preto para background do texto
    text_color = (255, 255, 255)  # Branco para texto

    try:
        # Tenta usar uma fonte legível, se não encontrar usa padrão
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for detection in detections:
        # Converte coordenadas normalizadas para pixels
        bbox = detection.bbox
        x1 = int(bbox.x1 * width)
        y1 = int(bbox.y1 * height)
        x2 = int(bbox.x2 * width)
        y2 = int(bbox.y2 * height)

        # Desenha o bounding box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

        # Prepara o texto com label e score
        label_text = f"{detection.label} ({detection.score:.2f})"

        # Desenha background para o texto
        bbox_text = draw.textbbox((x1, y1 - font_size - 4), label_text, font=font)
        draw.rectangle(bbox_text, fill=text_bg_color)

        # Desenha o texto
        draw.text((x1, y1 - font_size - 4), label_text, fill=text_color, font=font)

    return img_copy


# Paleta de cores distintas para anotação numerada (BGR→RGB)
_INDEXED_COLORS = [
    (255,  60,  60),  # vermelho
    (60,  220,  60),  # verde
    (60,  100, 255),  # azul
    (255, 210,  30),  # amarelo
    (220,  60, 220),  # magenta
    (30,  210, 220),  # ciano
    (255, 140,  30),  # laranja
    (140,  60, 255),  # violeta
    (30,  180, 255),  # azul claro
    (255,  60, 140),  # rosa
    (30,  200, 120),  # verde-água
    (200, 200,  60),  # oliva
]


def draw_indexed_detections(
    img: Image.Image,
    indexed_detections: list[tuple[int, object]],
    line_width: int = 3,
    font_size: int = 20,
) -> Image.Image:
    """
    Desenha bounding boxes NUMERADAS com cores distintas por índice.

    Cada box exibe apenas o índice original do objeto (ex: "0", "3", "7"),
    permitindo que o Qwen correlacione visualmente cada box com a entrada
    correspondente no JSON de detecções do YOLOE.

    Args:
        img (Image.Image): Imagem PIL no formato RGB.
        indexed_detections (list[tuple[int, DetectedObject]]): Lista de tuplas
            (índice_original, DetectedObject) a serem anotadas.
        line_width (int): Largura da linha dos bounding boxes.
        font_size (int): Tamanho da fonte do número índice.

    Returns:
        Image.Image: Cópia da imagem com as boxes numeradas desenhadas.
    """
    from PIL import ImageDraw, ImageFont

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = img.size

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except Exception:
        font = ImageFont.load_default()

    for seq, (original_idx, detection) in enumerate(indexed_detections):
        color = _INDEXED_COLORS[seq % len(_INDEXED_COLORS)]
        bbox = detection.bbox

        # Converte para pixels
        x1 = int(bbox.x1 * width)
        y1 = int(bbox.y1 * height)
        x2 = int(bbox.x2 * width)
        y2 = int(bbox.y2 * height)

        # Bounding box colorida
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Rótulo numérico com fundo da mesma cor da box
        label_text = str(original_idx)
        padding = 4
        text_x = x1 + padding
        text_y = y1 + padding

        text_bbox = draw.textbbox((text_x, text_y), label_text, font=font)
        # Fundo sólido para legibilidade
        draw.rectangle(
            [text_bbox[0] - padding, text_bbox[1] - padding,
             text_bbox[2] + padding, text_bbox[3] + padding],
            fill=color,
        )
        # Número em branco ou preto dependendo da luminância da cor
        r, g, b = color
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (0, 0, 0) if lum > 160 else (255, 255, 255)
        draw.text((text_x, text_y), label_text, fill=text_color, font=font)

    return img_copy