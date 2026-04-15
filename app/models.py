"""
Definição das classes de dados (dataclasses) utilizadas no sistema de detecção de objetos.

Este módulo contém as estruturas básicas para representar bounding boxes e objetos detectados.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BBox:
    """
    Representa uma bounding box em coordenadas normalizadas [0, 1].

    Atributos:
        x1 (float): Coordenada X do canto superior esquerdo.
        y1 (float): Coordenada Y do canto superior esquerdo.
        x2 (float): Coordenada X do canto inferior direito.
        y2 (float): Coordenada Y do canto inferior direito.
    """
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        """Calcula a área da bounding box."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def iou(self, other: 'BBox') -> float:
        """
        Calcula a Intersection over Union (IoU) entre duas bounding boxes.

        Args:
            other (BBox): Outra bounding box para comparar.

        Returns:
            float: Valor da IoU entre 0 e 1.
        """
        x1_inter = max(self.x1, other.x1)
        y1_inter = max(self.y1, other.y1)
        x2_inter = min(self.x2, other.x2)
        y2_inter = min(self.y2, other.y2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area_union = self.area() + other.area() - area_inter
        return area_inter / area_union if area_union > 0 else 0.0


@dataclass
class DetectedObject:
    """
    Representa um objeto detectado na imagem.

    Atributos:
        label (str): Nome do objeto.
        score (float): Confiança da detecção (0 a 1).
        bbox (BBox): Bounding box do objeto.
        source (str): Origem da detecção ('qwen', 'yoloe' ou 'fused').
        yoloe_conf (Optional[float]): Confiança específica do YOLOE (se aplicável).
    """
    label: str
    score: float
    bbox: BBox
    source: str
    yoloe_conf: Optional[float] = None