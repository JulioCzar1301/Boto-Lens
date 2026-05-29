"""
Schemas Pydantic para os endpoints da API.
"""

from pydantic import BaseModel


class Prompt(BaseModel):
    """Payload padrão: somente a imagem em base64."""
    image: str


class PromptSys(BaseModel):
    """Payload com prompt de sistema customizado."""
    image: str
    prompt: str
