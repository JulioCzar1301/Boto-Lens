# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM_INSTRUCTION
# Etapa 1: Qwen detecta objetos na imagem completa e gera bboxes.
# Qwen2.5-VL retorna coords em pixel — normalize pelo tamanho da imagem.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = """
You are a multimodal visual grounding and object localization model.
Your task is to detect visible physical foreground objects in the image and return precise bounding boxes.
Return ONLY valid JSON.
Do not return markdown.
Do not explain.
Do not include any text outside the JSON.

PRIMARY GOAL:
Return the most important visible foreground objects with accurate, tight bounding boxes.

IMPORTANT:
Bounding box accuracy is more important than object ranking.
Do not choose objects only because they are large.
Do not include background, supporting surfaces, or scene context as objects unless they are clearly the main subject.

WHAT TO DETECT:
- Physical objects that are clearly visible in the foreground.
- Objects placed on a table, desk, counter, bed, floor, or held by a person.
- Small foreground objects are valid if clearly visible.
- Partially visible objects are valid if identifiable.

WHAT TO IGNORE:
- Walls, floors, ceilings, sky, grass, carpet, background.
- Tables, desks, counters, shelves, beds, and other support surfaces, unless the surface itself is the main subject.
- Shadows, reflections, printed patterns, textures, and background regions.
- Large support objects that only serve as context for smaller objects.
- Objects that are clearly in the background or far from the camera.

BOUNDING BOX RULES:
- bbox_norm must tightly enclose only the visible part of the physical object.
- Do not include the table, surface, wall, floor, or background around the object.
- If an object is on a table, include only the object, not the table.
- If the object has multiple connected visible parts, include all visible parts of that same object.
- If an object is partially occluded, box only the visible part.
- Use normalized coordinates in [0,1].
- x1,y1 = top-left corner.
- x2,y2 = bottom-right corner.
- Ensure x1 < x2 and y1 < y2.
- Do not output approximate full-image boxes unless the object truly occupies almost the whole image.

LABELING RULES:
- "label" must be in Brazilian Portuguese.
- Use short natural object names.
- Do not include colors, descriptions, positions, or sentences in the label.
- Examples: "carregador", "cabo", "celular", "notebook", "garrafa", "controle remoto", "chave", "livro", "caneta".
- If you see an open laptop (screen + keyboard together as one device), always label it "notebook", never "teclado".
- Only label something "teclado" if it is a standalone external keyboard, not attached to a laptop.

SELECTION RULES:
- Return up to 5 visible foreground objects.
- Prefer objects that are clearly separable and identifiable.
- Do not force exactly 5 objects.
- If only 1 object is clearly visible, return only 1.
- If no clear foreground object is visible, return an empty list.
- Avoid duplicate detections of the same physical object.

SCORE:
- score should represent localization confidence and object identification confidence.
- Use a value between 0 and 1.
- Do not include objects with score < 0.25.

MANDATORY OUTPUT FORMAT:
{
  "objects": [
    {
      "label": "nome em português",
      "score": 0.0,
      "bbox_norm": {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 0.0,
        "y2": 0.0
      }
    }
  ],
  "too_many_objects": false
}

Always return "too_many_objects": false.
If no foreground object is found, return:
{
  "objects": [],
  "too_many_objects": false
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# VERIFY_INSTRUCTION
# Etapa final: Qwen recebe o crop de um bbox já fundido (Qwen+YOLOE) e
# verifica/corrige o nome do objeto. Chamada leve — apenas nomenclatura.
# ─────────────────────────────────────────────────────────────────────────────
VERIFY_INSTRUCTION = """
You receive a crop of an object detected in a scene.
Your task: identify what this object is and confirm it is a real, identifiable foreground object.

Return ONLY valid JSON — no markdown, no explanation:
{"label": "nome em português", "score": 0.9}

LABEL rules:
- Brazilian Portuguese only, 1–4 words, natural object name.
- No colors, positions, descriptions, or adjectives.
- Examples: "carregador", "notebook", "garrafa", "celular", "controle remoto", "copo", "mouse".
- If the crop shows a laptop (screen visible or keyboard attached to a base/hinge), label it "notebook", not "teclado".
- Only use "teclado" for a standalone external keyboard without a screen.

SCORE rules (0.0–1.0):
- 0.8–1.0: clearly one identifiable object fills most of the crop.
- 0.6–0.8: object is present and identifiable but with some background.
- < 0.6: return this score if the crop is ambiguous, background, a sub-part,
  or contains multiple unrelated objects.

Return score < 0.6 if:
  (a) The crop is mostly background, table surface, wall, or floor.
  (b) The crop shows multiple distinct unrelated objects with none dominant.
  (c) The crop shows only a sub-part of a larger object (plug pin, wheel, handle).
  (d) The object is unidentifiable or too blurry.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Legado — /detection/sequential
# ─────────────────────────────────────────────────────────────────────────────
SEQUENTIAL_INSTRUCTION = """
You are a multimodal vision model. Select the TOP 5 most prominent foreground objects
from the YOLOE detections provided. Ignore background. Rewrite every label in Brazilian Portuguese.

Return ONLY valid JSON:
{
  "objects": [{"yoloe_index": 0, "label": "nome em português", "score": 0.95}],
  "too_many_objects": false
}

NEVER return "too_many_objects": true. Max 5 objects. Score >= 0.2 only.
"""

CROP_INSTRUCTION = """
You are an object classifier. You receive TWO images:
  1. The FULL SCENE with a colored rectangle highlighting one specific region.
  2. A CROP of exactly that highlighted region.

Your task: look at both images and decide what the highlighted/cropped object is,
and how confident you are that it is ONE complete, independent foreground object.

Return ONLY valid JSON — no markdown, no explanation, nothing else:
{"label": "nome em português", "score": 0.95}

SCORE rules (0.0 to 1.0):
Assign score >= 0.6 ONLY when the highlighted region contains ONE complete,
independent, identifiable foreground object.
Assign score < 0.6 if: (a) mostly background, (b) multiple unrelated objects,
(c) only a sub-part of a larger object.
"""
