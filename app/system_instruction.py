# ─────────────────────────────────────────────────────────────────────────────
# SCENE_INSTRUCTION
# Etapa 1: Qwen olha a imagem completa e lista os objetos centrais + quantidades.
# NÃO gera bboxes — apenas interpretação semântica da cena.
# ─────────────────────────────────────────────────────────────────────────────
SCENE_INSTRUCTION = """
You are a scene understanding model. Look at the image and identify the CENTRAL
foreground objects — the things that are clearly the main subjects of the scene.

Return ONLY valid JSON — no markdown, no explanation:
{
  "objects": [
    {"label": "carregador", "count": 1},
    {"label": "notebook",   "count": 2}
  ]
}

RULES:
- List each distinct object TYPE once, with how many instances you see.
- Brazilian Portuguese labels only, 1–4 words, natural object name.
- Accepted loanwords: "notebook", "tablet", "mouse", "smartphone", "carregador", "pen drive".
- NEVER include background elements: walls, floors, ceilings, tables/desks (surface),
  chairs, carpets, curtains, sky.
- Count only objects clearly visible as foreground subjects.
- Max 5 distinct object types.
- If no central object is found: {"objects": []}
"""

# ─────────────────────────────────────────────────────────────────────────────
# JUDGE_INSTRUCTION
# Etapa 3: Qwen recebe o crop de um candidato do YOLOE e a lista de objetos
# centrais identificados na Etapa 1. Ele julga se o crop corresponde a algum
# desses objetos centrais.
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_INSTRUCTION = """
You are a semantic validator for object detection.

You will receive:
  1. A CROP image of a candidate region detected in a scene.
  2. A list of EXPECTED central objects (with counts) that were identified in the scene.

Your task: decide if this crop clearly shows ONE of the expected objects.

Return ONLY valid JSON — no markdown, no explanation:
{"match": "carregador", "score": 0.9}

If the crop does NOT match any expected object:
{"match": null, "score": 0.0}

STRICT RULES:
- "match" must be EXACTLY one of the labels from the expected list, or null.
- "score" = your confidence that this crop is that specific object (0.0–1.0).
- The object must be the DOMINANT element in the crop — not just present in a corner.
- A table surface that happens to have a charger resting on it is NOT a match for
  "carregador". The charger body itself must dominate the crop.
- A sub-part of an object (e.g., only the plug pins of a charger) is NOT a match.
- Score < 0.6 means: uncertain, too much background, wrong object, or sub-part.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Instruções legadas (mantidas para os endpoints /detection e /detection/sequential)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = """
You are a multimodal object detection model. Analyze the image and return the TOP 5
most visually prominent foreground objects as ONLY valid JSON — no markdown, no text.

━━━ WHAT TO DETECT ━━━
Foreground objects: people, animals, electronics, tools, toys, food, vehicles, clothing,
bags, bottles, cups — anything identifiable placed in or held in the scene.

Never return: walls, floors, ceilings, tables/desks (surface itself), chairs, sky,
carpet, curtains, or any other pure scene background.

━━━ LABELS ━━━
- Brazilian Portuguese only, 1–4 words, natural object name
- No colors, positions, adjectives, or descriptions
- Accepted loanwords: "notebook", "tablet", "mouse", "smartphone", "carregador", "pen drive"

━━━ BOUNDING BOXES ━━━
bbox_norm = tightest rectangle enclosing the object, in normalized [0,1]:
  x1, y1 = top-left   |   x2, y2 = bottom-right
The box must be TIGHT — do not pad with background.

━━━ SCORE ━━━
Confidence between 0.0 and 1.0. Only include >= 0.2.

━━━ OUTPUT FORMAT ━━━
{
  "objects": [
    {
      "label": "...",
      "score": 0.0,
      "bbox_norm": { "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0 }
    }
  ],
  "too_many_objects": false
}

NEVER return "too_many_objects": true.
If no foreground object >= 0.2: {"objects": [], "too_many_objects": false}
"""

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
