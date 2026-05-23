"""
API principal para detecção de objetos.

Endpoints:
- /detection/fused   [PRINCIPAL] YOLOE → filtros → Qwen por crop (contexto + crop)
- /detection         Qwen autônomo
- /detection/sys_prompt  Qwen com prompt customizado
- /detection/sequential  Pipeline legado
- /health

Arquitetura /detection/fused:
  1. YOLOE detecta bboxes (labels ignorados).
  2. Filtro de área: 0.5% ≤ área ≤ 75%.
  3. Filtro de contenção: bbox ≥70% dentro de outra maior → sub-parte → descartado.
  4. Para cada candidato, gera:
       a) Imagem completa com o bbox desenhado em vermelho (contexto espacial).
       b) Crop do objeto com padding (detalhe isolado).
  5. Envia AMBAS as imagens ao Qwen em paralelo via asyncio.gather.
  6. Qwen usa o contexto para validar se o bbox isola um único objeto real.
  7. Score < 60% → descartado.
  8. Deduplicação pós-Qwen com contenção: bbox menor dentro de maior → mantém menor (mais preciso).
  8b. Ranking final por score × centralidade.
  9. Retorna top 5.
"""

import math
import os
import asyncio
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image, ImageDraw

from system_instruction import CROP_INSTRUCTION, SYSTEM_INSTRUCTION, SEQUENTIAL_INSTRUCTION
from models import DetectedObject, BBox
from services.yoloe import run_yoloe, get_yoloe
from services.qwen import parse_crop_response, parse_qwen_response, parse_qwen_refine_response
from services.fusion import serialize_detections
from utils.image import resize_base64_image, b64_to_pil, pil_to_b64

app = FastAPI()


# ──────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("Inicializando... Carregando modelo YOLO...")
    try:
        get_yoloe()
        print("Modelo YOLO carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo YOLO: {e}")
        raise

VLLM_URL   = os.getenv("VLLM_URL",   "http://vllm:8000") + "/v1/chat/completions"
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class Prompt(BaseModel):
    image: str

class PromptSys(BaseModel):
    image: str
    prompt: str


# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────

_MIN_AREA      = 0.005
_MAX_AREA      = 0.75
_CROP_PADDING  = 0.05
_BOX_COLOR     = (255, 50, 50)   # vermelho para o bbox na imagem completa
_BOX_WIDTH     = 4               # espessura do retângulo em pixels


# ──────────────────────────────────────────────
# Geometria
# ──────────────────────────────────────────────

def _bbox_area(bbox: BBox) -> float:
    return max(0.0, (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1))


def _intersection_area(a: BBox, b: BBox) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _containment(a: BBox, b: BBox) -> float:
    area_a = _bbox_area(a)
    return 0.0 if area_a == 0 else _intersection_area(a, b) / area_a


def _centrality(bbox: BBox) -> float:
    """Centralidade: 1.0 quando o centro do bbox coincide com o centro da imagem."""
    cx = (bbox.x1 + bbox.x2) / 2
    cy = (bbox.y1 + bbox.y2) / 2
    return max(0.0, 1.0 - 2 * max(abs(cx - 0.5), abs(cy - 0.5)))


def _deduplicate_results(objects: list[DetectedObject]) -> list[DetectedObject]:
    """
    Deduplicação pós-Qwen com lógica de contenção.

    - Se containment(A em B) > 0.5 ou containment(B em A) > 0.5:
        → contenção: mantém o bbox MENOR (mais preciso).
    - Se IoU > 0.20 sem contenção clara:
        → mantém o de maior score.
    """
    sorted_objs = sorted(objects, key=lambda o: o.score, reverse=True)
    kept: list[DetectedObject] = []

    for obj in sorted_objs:
        discard_obj = False
        to_remove_from_kept: list[int] = []

        for i, k in enumerate(kept):
            obj_area = _bbox_area(obj.bbox)
            k_area   = _bbox_area(k.bbox)
            obj_in_k = _containment(obj.bbox, k.bbox)   # fração de obj dentro de k
            k_in_obj = _containment(k.bbox, obj.bbox)   # fração de k dentro de obj

            has_containment = obj_in_k > 0.5 or k_in_obj > 0.5
            has_iou         = obj.bbox.iou(k.bbox) > 0.20

            if has_containment or has_iou:
                if has_containment:
                    if obj_area < k_area:
                        # obj é o menor/mais preciso → remove k (maior/genérico)
                        to_remove_from_kept.append(i)
                        print(f"Dedup contenção: '{k.label}' removido (área={k_area:.3f})"
                              f" → mantém '{obj.label}' (área={obj_area:.3f})")
                    else:
                        # k é o menor/mais preciso → descarta obj
                        discard_obj = True
                        print(f"Dedup contenção: '{obj.label}' descartado (área={obj_area:.3f})"
                              f" → mantém '{k.label}' (área={k_area:.3f})")
                        break
                else:
                    # Sobreposição por IoU sem contenção clara → maior score vence
                    if obj.score > k.score:
                        to_remove_from_kept.append(i)
                        print(f"Dedup IoU: '{k.label}' removido (score={k.score:.2f})"
                              f" → mantém '{obj.label}' (score={obj.score:.2f})")
                    else:
                        discard_obj = True
                        print(f"Dedup IoU: '{obj.label}' descartado (score={obj.score:.2f})")
                        break

        if discard_obj:
            continue

        for i in sorted(to_remove_from_kept, reverse=True):
            kept.pop(i)

        kept.append(obj)

    return kept


# ──────────────────────────────────────────────
# Geração de imagens para o Qwen
# ──────────────────────────────────────────────

def _draw_bbox_on_image(img: Image.Image, bbox: BBox) -> Image.Image:
    """
    Retorna uma cópia da imagem completa com o bbox destacado em vermelho.
    O modelo usa essa imagem para validar o contexto espacial do objeto.
    """
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = img.size
    x1, y1 = int(bbox.x1 * w), int(bbox.y1 * h)
    x2, y2 = int(bbox.x2 * w), int(bbox.y2 * h)
    for t in range(_BOX_WIDTH):
        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=_BOX_COLOR)
    return annotated


def _crop_object(img: Image.Image, bbox: BBox) -> Image.Image:
    """Crop com padding, sem sair dos limites."""
    w, h = img.size
    pad_x = (bbox.x2 - bbox.x1) * _CROP_PADDING
    pad_y = (bbox.y2 - bbox.y1) * _CROP_PADDING
    x1 = max(0.0, bbox.x1 - pad_x)
    y1 = max(0.0, bbox.y1 - pad_y)
    x2 = min(1.0, bbox.x2 + pad_x)
    y2 = min(1.0, bbox.y2 + pad_y)
    return img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))


async def _classify_crop(
    client: httpx.AsyncClient,
    img: Image.Image,
    obj: DetectedObject,
) -> DetectedObject | None:
    """
    Envia ao Qwen:
      1. Imagem completa com bbox destacado (contexto espacial).
      2. Crop isolado do objeto (detalhe).
    Qwen usa as duas para validar se o bbox isola um único objeto real.
    """
    annotated_b64 = pil_to_b64(_draw_bbox_on_image(img, obj.bbox))
    crop_b64      = pil_to_b64(_crop_object(img, obj.bbox))

    try:
        response = await client.post(
            VLLM_URL,
            json={
                "model": VLLM_MODEL,
                "messages": [
                    {"role": "system", "content": CROP_INSTRUCTION},
                    {
                        "role": "user",
                        "content": [
                            # Imagem 1: cena completa com bbox destacado
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{annotated_b64}"}},
                            # Imagem 2: crop isolado
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
                            {"type": "text",
                             "text": (
                                 "A primeira imagem mostra a cena completa com um retângulo vermelho "
                                 "destacando a região do objeto. "
                                 "A segunda imagem é o recorte (crop) dessa região. "
                                 "O que é o objeto dentro do retângulo vermelho?"
                             )},
                        ],
                    },
                ],
                "max_tokens": 64,
            },
        )
        result = response.json()
    except Exception as e:
        print(f"Erro ao chamar vLLM (crop): {e}")
        return None

    label, score = parse_crop_response(result)
    if not label:
        print(f"Descartado: score={score:.2f} bbox={obj.bbox}")
        return None

    print(f"Classificado: '{label}' score={score:.2f}")
    return DetectedObject(
        label=label,
        score=score,
        bbox=obj.bbox,
        source="fused",
        yoloe_conf=obj.yoloe_conf,
    )


# ──────────────────────────────────────────────
# Helpers legados
# ──────────────────────────────────────────────

async def _call_vllm(client: httpx.AsyncClient, image_b64: str, system: str) -> dict:
    response = await client.post(
        VLLM_URL,
        json={
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": "Descreva essa imagem"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]},
            ],
        },
    )
    return response.json()


def _bbox_prominence(bbox: BBox) -> float:
    area = _bbox_area(bbox)
    area_score = min(1.0, math.sqrt(area) * 2)
    cx = (bbox.x1 + bbox.x2) / 2
    cy = (bbox.y1 + bbox.y2) / 2
    centrality = max(0.0, 1.0 - 2 * max(abs(cx - 0.5), abs(cy - 0.5)))
    return round(0.6 * area_score + 0.4 * centrality, 4)


async def _call_vllm_sequential(
    client: httpx.AsyncClient,
    img: Image.Image,
    yoloe_objects: list[DetectedObject],
) -> dict:
    import json as _json
    filtered = [
        (i, obj) for i, obj in enumerate(yoloe_objects)
        if _bbox_area(obj.bbox) >= 0.005
    ] or list(enumerate(yoloe_objects))
    clean_b64 = pil_to_b64(img)
    payload = _json.dumps(
        [{"index": i, "label": obj.label,
          "conf": round(obj.yoloe_conf or obj.score, 4),
          "bbox_norm": {"x1": round(obj.bbox.x1, 4), "y1": round(obj.bbox.y1, 4),
                        "x2": round(obj.bbox.x2, 4), "y2": round(obj.bbox.y2, 4)},
          "area": round(_bbox_area(obj.bbox), 4),
          "prominence": _bbox_prominence(obj.bbox)}
         for i, obj in filtered],
        ensure_ascii=False,
    )
    response = await client.post(
        VLLM_URL,
        json={
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": SEQUENTIAL_INSTRUCTION},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": f"YOLOE detections:\n{payload}\n\nRename in Brazilian Portuguese."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}"}},
                ]},
            ],
        },
    )
    return response.json()


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.post("/detection/fused")
async def detection_fused(body: Prompt) -> dict:
    """
    Pipeline principal.

    Filtros:
      1. Área: 0.5% ≤ área ≤ 75%
      2. Qwen com contexto (imagem anotada + crop): score < 60% → descartado
      3. Dedup pós-Qwen com contenção: bbox menor dentro de maior → mantém menor
         Ou IoU > 0.20 → mantém maior score
      4. Ranking final: score × (0.6 + 0.4 × centralidade)
    """
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)

    loop = asyncio.get_event_loop()
    yoloe_objects = await loop.run_in_executor(None, run_yoloe, img)

    if not yoloe_objects:
        return serialize_detections([], too_many=False)

    candidates = [o for o in yoloe_objects if _MIN_AREA <= _bbox_area(o.bbox) <= _MAX_AREA] \
                 or yoloe_objects

    print(f"YOLOE: {len(yoloe_objects)} → {len(candidates)} candidatos após filtro de área")

    if not candidates:
        return serialize_detections([], too_many=False)

    async with httpx.AsyncClient(timeout=120.0) as client:
        results = await asyncio.gather(
            *[_classify_crop(client, img, obj) for obj in candidates]
        )

    classified = [r for r in results if r is not None]
    deduped    = _deduplicate_results(classified)
    # Ranking final: score do Qwen × bônus de centralidade
    final = sorted(deduped, key=lambda o: o.score * (0.6 + 0.4 * _centrality(o.bbox)), reverse=True)[:5]

    print(f"Resultado final: {len(final)} objeto(s)")
    return serialize_detections(final)


@app.post("/detection")
async def detection(body: Prompt) -> dict:
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        return serialize_detections(
            parse_qwen_response(await _call_vllm(client, image_b64, SYSTEM_INSTRUCTION)),
            too_many=False,
        )


@app.post("/detection/sys_prompt")
async def detection_sys(body: PromptSys) -> dict:
    image_b64 = resize_base64_image(body.image)
    async with httpx.AsyncClient(timeout=120.0) as client:
        return serialize_detections(
            parse_qwen_response(await _call_vllm(client, image_b64, body.prompt)),
            too_many=False,
        )


@app.post("/detection/sequential")
async def detection_sequential(body: Prompt) -> dict:
    image_b64 = resize_base64_image(body.image)
    img = b64_to_pil(image_b64)
    loop = asyncio.get_event_loop()
    yoloe_objects = await loop.run_in_executor(None, run_yoloe, img)
    if not yoloe_objects:
        return serialize_detections([], too_many=False)
    async with httpx.AsyncClient(timeout=120.0) as client:
        vllm_response = await _call_vllm_sequential(client, img, yoloe_objects)
    return serialize_detections(parse_qwen_refine_response(vllm_response, yoloe_objects))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
