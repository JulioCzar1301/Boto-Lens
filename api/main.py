from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
from SYSTEM_INSTRUCTION import SYSTEM_INSTRUCTION

app = FastAPI()

VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000") + "/v1/chat/completions"

class Prompt(BaseModel):
    image: str

@app.post("/detection")
async def gerar_texto(body: Prompt):
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(VLLM_URL, json={
            "model": "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit",
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Descreva essa imagem"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{body.image}"
                            }
                        }
                    ]
                }
            ]
        })
        return response.json()

@app.get("/health")
async def health():
    return {"status": "ok"}