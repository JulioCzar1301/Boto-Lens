FROM ultralytics/ultralytics:latest

WORKDIR /app

# Só instala os pacotes LEVES da sua API
RUN pip install --timeout=300 fastapi uvicorn[standard] httpx pydantic python-dotenv
# CLIP é necessário para YOLOE text prompts (get_text_pe)
RUN pip install --timeout=300 "git+https://github.com/ultralytics/CLIP.git"

COPY app/ .

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]