FROM ultralytics/ultralytics:latest

WORKDIR /app

# Só instala os pacotes LEVES da sua API
RUN pip install --timeout=300 fastapi uvicorn[standard] httpx pydantic python-dotenv && python -c "from ultralytics import YOLOE; YOLOE('yoloe-11l-seg-pf.pt')"
COPY app/ .

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]