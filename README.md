# Serviço de Detecção de Objetos com Fusão Qwen + YOLOE

Uma API FastAPI que combina a compreensão semântica do modelo **Qwen** (via vLLM) com a precisão espacial do **YOLOE** para detecção robusta de objetos em imagens.

## 📋 Visão Geral

A aplicação oferece múltiplos endpoints para detecção de objetos:

- **`/detection`** - Detecção usando apenas Qwen (resposta bruta)
- **`/detection/sys_prompt`** - Detecção com Qwen e prompt customizado
- **`/detection/fused`** - **Recomendado**: Detecção com fusão Qwen + YOLOE
- **`/health`** - Health check da API

## 🏗️ Arquitetura

### Componentes Principais

```
┌─────────────────────────────────────────────┐
│            FastAPI Application              │
│                  (Port 8001)                │
└─────────────────────────────────────────────┘
           │                    │
           ▼                    ▼
      ┌──────────┐         ┌─────────┐
      │  Qwen    │         │ YOLOE   │
      │ (vLLM)   │         │ Service │
      │ Port 8000│         │(Built-in)
      └──────────┘         └─────────┘
           │                    │
           └────────┬───────────┘
                    ▼
           ┌─────────────────┐
           │ Fusion Service  │
           │    (IoU Match)  │
           └─────────────────┘
```

### Fluxo de Detecção

1. **Entrada**: Imagem em Base64
2. **Processamento Paralelo**:
   - Qwen detecta objetos semanticamente (labels + scores)
   - YOLOE detecta objetos com precisão espacial (bboxes)
3. **Fusão**: Combina resultados usando **IoU (Intersection over Union)**
4. **Saída**: Objetos detectados com labels semânticos + bboxes precisas

## 🚀 Como Executar

### Pré-requisitos

- Docker
- Docker Compose
- GPU NVIDIA (para melhor performance)

### Opção 1: Docker Compose (Recomendado)

```bash
docker-compose up --build
```

Isso iniciará:
- **vLLM service** (Qwen) na porta 8000
- **FastAPI** na porta 8001

### Opção 2: Execução Local

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

(Nota: Qwen/vLLM precisa estar executando separadamente)

## 📦 Dependências

- **FastAPI** - Framework web
- **Uvicorn** - Servidor ASGI
- **Pydantic** - Validação de dados
- **httpx** - Cliente HTTP assíncrono
- **Ultralytics** - YOLOE (pré-instalado na imagem)
- **vLLM** - Servidor LLM para Qwen (container separado)

## 🔌 Endpoints

### 1. `/detection` - Apenas Qwen

**Método**: `POST`

**Request**:
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
}
```

**Response**:
```json
{
  "detections": [
    {
      "label": "ball",
      "score": 0.85,
      "bbox": {
        "x1": 0.3,
        "y1": 0.4,
        "x2": 0.7,
        "y2": 0.8
      },
      "source": "qwen"
    }
  ],
  "processing_time_ms": 1234
}
```

### 2. `/detection/sys_prompt` - Qwen com Prompt Customizado

**Método**: `POST`

**Request**:
```json
{
  "image": "base64_image_string",
  "prompt": "Procure apenas por brinquedos nesta imagem"
}
```

**Response**: Similar ao endpoint anterior

### 3. `/detection/fused` - Fusão Qwen + YOLOE

**Método**: `POST`

**Request**:
```json
{
  "image": "base64_image_string"
}
```

**Response**:
```json
{
  "detections": [
    {
      "label": "toy",
      "score": 0.92,
      "bbox": {
        "x1": 0.25,
        "y1": 0.35,
        "x2": 0.75,
        "y2": 0.85
      },
      "source": "fused"
    },
    {
      "label": "background",
      "score": 0.45,
      "bbox": {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 1.0,
        "y2": 1.0
      },
      "source": "qwen"
    }
  ],
  "processing_time_ms": 2567
}
```

**Valores possíveis de `source`**:
- `"fused"` - Combinado (labels Qwen + bbox YOLOE)
- `"qwen"` - Apenas Qwen (sem match com YOLOE)
- `"yoloe"` - Apenas YOLOE (sem match com Qwen)

### 4. `/health` - Health Check

**Método**: `GET`

**Response**:
```json
{
  "status": "ok",
  "vllm_status": "healthy",
  "uptime_seconds": 3600
}
```

## 📐 Estrutura do Projeto

```
server/
├── Dockerfile              # Container para API FastAPI
├── docker-compose.yml      # Orquestração de containers (vLLM + API)
├── requirements.txt        # Dependências Python
├── class.puml              # Diagrama de classes (PlantUML)
├── README.md               # Este arquivo
└── app/
    ├── main.py             # Aplicação principal e endpoints
    ├── models.py           # Definições de dados (BBox, DetectedObject)
    ├── system_instruction.py # Prompt do Qwen
    ├── utils.py            # Utilitários gerais
    ├── services/           # Serviços de detecção
    │   ├── __init__.py
    │   ├── fusion.py       # Fusão de detecções (IoU matching)
    │   ├── qwen.py         # Interface com Qwen/vLLM
    │   └── yoloe.py        # Interface com YOLOE
    └── utils/              # Utilitários específicos
        ├── __init__.py
        └── image.py        # Processamento de imagens
```

## 🔧 Configurações

### Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `VLLM_URL` | `http://vllm:8000` | URL do servidor vLLM |
| `IOU_THRESHOLD` | `0.15` | Limiar mínimo de IoU para fusão |
| `MIN_SCORE` | `0.2` | Score mínimo para incluir detecção |

### Configuração do vLLM (docker-compose.yml)

```yaml
- "--model": "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"
- "--dtype": "bfloat16"
- "--max-model-len": "4096"
- "--gpu-memory-utilization": "0.95"
```

## 📚 Modelos de Dados

### BBox
```python
@dataclass
class BBox:
    x1: float  # Canto superior esquerdo (X)
    y1: float  # Canto superior esquerdo (Y)
    x2: float  # Canto inferior direito (X)
    y2: float  # Canto inferior direito (Y)
    
    # Métodos
    area() -> float        # Calcula a área
    iou(other: BBox) -> float  # Calcula IoU com outra bbox
```

### DetectedObject
```python
@dataclass
class DetectedObject:
    label: str
    score: float
    bbox: BBox
    source: str  # "qwen", "yoloe", ou "fused"
```

## 🧮 Algoritmo de Fusão

O sistema combina detecções usando **IoU (Intersection over Union)**:

1. **Para cada objeto do Qwen**:
   - Encontra o objeto do YOLOE com maior IoU
   - Se IoU ≥ threshold (0.15):
     - Usa label + score do Qwen
     - Usa bbox do YOLOE
     - Marca como `source="fused"`
   - Senão:
     - Mantém objeto do Qwen com `source="qwen"`

2. **Objetos do YOLOE** não emparelhados:
   - Marcados como `source="yoloe"`

## 🎯 Casos de Uso

### 1. Aplicações Infantis
Detecção de brinquedos e objetos seguros para crianças:
```bash
curl -X POST http://localhost:8001/detection/sys_prompt \
  -H "Content-Type: application/json" \
  -d '{
    "image": "YOUR_BASE64_IMAGE",
    "prompt": "Identifique apenas brinquedos nesta imagem. Brinquedos são objetos seguros para crianças."
  }'
```

### 2. Detecção Geral Robusta
Para máxima precisão e recall:
```bash
curl -X POST http://localhost:8001/detection/fused \
  -H "Content-Type: application/json" \
  -d '{"image": "YOUR_BASE64_IMAGE"}'
```

### 3. Análise Semântica
Quando apenas a semântica é importante:
```bash
curl -X POST http://localhost:8001/detection \
  -H "Content-Type: application/json" \
  -d '{"image": "YOUR_BASE64_IMAGE"}'
```

## 🐛 Troubleshooting

### vLLM demora muito para iniciar
- Normal para primeira inicialização (carrega modelo)
- Paciência: pode levar 2-5 minutos
- Verifique com: `curl http://localhost:8000/health`

### Erro de memória GPU
- Reduza `--gpu-memory-utilization` em docker-compose.yml
- Use modelo menor ou quantização

### Detecções incorretas
- Ajuste `IOU_THRESHOLD` em `services/fusion.py`
- Customize o `SYSTEM_INSTRUCTION` em `system_instruction.py`
- Use `/detection/sys_prompt` com prompt específico

## 📝 Logs e Debugging

Todos os logs são exibidos no stdout:

```bash
# Ver logs em tempo real
docker-compose logs -f api
docker-compose logs -f vllm
```

## 🔒 Segurança

- ✅ Validação de entrada com Pydantic
- ✅ Rate limiting (recomendado em produção)
- ✅ CORS (configurável em main.py)
- ⚠️ Sem autenticação por padrão (adicione em produção)

## 📈 Performance

**Tempo médio de processamento**:
- Qwen apenas: ~800-1200ms
- YOLOE apenas: ~100-200ms
- Fusão completa: ~1200-1500ms

**Memória**:
- vLLM (Qwen): ~8GB GPU
- API FastAPI: ~500MB RAM

## 🤝 Contribuições

Para melhorias, considere:
1. Otimizar threshold de IoU
2. Adicionar suporte para batch processing
3. Caching de detecções
4. Métricas e monitoring

## 📄 Licença

Este projeto utiliza modelos e ferramentas de código aberto.

---

**Desenvolvido para detecção robusta de objetos com semântica e precisão espacial.**
