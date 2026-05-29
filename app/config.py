"""
Configurações centralizadas da aplicação.

Variáveis de ambiente e constantes utilizadas pelos serviços e endpoints.
"""

import os

# ── vLLM ──────────────────────────────────────────────────────────────────────
VLLM_URL   = os.getenv("VLLM_URL",   "http://vllm:8000") + "/v1/chat/completions"
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")

# ── Filtros de foreground ──────────────────────────────────────────────────────
MIN_FOREGROUND_AREA = 0.015   # 1.5 % — abaixo é ruído/fundo
MAX_FOREGROUND_AREA = 0.75    # 75  % — acima é a cena inteira
SMALL_AREA          = 0.03    # 3   % — pequeno mas não microscópico
SMALL_CENTRALITY    = 0.25    # muito periférico (canto)

# ── Pipeline ──────────────────────────────────────────────────────────────────
CROP_PADDING        = 0.05    # padding relativo ao tamanho do bbox
IOI_THRESHOLD       = 0.15    # IoU mínimo para match Qwen ↔ YOLOE
SUBPART_CONTAINMENT = 0.60    # > 60 % de A dentro de B → A é sub-parte
VERIFY_SCORE_MIN    = 0.6     # score mínimo pós-verificação de crop
TOP_K_RESULTS       = 5       # número máximo de objetos no resultado final
