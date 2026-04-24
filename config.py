"""
config.py — Central configuration. Change MODEL_NAME to swap models instantly.
"""
import os

# ══════════════════════════════════════════════════════════════
#  LLM  ← change model here (e.g. "llama3.2", "mistral", etc.)
# ══════════════════════════════════════════════════════════════
MODEL_NAME = os.getenv("OLLAMA_MODEL", "Qwen3.5-9B.Q4_K_M:latest")
MODEL_FALLBACKS = [
    model.strip()
    for model in os.getenv("OLLAMA_FALLBACK_MODELS", "").split(",")
    if model.strip()
]
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
OLLAMA_NUM_CTX     = int(os.getenv("OLLAMA_NUM_CTX",     "4096"))   # raised from 2048
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))   # raised from 512
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.15"))

# ══════════════════════════════════════════════════════════════
#  WEB SEARCH — SerpAPI
#  Key can be overridden by the SERPAPI_KEY env-var.
# ══════════════════════════════════════════════════════════════
SERPAPI_KEY = os.getenv(
    "SERPAPI_KEY",
    "526ad2a8cf33571f5078b456f4a7fe932e59fd57ef52006369b4b88ec676c309",
)

# ══════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR  = os.path.join(BASE_DIR, "agent_workspace")
DATA_DIR       = os.path.join(BASE_DIR, "data")
LT_MEMORY_FILE = os.path.join(DATA_DIR, "long_term_memory.json")
CHROMA_DIR     = os.path.join(DATA_DIR, "chroma_db")
TEMPLATES_DIR  = os.path.join(BASE_DIR, "templates")

# ══════════════════════════════════════════════════════════════
#  AGENT SETTINGS
# ══════════════════════════════════════════════════════════════
MAX_ITERATIONS     = 14
MAX_RETRIES        = 2
PERMISSION_TIMEOUT = 180  # seconds the user has to approve/deny a tool call

# ══════════════════════════════════════════════════════════════
#  AUTO-CREATE DIRECTORIES
# ══════════════════════════════════════════════════════════════
for _d in [WORKSPACE_DIR, DATA_DIR, CHROMA_DIR, TEMPLATES_DIR]:
    os.makedirs(_d, exist_ok=True)
