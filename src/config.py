import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
SEED_DIR = BASE_DIR / os.getenv("SEED_DIR", "data/seed")
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads")
INDEX_DIR = BASE_DIR / os.getenv("INDEX_DIR", "data/index")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
TOP_P = float(os.getenv("TOP_P", "0.1"))

for directory in (SEED_DIR, UPLOAD_DIR, INDEX_DIR):
    directory.mkdir(parents=True, exist_ok=True)
