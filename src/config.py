import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads")
INDEX_DIR = BASE_DIR / os.getenv("INDEX_DIR", "data/index")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1")
EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-mistral-7b-v2")
RERANKER_MODEL = os.getenv("NVIDIA_RERANKER_MODEL", "nvidia/llama-3.2-nv-rerankqa-1b-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
TOP_P = float(os.getenv("TOP_P", "0.1"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)