import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


_SECRET_SECTION_KEYS = {
    "retrieval": {
        "chunk_size": "CHUNK_SIZE",
        "chunk_overlap": "CHUNK_OVERLAP",
        "top_k": "TOP_K",
        "similarity_threshold": "SIMILARITY_THRESHOLD",
    },
    "generation": {"temperature": "TEMPERATURE", "top_p": "TOP_P"},
    "llm": {"provider": "LLM_PROVIDER"},
    "embedding": {"model": "EMBEDDING_MODEL"},
    "directories": {"seed_dir": "SEED_DIR", "upload_dir": "UPLOAD_DIR", "index_dir": "INDEX_DIR"},
}
_PROVIDER_SECTIONS = ("openrouter", "nvidia", "openai", "ollama")


def _secret_env_pairs(secrets) -> list[tuple[str, str]]:
    pairs = []
    for key in secrets:
        value = secrets[key]
        if isinstance(value, (str, int, float)):
            pairs.append((key, str(value)))
            continue
        if key in _SECRET_SECTION_KEYS:
            mapping = _SECRET_SECTION_KEYS[key]
        elif key in _PROVIDER_SECTIONS:
            mapping = None
        else:
            continue
        for subkey in value:
            subvalue = value[subkey]
            if not isinstance(subvalue, (str, int, float)):
                continue
            env_name = mapping.get(subkey) if mapping else f"{key}_{subkey}".upper()
            if env_name:
                pairs.append((env_name, str(subvalue)))
    return pairs


def _apply_secrets(secrets) -> None:
    for name, value in _secret_env_pairs(secrets):
        if value != "" and name not in os.environ:
            os.environ[name] = value


def _load_streamlit_secrets() -> None:
    # No Streamlit Community Cloud, as configurações ficam em Secrets (não há .env).
    try:
        import streamlit as st

        _apply_secrets(st.secrets)
    except Exception:
        pass


_load_streamlit_secrets()

BASE_DIR = Path(__file__).resolve().parent.parent
SEED_DIR = BASE_DIR / os.getenv("SEED_DIR", "data/seed")
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads")
INDEX_DIR = BASE_DIR / os.getenv("INDEX_DIR", "data/index")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

QDRANT_URL = os.getenv("QDRANT_URL", "").strip() or None
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.40"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
TOP_P = float(os.getenv("TOP_P", "0.1"))

for directory in (SEED_DIR, UPLOAD_DIR, INDEX_DIR):
    directory.mkdir(parents=True, exist_ok=True)
