import re
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from openai import OpenAI

from src.config import (
    UPLOAD_DIR,
    INDEX_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    EMBEDDING_MODEL,
)

ARGS_STR_CHARS = 4  # ~1 token ≈ 4 chars


@dataclass
class Chunk:
    text: str
    source: str
    page: int


@dataclass
class Corpus:
    chunks: list[Chunk] = field(default_factory=list)
    index: faiss.IndexFlatIP | None = None
    empty: bool = True


def sanitize_filename(name: str) -> str:
    """Sanitiza o nome do arquivo removendo caracteres perigosos e path traversal."""
    name = Path(name).name
    name = re.sub(r"[^\w.\-() ]", "_", name)
    name = re.sub(r"^\.+", "", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name or "upload.pdf"


def save_upload(file_bytes: bytes, filename: str) -> Path:
    """Salva um PDF enviado em UPLOAD_DIR. Retorna o caminho salvo."""
    safe = sanitize_filename(filename)
    target = UPLOAD_DIR / safe
    if target.exists():
        stem, suffix = target.stem, target.suffix
        i = 1
        while target.exists():
            target = UPLOAD_DIR / f"{stem}_{i}{suffix}"
            i += 1
    target.write_bytes(file_bytes)
    return target


def delete_upload(filename: str) -> bool:
    target = UPLOAD_DIR / sanitize_filename(filename)
    if target.exists():
        target.unlink()
        return True
    return False


def list_uploads() -> list[str]:
    return sorted(p.name for p in UPLOAD_DIR.glob("*.pdf"))


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Retorna [(numero_pagina, texto), ...]. Páginas sem texto vêm com ''."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text.strip()))
    return pages


def chunk_text(text: str, source: str, page: int) -> list[Chunk]:
    """Fatiamento deterministico em caracteres (~800 tokens de largura) com overlap."""
    max_chars = CHUNK_SIZE * ARGS_STR_CHARS
    overlap_chars = CHUNK_OVERLAP * ARGS_STR_CHARS
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # prefere quebrar em limite de parágrafo/frase se próximo do fim
        if end < len(text):
            boundary = text.rfind("\n", start + max_chars // 2, end)
            if boundary == -1:
                boundary = text.rfind(". ", start + max_chars // 2, end)
            if boundary != -1:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, page=page))
        start = end - overlap_chars if end < len(text) else end
    return chunks


def _files_hash() -> str:
    """Hash do conjunto de PDFs em UPLOAD_DIR (nome + conteúdo)."""
    h = hashlib.sha256()
    for p in sorted(UPLOAD_DIR.glob("*.pdf")):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def _get_client() -> OpenAI:
    return OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)


def _embed_texts(texts: list[str], input_type: str = "passage") -> np.ndarray:
    """Gera embeddings em lote via API NVIDIA. Retorna matriz (n, dim) L2-normalizada."""
    client = _get_client()
    all_vectors: list[list[float]] = []
    batch = 32
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(
            input=texts[i : i + batch],
            model=EMBEDDING_MODEL,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "END"},
        )
        all_vectors.extend(item.embedding for item in resp.data)
    mat = np.array(all_vectors, dtype="float32")
    faiss.normalize_L2(mat)
    return mat


def embed_query(query: str) -> np.ndarray:
    return _embed_texts([query], input_type="query")


def build_corpus() -> Corpus:
    """(Re)constrói o corpus a partir de UPLOAD_DIR. Lida com pasta vazia."""
    chunks: list[Chunk] = []
    for pdf_path in sorted(UPLOAD_DIR.glob("*.pdf")):
        for page_num, text in extract_pages(pdf_path):
            if text:
                chunks.extend(chunk_text(text, source=pdf_path.name, page=page_num))

    index = None
    if chunks:
        vectors = _embed_texts([c.text for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

    return Corpus(chunks=chunks, index=index, empty=not chunks)


def _save_corpus(corpus: Corpus) -> None:
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps([asdict(c) for c in corpus.chunks], ensure_ascii=False),
        encoding="utf-8",
    )
    (INDEX_DIR / "files.hash").write_text(_files_hash(), encoding="utf-8")
    if corpus.index is not None:
        faiss.write_index(corpus.index, str(INDEX_DIR / "index.faiss"))
    elif (INDEX_DIR / "index.faiss").exists():
        (INDEX_DIR / "index.faiss").unlink()


def _load_cached_corpus() -> Corpus | None:
    chunks_file = INDEX_DIR / "chunks.json"
    hash_file = INDEX_DIR / "files.hash"
    index_file = INDEX_DIR / "index.faiss"
    if not (chunks_file.exists() and hash_file.exists()):
        return None
    if hash_file.read_text(encoding="utf-8") != _files_hash():
        return None
    chunks = [Chunk(**c) for c in json.loads(chunks_file.read_text(encoding="utf-8"))]
    if chunks and index_file.exists():
        index = faiss.read_index(str(index_file))
        return Corpus(chunks=chunks, index=index, empty=False)
    return Corpus(chunks=[], index=None, empty=True)


def get_corpus(force_rebuild: bool = False) -> Corpus:
    """Retorna o corpus, usando cache quando o conjunto de arquivos não mudou."""
    if not force_rebuild:
        cached = _load_cached_corpus()
        if cached is not None:
            return cached
    corpus = build_corpus()
    _save_corpus(corpus)
    return corpus