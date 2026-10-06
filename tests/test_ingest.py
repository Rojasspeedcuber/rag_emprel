from pathlib import Path

import numpy as np

from src import ingest
from src.ingest import sanitize_filename, chunk_text, Chunk, ARGS_STR_CHARS


def test_sanitize_filename_keeps_valid_name():
    assert sanitize_filename("documento.pdf") == "documento.pdf"


def test_sanitize_filename_strips_path_traversal():
    assert sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("..\\..\\windows\\system32.pdf") == "system32.pdf"


def test_sanitize_filename_replaces_dangerous_chars():
    name = sanitize_filename("meu/arquivo<teste>.pdf")
    assert "/" not in name and "<" not in name and ">" not in name
    assert name.endswith(".pdf")


def test_sanitize_filename_adds_extension():
    assert sanitize_filename("semextensao").endswith(".pdf")


def test_chunk_text_respects_max_size():
    text = "Palavra " * 2000  # texto longo
    chunks = chunk_text(text, source="doc.pdf", page=1)
    assert len(chunks) > 1
    max_chars = 800 * ARGS_STR_CHARS
    for c in chunks:
        assert len(c.text) <= max_chars + 100  # margem de fronteira de frase


def test_chunk_text_keeps_metadata():
    chunks = chunk_text("Texto curto.", source="x.pdf", page=3)
    assert chunks == [Chunk(text="Texto curto.", source="x.pdf", page=3)]


def test_chunk_text_empty():
    assert chunk_text("", source="x.pdf", page=1) == []
    assert chunk_text("   \n  ", source="x.pdf", page=1) == []


def test_chunk_text_overlap():
    text = ("Frase uno. " * 200).strip()
    chunks = chunk_text(text, source="d.pdf", page=1, )
    if len(chunks) > 1:
        tail = chunks[0].text[-60:]
        assert tail in chunks[1].text  # overlap presente


def configure_document_dirs(monkeypatch, tmp_path):
    seed = tmp_path / "seed"
    uploads = tmp_path / "uploads"
    index = tmp_path / "index"
    for directory in (seed, uploads, index):
        directory.mkdir()
    monkeypatch.setattr(ingest, "SEED_DIR", seed)
    monkeypatch.setattr(ingest, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(ingest, "INDEX_DIR", index)
    return seed, uploads, index


def test_list_documents_separates_seed_and_uploads(monkeypatch, tmp_path):
    seed, uploads, _ = configure_document_dirs(monkeypatch, tmp_path)
    (seed / "base.pdf").write_bytes(b"base")
    (uploads / "user.pdf").write_bytes(b"user")

    assert ingest.list_seed_documents() == ["base.pdf"]
    assert ingest.list_uploads() == ["user.pdf"]


def test_duplicate_upload_does_not_shadow_seed(monkeypatch, tmp_path):
    seed, _, _ = configure_document_dirs(monkeypatch, tmp_path)
    (seed / "base.pdf").write_bytes(b"base")

    saved = ingest.save_upload(b"upload", "base.pdf")

    assert saved.name == "base_1.pdf"


def test_build_corpus_combines_seed_and_uploads(monkeypatch, tmp_path):
    seed, uploads, _ = configure_document_dirs(monkeypatch, tmp_path)
    (seed / "base.pdf").write_bytes(b"base")
    (uploads / "user.pdf").write_bytes(b"user")
    monkeypatch.setattr(ingest, "extract_pages", lambda path: [(1, path.stem)])
    monkeypatch.setattr(
        ingest,
        "embed_documents",
        lambda texts: np.ones((len(texts), 2), dtype="float32"),
    )

    corpus = ingest.build_corpus()

    assert {chunk.source for chunk in corpus.chunks} == {"base.pdf", "user.pdf"}
    assert corpus.errors == []


def test_build_corpus_reports_one_bad_pdf_and_keeps_others(monkeypatch, tmp_path):
    seed, uploads, _ = configure_document_dirs(monkeypatch, tmp_path)
    (seed / "good.pdf").write_bytes(b"good")
    (uploads / "bad.pdf").write_bytes(b"bad")

    def extract(path: Path):
        if path.name == "bad.pdf":
            raise ValueError("invalid PDF")
        return [(1, "valid text")]

    monkeypatch.setattr(ingest, "extract_pages", extract)
    monkeypatch.setattr(
        ingest,
        "embed_documents",
        lambda texts: np.ones((len(texts), 2), dtype="float32"),
    )

    corpus = ingest.build_corpus()

    assert [chunk.source for chunk in corpus.chunks] == ["good.pdf"]
    assert corpus.errors == ["bad.pdf: invalid PDF"]
