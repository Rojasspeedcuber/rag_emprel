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
