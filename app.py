import streamlit as st

from src.config import NVIDIA_API_KEY
from src.ingest import save_upload, delete_upload, list_uploads, get_corpus
from src.retriever import retrieve
from src.llm import answer_question

st.set_page_config(page_title="RAG Emprel", page_icon="📄", layout="centered")

REFORMULATE_ENABLED = True  # histórico usado só para reformular a pergunta


@st.cache_resource(show_spinner=False)
def load_corpus():
    return get_corpus()


def rebuild_corpus():
    load_corpus.clear()
    return load_corpus()


st.title("📄 Chat RAG — Pergunte aos seus PDFs")

if not NVIDIA_API_KEY:
    st.error(
        "NVIDIA_API_KEY não configurada. Crie uma chave gratuita em "
        "[build.nvidia.com](https://build.nvidia.com) e coloque-a no arquivo `.env`."
    )
    st.stop()

# ---------- Sidebar: gerenciamento de documentos ----------
with st.sidebar:
    st.header("📚 Documentos")

    uploaded = st.file_uploader(
        "Enviar PDFs",
        type="pdf",
        accept_multiple_files=True,
    )
    if uploaded:
        with st.spinner("Salvando e indexando..."):
            for f in uploaded:
                save_upload(f.getvalue(), f.name)
            corpus = rebuild_corpus()
        st.success(f"{len(uploaded)} arquivo(s) enviado(s). Acervo reindexado.")
        st.rerun()

    names = list_uploads()
    if names:
        st.subheader(f"Acervo ({len(names)})")
        for name in names:
            col1, col2 = st.columns([5, 1])
            col1.write(name)
            if col2.button("🗑", key=f"del_{name}", help=f"Remover {name}"):
                delete_upload(name)
                rebuild_corpus()
                st.rerun()
    else:
        st.info("Nenhum documento enviado.")

    if st.button("🔄 Reindexar"):
        with st.spinner("Reindexando acervo..."):
            rebuild_corpus()
        st.success("Índice reconstruído.")
        st.rerun()

    corpus = load_corpus()
    if not corpus.empty:
        st.caption(f"{len(corpus.chunks)} trechos indexados.")

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Fontes consultadas"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** — p. {s['page']}")
                    st.caption(s["text"])

if corpus.empty:
    st.info("📤 Envie pelo menos um PDF pela barra lateral para começar a perguntar.")

question = st.chat_input("Faça uma pergunta sobre os documentos...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando nos documentos..."):
            retrieved, fallback = retrieve(corpus, question)

            if fallback is not None:
                answer = fallback
                sources = []
            else:
                hist = st.session_state.messages[:-1]
                answer = answer_question(question, retrieved, history=hist)
                sources = [
                    {"source": r.chunk.source, "page": r.chunk.page, "text": r.chunk.text}
                    for r in retrieved
                ]

        st.markdown(answer)
        if sources:
            with st.expander("Fontes consultadas"):
                for s in sources:
                    st.markdown(f"**{s['source']}** — p. {s['page']}")
                    st.caption(s["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
