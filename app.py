import streamlit as st

from src.config import LLM_PROVIDER
from src.ingest import (
    delete_upload,
    get_corpus,
    list_seed_documents,
    list_uploads,
    save_upload,
)
from src.llm import LLMConfigurationError, LLMProviderError
from src.service import answer

st.set_page_config(page_title="RAG Emprel", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner=False)
def load_corpus():
    return get_corpus()


def rebuild_corpus():
    load_corpus.clear()
    return get_corpus(force_rebuild=True)


st.title("📄 Chat RAG — Pergunte aos seus PDFs")

with st.sidebar:
    st.header("📚 Documentos")
    st.caption(f"Provedor de respostas: {LLM_PROVIDER}")

    uploaded = st.file_uploader(
        "Enviar PDFs",
        type="pdf",
        accept_multiple_files=True,
    )
    if st.button("Adicionar ao acervo", disabled=not uploaded):
        with st.spinner("Salvando e indexando..."):
            for file in uploaded:
                save_upload(file.getvalue(), file.name)
            rebuild_corpus()
        st.success(f"{len(uploaded)} arquivo(s) enviado(s).")
        st.rerun()

    seed_names = list_seed_documents()
    if seed_names:
        st.subheader("Documentos-base")
        for name in seed_names:
            st.write(name)

    upload_names = list_uploads()
    if upload_names:
        st.subheader("Seus documentos")
        for name in upload_names:
            label, action = st.columns([5, 1])
            label.write(name)
            if action.button("🗑", key=f"del_{name}", help=f"Remover {name}"):
                delete_upload(name)
                rebuild_corpus()
                st.rerun()

    if st.button("🔄 Reindexar"):
        with st.spinner("Reindexando acervo..."):
            rebuild_corpus()
        st.success("Índice reconstruído.")
        st.rerun()

    corpus = load_corpus()
    st.caption(f"{len(corpus.chunks)} trechos indexados.")
    for error in corpus.errors:
        st.error(f"Falha ao ler {error}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Fontes consultadas"):
                for source in message["sources"]:
                    st.markdown(f"**{source['source']}** — p. {source['page']}")
                    st.caption(source["text"])

if corpus.empty:
    st.info("Nenhum documento com texto extraível foi encontrado.")

question = st.chat_input(
    "Faça uma pergunta sobre os documentos...",
    disabled=corpus.empty,
)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando nos documentos..."):
            try:
                result = answer(
                    corpus,
                    question,
                    history=st.session_state.messages[:-1],
                )
            except (LLMConfigurationError, LLMProviderError) as exc:
                result_content = str(exc)
                sources = []
            else:
                result_content = result.content
                sources = result.sources

        st.markdown(result_content)
        if sources:
            with st.expander("Fontes consultadas"):
                for source in sources:
                    st.markdown(f"**{source['source']}** — p. {source['page']}")
                    st.caption(source["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result_content, "sources": sources}
    )
