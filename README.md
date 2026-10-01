# rag_emprel

Chat RAG (Retrieval-Augmented Generation) em Streamlit para perguntar sobre PDFs, com LLM da NVIDIA (NIM) configurado para **não alucinar**: só responde com base nos trechos recuperados dos documentos, cita as fontes e diz "Não encontrei essa informação nos documentos" quando não há base.

## Guardrails anti-alucinação

1. **Contexto fechado** — o LLM recebe apenas os trechos recuperados dos seus PDFs
2. **Prompt estrito** — proíbe uso de conhecimento externo e exige citação de fonte
3. **Limiar de similaridade** — pergunta sem base no acervo nem chega ao LLM
4. **Decodificação determinística** — `temperature=0`, `top_p=0.1`
5. **Transparência** — cada resposta exibe os trechos-fonte (arquivo e página) para auditoria

## Pré-requisitos

- Python 3.10+
- Chave gratuita da API NVIDIA NIM: crie em [build.nvidia.com](https://build.nvidia.com)

## Instalação

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Crie um arquivo `.env` a partir do modelo:

```powershell
copy .env.example .env
```

Edite `.env` e preencha `NVIDIA_API_KEY` com sua chave.

## Execução

```powershell
streamlit run app.py
```

## Como usar

1. **Enviar documentos**: na barra lateral, clique em "Enviar PDFs" e selecione um ou mais arquivos. Eles são salvos em `data/uploads/` e indexados automaticamente.
2. **Perguntar**: digite a pergunta no chat. A resposta cita a fonte `[arquivo, p. N]` e mostra os trechos consultados no painel "Fontes consultadas".
3. **Remover documentos**: clique em 🗑 ao lado do arquivo na barra lateral e o índice é reconstruído.
4. **Reindexar**: o botão 🔄 força a reconstrução do índice (útil se algo parecer desatualizado).

## Estrutura

```
rag_emprel/
├── app.py              # UI Streamlit
├── src/
│   ├── config.py       # Configurações (.env)
│   ├── ingest.py       # Upload, extração, chunking, embeddings, índice FAISS com cache
│   ├── retriever.py    # Busca vetorial + reranker + limiar (fallback honesto)
│   ├── llm.py          # Cliente NVIDIA NIM (OpenAI-compatible)
│   └── prompts.py      # Prompt anti-alucinação
├── data/
│   ├── uploads/        # PDFs enviados (persistidos)
│   └── index/          # Índice FAISS + cache (gerado automaticamente)
├── tests/
├── requirements.txt
└── .env.example
```

## Testes

```powershell
pytest
```

## Observações

- O índice FAISS é cacheado em `data/index/` e só é reconstruído quando o conjunto de PDFs muda (hash do conteúdo).
- PDFs escaneados (sem camada de texto) não geram trechos — apenas páginas com texto extraível são indexadas.
- Modelos podem ser trocados no `.env` (`NVIDIA_LLM_MODEL`, `NVIDIA_EMBEDDING_MODEL`, `NVIDIA_RERANKER_MODEL`).
