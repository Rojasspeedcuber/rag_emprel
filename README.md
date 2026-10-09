# rag_emprel

Chat RAG (Retrieval-Augmented Generation) em Streamlit para perguntar sobre PDFs, com **não alucinação**: só responde com base nos trechos recuperados dos documentos, cita as fontes e diz "Não encontrei essa informação nos documentos enviados." quando não há base. As respostas são geradas pelo provedor escolhido no `.env` (OpenRouter, NVIDIA NIM, OpenAI ou Ollama).

## Guardrails anti-alucinação

1. **Contexto fechado** — o LLM recebe apenas os trechos recuperados dos seus PDFs
2. **Prompt estrito** — proíbe uso de conhecimento externo, trata instruções nos documentos como dados e exige citação `[arquivo, p. N]`
3. **Recusa determinística** — pergunta sem evidência suficiente acima do limiar retorna a mensagem fixa **antes** de qualquer chamada ao LLM
4. **Erros de provedor não viram resposta** — falha de API gera mensagem de indisponibilidade, nunca conteúdo improvisado
5. **Decodificação determinística** — `temperature=0`, `top_p=0.1`
6. **Transparência** — cada resposta exibe os trechos-fonte (arquivo e página) para auditoria

## Pré-requisitos

- Python 3.10+
- Chave de API do provedor escolhido (`OPENROUTER_API_KEY`, `NVIDIA_API_KEY` ou `OPENAI_API_KEY`). Apenas a chave do provedor selecionado em `LLM_PROVIDER` é necessária para gerar respostas; indexação e recusa funcionam sem nenhuma chave. Se `LLM_PROVIDER=ollama`, não é necessária chave, mas o servidor Ollama deve estar em execução (`ollama serve`).

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

Edite `.env`, defina `LLM_PROVIDER` (`openrouter`, `nvidia`, `openai` ou `ollama`) e preencha a chave correspondente. Para Ollama, basta ter o servidor local rodando (padrão `http://localhost:11434/v1`) e escolher um modelo com `OLLAMA_MODEL` (ex.: `qwen3:4b`). Também é possível gerar **embeddings** com Ollama definindo `EMBEDDING_MODEL=ollama:<modelo>` (ex.: `ollama:nomic-embed-text`); nesse caso o índice é reconstruído automaticamente.

## Execução

```powershell
streamlit run app.py
```

## Como usar

1. **Documentos-base**: o PDF de NFS-e em `data/seed/` já faz parte do acervo e não pode ser removido pela interface.
2. **Enviar documentos**: na barra lateral, clique em "Enviar PDFs" e selecione um ou mais arquivos. Eles são salvos em `data/uploads/` e indexados automaticamente.
3. **Perguntar**: digite a pergunta no chat. A resposta cita a fonte `[arquivo, p. N]` e mostra os trechos consultados no painel "Fontes consultadas".
4. **Remover documentos**: clique em 🗑 ao lado do arquivo enviado na barra lateral e o índice é reconstruído.
5. **Reindexar**: o botão 🔄 força a reconstrução do índice. Trocar `LLM_PROVIDER` e reiniciar roteia a geração para o novo provedor.

## Estrutura

```
rag_emprel/
├── app.py              # UI Streamlit (camada fina)
├── src/
│   ├── config.py       # Configurações (.env)
│   ├── embeddings.py   # Embeddings locais multilingues (MiniLM)
│   ├── ingest.py       # Seed + uploads, extração, chunking, índice FAISS com cache
│   ├── retriever.py    # Busca vetorial + limiar (recusa determinística)
│   ├── llm.py          # Geração via OpenRouter / NVIDIA NIM / OpenAI / Ollama
│   ├── prompts.py      # Prompt anti-alucinação
│   └── service.py      # Orquestração: recusa antes do LLM, fontes da resposta
├── data/
│   ├── seed/           # Documentos-base iniciais (versionados, não removíveis)
│   ├── uploads/        # PDFs enviados pelo usuário (removíveis)
│   └── index/          # Índice FAISS + cache (gerado automaticamente)
├── tests/
├── requirements.txt
└── .env.example
```

## Testes

```powershell
python -m pytest
```

Os testes não fazem chamadas de rede nem baixam modelos.

## Observações

- Os embeddings são gerados **localmente** com `paraphrase-multilingual-MiniLM-L12-v2` (configurável via `EMBEDDING_MODEL`). A primeira execução baixa o modelo; as demais usam o cache local do HuggingFace. Alternativa: defina `EMBEDDING_MODEL=ollama:<modelo>` para usar a API local do Ollama.
- O `SIMILARITY_THRESHOLD` padrão (0.45) foi calibrado empiricamente para esse modelo: perguntas fora do acervo ficam abaixo e perguntas suportadas ficam acima. Ao trocar o modelo de embedding, recalibre (modelos como E5 comprimem as similaridades e exigem outro valor).
- O índice FAISS é cacheado em `data/index/` e reconstruído quando muda o hash dos PDFs (seed + uploads), dos parâmetros de chunking ou do modelo de embedding.
- PDFs escaneados (sem camada de texto) não geram trechos; um PDF inválido é reportado individualmente sem invalidar os demais.
- Perguntas sem evidência suficiente retornam exatamente `Não encontrei essa informação nos documentos enviados.` sem acionar nenhum provedor de LLM.
