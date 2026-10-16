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

Na barra lateral, o PDF em “Documentos-base” já integra o acervo e não tem controle de exclusão. Para adicionar arquivos, selecione PDFs em “Enviar PDFs” e clique em **Adicionar ao acervo**; a seleção sozinha não os salva. Os arquivos são gravados em `data/uploads/` e o índice é reconstruído. Pergunte no chat e abra “Fontes consultadas” para examinar os trechos usados. O botão 🗑 remove um PDF enviado e reconstrói o índice; **🔄 Reindexar** força uma reconstrução.

O índice em `data/index/` é reutilizado enquanto não mudarem o conteúdo ou nome dos PDFs, o modelo de embeddings e os parâmetros de divisão dos trechos. As pastas de uploads e índice, assim como `.env`, são ignoradas pelo Git. Falhas de leitura de um PDF são mostradas separadamente, sem impedir a indexação dos demais.

## Testes

Com as dependências instaladas no ambiente virtual, execute na raiz:

```sh
python -m pytest
```

Os testes não fazem chamadas de rede nem baixam modelos.

## Observações

- Os embeddings são gerados **localmente** com `paraphrase-multilingual-MiniLM-L12-v2` (configurável via `EMBEDDING_MODEL`). A primeira execução baixa o modelo; as demais usam o cache local do HuggingFace. Alternativa: defina `EMBEDDING_MODEL=ollama:<modelo>` para usar a API local do Ollama.
- O `SIMILARITY_THRESHOLD` padrão (0.45) foi calibrado empiricamente para esse modelo: perguntas fora do acervo ficam abaixo e perguntas suportadas ficam acima. Ao trocar o modelo de embedding, recalibre (modelos como E5 comprimem as similaridades e exigem outro valor).
- O índice FAISS é cacheado em `data/index/` e reconstruído quando muda o hash dos PDFs (seed + uploads), dos parâmetros de chunking ou do modelo de embedding.
- PDFs escaneados (sem camada de texto) não geram trechos; um PDF inválido é reportado individualmente sem invalidar os demais.
- Perguntas sem evidência suficiente retornam exatamente `Não encontrei essa informação nos documentos enviados.` sem acionar nenhum provedor de LLM.
