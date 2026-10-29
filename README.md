# rag_emprel

Chat RAG (Retrieval-Augmented Generation) em Streamlit para perguntar sobre PDFs, com **não alucinação**: só responde com base nos trechos recuperados dos documentos, cita as fontes e diz "Não encontrei essa informação nos documentos enviados." para perguntas fora do tema. As respostas são geradas pelo provedor escolhido no `.env` (OpenRouter, NVIDIA NIM, OpenAI ou Ollama).

## Guardrails

1. **Contexto fechado** — o LLM recebe apenas os trechos recuperados dos seus PDFs
2. **Prompt estrito** — proíbe uso de conhecimento externo, trata instruções nos documentos como dados e exige citação `[arquivo, p. N]`
3. **Recusa determinística** — pergunta fora do tema (similaridade máxima abaixo de `SIMILARITY_THRESHOLD`, padrão 0.40) retorna a mensagem fixa **antes** de qualquer chamada ao LLM. Busca exata por código ou sigla presente no documento sempre passa
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

Os vetores ficam em um banco **Qdrant**. Por padrão ele roda embutido em `data/index/qdrant/`, sem servidor. Ao adicionar ou remover um PDF, só os trechos novos são embedados e os obsoletos são apagados do banco; o restante é reaproveitado. Os metadados (arquivo, página e texto) ficam no próprio banco, e `data/index/chunks.json` guarda o acervo usado pela busca exata por código. Para usar um servidor Qdrant (local via Docker ou Qdrant Cloud), defina `QDRANT_URL` e, se necessário, `QDRANT_API_KEY` no `.env`. As pastas de uploads e índice, assim como `.env`, são ignoradas pelo Git. Falhas de leitura de um PDF são mostradas separadamente, sem impedir a indexação dos demais.

### Acessibilidade por voz

- **Perguntar por voz**: o botão 🎤 **Falar pergunta**, acima da caixa de texto, capta a fala em português (pt-BR) e envia a pergunta ao chat.
- **Ouvir respostas**: cada resposta tem o botão 🔊 **Ouvir resposta** (e ⏹ **Parar leitura**). Para ler automaticamente cada nova resposta, ative **Ler respostas em voz alta** na barra lateral.

Os recursos usam a Web Speech API do navegador, sem chave de API: funcionam no Google Chrome e no Microsoft Edge (o reconhecimento de voz depende de conexão com a internet e o navegador precisa de permissão para o microfone). A leitura usa a voz em português instalada no sistema operacional.

### Deploy no Streamlit Community Cloud

O Qdrant embutido não deve ser usado no Streamlit Cloud: o app recarrega módulos quando o código é atualizado e o arquivo de índice fica travado pelo processo anterior, causando o erro “Storage folder ... is already accessed”. Use um servidor Qdrant:

1. Crie um cluster gratuito no [Qdrant Cloud](https://cloud.qdrant.io) e copie a URL do cluster e uma API key.
2. No painel do app, em **Settings → Secrets**, adicione:

   ```toml
   QDRANT_URL = "https://seu-cluster.cloud.qdrant.io:6333"
   QDRANT_API_KEY = "sua-api-key"
   ```

3. Salve e reinicie o app. O índice será reconstruído no novo servidor na primeira execução.

## Testes

Com as dependências instaladas no ambiente virtual, execute na raiz:

```sh
python -m pytest
```

Os testes não fazem chamadas de rede nem baixam modelos.

## Observações

- Os embeddings são gerados **localmente** com `paraphrase-multilingual-MiniLM-L12-v2` (configurável via `EMBEDDING_MODEL`). A primeira execução baixa o modelo; as demais usam o cache local do HuggingFace. Alternativa: defina `EMBEDDING_MODEL=ollama:<modelo>` para usar a API local do Ollama.
- O índice é revalidado a cada mudança no hash dos PDFs (seed + uploads), nos parâmetros de chunking ou no modelo de embedding. Trechos cujo ID (modelo + arquivo + página + texto) já existe no Qdrant não são re-embedados. Ao trocar o modelo de embedding, o banco é recriado automaticamente se a dimensão mudar.
- O arquivo `data/index/index.faiss` de versões anteriores não é mais usado e pode ser apagado.
- PDFs escaneados (sem camada de texto) não geram trechos; um PDF inválido é reportado individualmente sem invalidar os demais.
- O `SIMILARITY_THRESHOLD` padrão (0.40) foi calibrado com o MiniLM: perguntas fora do tema ficaram até 0.35 e perguntas do acervo a partir de 0.46 (exceto as que usam sigla exata, aceitas pela busca lexical). Ao trocar o modelo de embedding, recalibre.
- Perguntas sem relação com os trechos recuperados também podem receber a mesma mensagem fixa pelo próprio LLM, conforme o prompt.
