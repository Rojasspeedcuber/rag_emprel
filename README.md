# RAG Emprel

Aplicação em Streamlit para consultar PDFs por meio de recuperação de trechos e geração de respostas (RAG). O acervo inicial contém um PDF de perguntas e respostas sobre NFS-e em `data/seed/`; outros PDFs podem ser enviados pela interface. A busca encontra passagens com arquivo e página, e o provedor de linguagem configurado redige a resposta com base nelas.

## Como funciona

1. `src/ingest.py` lê os PDFs de `data/seed/` e `data/uploads/` com `pypdf`, divide o texto extraído em trechos por página e guarda um índice FAISS em `data/index/`.
2. `src/embeddings.py` gera vetores normalizados com o modelo local `paraphrase-multilingual-MiniLM-L12-v2` ou, se configurado, com a API de embeddings do Ollama.
3. `src/retriever.py` procura até `TOP_K` trechos. Para consultas que contêm códigos no formato reconhecido pela aplicação (por exemplo, `E3351`), tenta primeiro a correspondência textual exata; nos demais casos, usa similaridade vetorial e recusa a pergunta se a melhor pontuação ficar abaixo de `SIMILARITY_THRESHOLD`.
4. `src/service.py` devolve a recusa sem chamar o modelo de linguagem quando a busca não encontra evidência. Caso haja trechos, `src/llm.py` envia a pergunta, as passagens e até quatro mensagens anteriores ao provedor escolhido. `src/prompts.py` instrui o modelo a usar apenas essas passagens, ignorar instruções contidas nos documentos e citar `[arquivo, p. N]`.
5. `app.py` apresenta o chat, o gerenciamento de PDFs e os trechos consultados em “Fontes consultadas”.

A recusa por ausência de trechos ou pontuação suficiente é implementada no código; já a fidelidade do texto gerado e das citações depende do modelo. O aplicativo não faz uma verificação automática de cada afirmação ou citação da resposta. O limiar é aplicado à melhor pontuação, não individualmente a todos os trechos enviados ao modelo.

<<<<<<< HEAD
- Python 3.10+
- Chave de API do provedor escolhido (`OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `OPENAI_API_KEY` ou `OLLAMA_API_KEY`). Apenas a chave do provedor selecionado em `LLM_PROVIDER` é necessária para gerar respostas; indexação e recusa funcionam sem nenhuma chave. Para `LLM_PROVIDER=ollama`, a chave é obtida em https://ollama.com/settings/keys.
=======
## Preparação
>>>>>>> 0303a3008955df9b2bdca08f433d4a55a0509d91

É necessário ter Python, acesso para instalar as dependências de `requirements.txt` e, para respostas geradas, um provedor configurado. O modelo de embeddings padrão é baixado na primeira indexação; essa opção exige acesso à rede nesse momento. Com Ollama para respostas ou embeddings, o servidor local precisa estar em execução e os modelos escolhidos precisam estar disponíveis nele. Uma chave de API é exigida apenas para gerar respostas com OpenRouter, NVIDIA NIM ou OpenAI.

Na raiz do repositório, em um terminal Unix:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

No PowerShell, use `py -m venv .venv`, `.venv\Scripts\Activate.ps1`, `python -m pip install -r requirements.txt` e `Copy-Item .env.example .env`. Ative o ambiente antes dos comandos abaixo. O arquivo `.env` é ignorado pelo Git: não inclua chaves no repositório.

O `.env.example` seleciona `LLM_PROVIDER=ollama`. Para essa opção, inicie o servidor Ollama e disponibilize o modelo indicado em `OLLAMA_MODEL` (no exemplo, `qwen3:4b`). Para outro provedor, ajuste `LLM_PROVIDER` para `openrouter`, `nvidia` ou `openai` e preencha somente a chave correspondente (`OPENROUTER_API_KEY`, `NVIDIA_API_KEY` ou `OPENAI_API_KEY`). A configuração é lida na inicialização; reinicie a aplicação após alterá-la. O código também permite ajustar a URL e o modelo de cada provedor no `.env`.

<<<<<<< HEAD
Edite `.env`, defina `LLM_PROVIDER` (`openrouter`, `nvidia`, `openai` ou `ollama`) e preencha a chave correspondente. Para Ollama Cloud, defina `OLLAMA_API_KEY` (o padrão de `OLLAMA_BASE_URL` é `https://ollama.com/v1`) e escolha um modelo de nuvem com `OLLAMA_MODEL` (ex.: `gpt-oss:120b`; lista em https://ollama.com/api/tags). Também é possível gerar **embeddings** com Ollama definindo `EMBEDDING_MODEL=ollama:<modelo>` (ex.: `ollama:nomic-embed-text`); nesse caso o índice é reconstruído automaticamente.

## Deploy no Streamlit Community Cloud

1. Suba o projeto para um repositório no GitHub (o `.env` não é versionado).
2. Em https://share.streamlit.io, crie o app apontando para `app.py`.
3. Em **Advanced settings → Secrets**, informe as configurações no formato TOML, por exemplo:

   ```toml
   LLM_PROVIDER = "ollama"
   OLLAMA_API_KEY = "sua-chave-do-ollama"
   OLLAMA_MODEL = "gpt-oss:120b"
   ```

   As chaves são lidas automaticamente de `st.secrets` (ou do `.env` localmente).

Observações sobre a nuvem: o sistema de arquivos é temporário, então PDFs enviados pela interface e o índice são perdidos quando o app reinicia (o índice é reconstruído a partir de `data/seed/`).
=======
| Variável | Uso | Valor do exemplo |
| --- | --- | --- |
| `EMBEDDING_MODEL` | Modelo local ou `ollama:<modelo>` para embeddings via Ollama | `paraphrase-multilingual-MiniLM-L12-v2` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Tamanho e sobreposição aproximados em tokens (o código usa quatro caracteres por token) | `800` / `150` |
| `TOP_K` / `SIMILARITY_THRESHOLD` | Quantidade máxima de trechos e limiar da busca vetorial | `5` / `0.45` |
| `TEMPERATURE` / `TOP_P` | Parâmetros enviados à API de geração | `0` / `0.1` |
| `SEED_DIR` / `UPLOAD_DIR` / `INDEX_DIR` | Diretórios do acervo e do índice, relativos à raiz do projeto quando não forem caminhos absolutos | `data/seed` / `data/uploads` / `data/index` |
>>>>>>> 0303a3008955df9b2bdca08f433d4a55a0509d91

Se optar por embeddings no Ollama, defina, por exemplo, `EMBEDDING_MODEL=ollama:nomic-embed-text` e disponibilize esse modelo no servidor configurado por `OLLAMA_BASE_URL` (padrão `http://localhost:11434/v1`). A troca do modelo de embeddings altera o hash do acervo e provoca a reconstrução do índice; avalie novamente o limiar de similaridade, pois as pontuações dependem do modelo.

## Executar e usar

```sh
streamlit run app.py
```

Na barra lateral, o PDF em “Documentos-base” já integra o acervo e não tem controle de exclusão. Para adicionar arquivos, selecione PDFs em “Enviar PDFs” e clique em **Adicionar ao acervo**; a seleção sozinha não os salva. Os arquivos são gravados em `data/uploads/` e o índice é reconstruído. Pergunte no chat e abra “Fontes consultadas” para examinar os trechos usados. O botão 🗑 remove um PDF enviado e reconstrói o índice; **🔄 Reindexar** força uma reconstrução.

O índice em `data/index/` é reutilizado enquanto não mudarem o conteúdo ou nome dos PDFs, o modelo de embeddings e os parâmetros de divisão dos trechos. As pastas de uploads e índice, assim como `.env`, são ignoradas pelo Git. Falhas de leitura de um PDF são mostradas separadamente, sem impedir a indexação dos demais.

## Testes

Com as dependências instaladas no ambiente virtual, execute na raiz:

```sh
python -m pytest
```

Os testes isolam o modelo de embeddings, a busca, a geração e a ingestão com substituições locais; não precisam de chaves de API, modelos baixados ou servidor Ollama. Eles não exercitam uma consulta ponta a ponta com um provedor real. Na execução em Linux, `test_sanitize_filename_strips_path_traversal` falha para um caminho com barras invertidas do Windows: `sanitize_filename` retorna `_.._windows_system32.pdf` em vez de `system32.pdf`. É uma limitação do código atual, não da instalação.

## Limitações

<<<<<<< HEAD
- Os embeddings são gerados **localmente** com `paraphrase-multilingual-MiniLM-L12-v2` (configurável via `EMBEDDING_MODEL`). A primeira execução baixa o modelo; as demais usam o cache local do HuggingFace. Alternativa: defina `EMBEDDING_MODEL=ollama:<modelo>` para usar a API do Ollama (com `OLLAMA_API_KEY`, o endpoint é o da nuvem).
- O `SIMILARITY_THRESHOLD` padrão (0.45) foi calibrado empiricamente para esse modelo: perguntas fora do acervo ficam abaixo e perguntas suportadas ficam acima. Ao trocar o modelo de embedding, recalibre (modelos como E5 comprimem as similaridades e exigem outro valor).
- O índice FAISS é cacheado em `data/index/` e reconstruído quando muda o hash dos PDFs (seed + uploads), dos parâmetros de chunking ou do modelo de embedding.
- PDFs escaneados (sem camada de texto) não geram trechos; um PDF inválido é reportado individualmente sem invalidar os demais.
- Perguntas sem evidência suficiente retornam exatamente `Não encontrei essa informação nos documentos enviados.` sem acionar nenhum provedor de LLM.
=======
- PDFs digitalizados sem texto extraível não produzem trechos. Não há OCR; se não houver texto indexável, o chat fica desabilitado.
- Uma consulta sem resultado suficiente retorna `Não encontrei essa informação nos documentos enviados.` antes de acionar o provedor. Quando há resultado, as citações são solicitadas pelo prompt, mas não validadas pelo código.
- Falhas de configuração ou de chamada ao provedor são exibidas como erro na conversa, sem gerar uma resposta substituta. A interface mantém o histórico na sessão do Streamlit; não há armazenamento persistente das conversas no projeto.
- `SIMILARITY_THRESHOLD=0.45` é um valor inicial do exemplo, não uma garantia de acerto. Ajuste-o conforme o acervo e o modelo de embeddings utilizados.
>>>>>>> 0303a3008955df9b2bdca08f433d4a55a0509d91
