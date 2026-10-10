# ndo-ai-llama

A small Retrieval-Augmented Generation (RAG) chat API. It answers questions about a local
document corpus using a locally-running [Ollama](https://ollama.com) `llama3.2` model for
generation and a HuggingFace sentence-transformer for embeddings, indexed with
[Chroma](https://www.trychroma.com/) via [LlamaIndex](https://www.llamaindex.ai/).

## Architecture

```
Flask API (app/routes.py)
      │
      ▼
RagEngine (app/rag.py)
      │
      ├─ Ollama LLM            (generation, talks to a local Ollama server)
      ├─ HuggingFace embedding  (BAAI/bge-small-en-v1.5, downloaded on first run)
      └─ Chroma vector store    (persisted on disk, built once from ./docs)
```

- `app/` - the Flask application: factory (`__init__.py`), config, the RAG engine, and routes.
- `docs/` - the source document corpus that gets indexed and queried.
- `scripts/` - standalone dev tools that aren't part of the served API:
  - `chat_cli.py` - a terminal REPL for chatting with the RAG engine directly.
  - `viewer.py` - a Streamlit app for inspecting a persisted Chroma DB's collections.
- `tests/` - unit tests for every module above.
- `run.py` - the entrypoint that starts the Flask server.

The project intentionally lives in one repo with one layout, rather than scattered
top-level folders for the API, the doc corpus, and the Chroma viewer tool.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally, with the model pulled:
  ```
  ollama pull llama3.2
  ```

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
copy .env.example .env        # Windows; `cp` on macOS/Linux - then edit as needed
```

## Running the API

```
python run.py
```

On first run (or whenever `INIT_INDEX=true`), it reads every file under `DOCS_DIR`
(default `./docs`), embeds it, and persists the index to `INDEX_PERSIST_DIRECTORY`
(default `./data/chromadb`, gitignored). Subsequent runs load the persisted index
instead of rebuilding it.

Send a question:

```
curl -X POST http://localhost:7654/api/question \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is open5gs?\", \"user_id\": \"demo\"}"
```

Response:

```json
{ "answer": "As per my understanding, ..." }
```

`user_id` is required but currently only used for logging; it's reserved for future
per-user conversation history (see **Reserved configuration** below).

## Dev tools

```
pip install -r requirements-dev.txt -r requirements-viewer.txt

python -m scripts.chat_cli                              # terminal REPL
streamlit run scripts/viewer.py -- ./data/chromadb       # inspect the Chroma DB
```

## Configuration

All settings are environment variables, loaded from `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `DOCS_DIR` | `./docs` | Source document corpus to index |
| `INDEX_PERSIST_DIRECTORY` | `./data/chromadb` | Where the Chroma index is persisted |
| `INIT_INDEX` | `false` | Force a full reindex of `DOCS_DIR` on startup |
| `OLLAMA_MODEL` | `llama3.2` | Model name passed to Ollama |
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `HTTP_PORT` | `7654` | Flask server port |
| `FLASK_DEBUG` | `false` | Flask debug mode (never enable in production - the debugger allows remote code execution) |
| `DISABLE_TLS_VERIFICATION` | `false` | Disable TLS cert verification process-wide (only for local dev behind a self-signed proxy) |

### Reserved configuration

`TARGET_URL`, `MONGO_HOST`, `MONGO_PORT`, `MONGO_USER`, `MONGO_PASS` are accepted but not
yet used by the application. They're placeholders for a planned document scraper and
per-user chat history store, not wired up in this refactor.

## Testing

```
pip install -r requirements-dev.txt -r requirements-viewer.txt
pytest
```

Tests mock out Ollama, the HuggingFace embedding model, and Chroma - no local Ollama
server, network access, or model download is required to run the suite. Route tests
inject a fake `RagEngine`, so they don't touch llama_index/chromadb at all;
`RagEngine` tests stub its own `_build_index()` seam rather than mocking deep into
llama_index/chromadb internals.

## Known limitations

- `chat()` currently builds one shared `RagEngine` per process; it is not safe for
  concurrent reindexing while serving requests.
- `user_id` is accepted but has no persisted effect yet (see **Reserved configuration**).
