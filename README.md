# 📚 miniplat — Chat With Your Documents, 100% Locally

> A tiny but complete **AI platform** you can run on your laptop. Upload your docs, pick an open-weight model, and ask questions grounded in your own content — no cloud account, no API keys, no data leaving your machine.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white">
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-local%20LLMs-000000?logo=ollama&logoColor=white">
  <img alt="Qdrant" src="https://img.shields.io/badge/Qdrant-vector%20DB-DC244C">
  <img alt="Gradio" src="https://img.shields.io/badge/Gradio-UI-FF7C00?logo=gradio&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## ✨ Why miniplat?

- 🔒 **Fully local & private** — runs on open-weight models via [Ollama](https://ollama.com). Your documents never leave your machine.
- 💸 **Free & open-source** — no cloud bill, no paid API, no signup.
- 🧠 **Real RAG, not a toy** — embeds your docs, stores vectors in [Qdrant](https://qdrant.tech), retrieves the most relevant chunks, and answers **with citations**.
- 🔁 **Persistent & deduplicated** — your index survives restarts, and re-uploading the same file won't create duplicates.
- 🎛️ **Multi-model** — swap embedding and answering models from a dropdown in the UI.
- 🐳 **One command to run** — `docker compose up` brings up the whole stack.

---

## 🏗️ How it works

```
                ┌──────────────┐
   You ───────▶ │  Gradio UI   │   upload docs · pick models · ask questions
                │  (port 7860) │
                └──────┬───────┘
                       │ HTTP
                ┌──────▼───────┐
                │  FastAPI API │   /ingest · /search · /ask · /documents
                │  (port 8000) │
                └───┬──────┬───┘
          embeddings│      │ vectors
           & chat   │      │
            ┌───────▼─┐  ┌─▼─────────┐
            │ Ollama  │  │  Qdrant   │   persistent vector store
            │ (11434) │  │  (6333)   │   (survives restarts)
            └─────────┘  └───────────┘
```

**The RAG flow:** Upload a document → it's split into chunks → each chunk is embedded by an open model → vectors are stored in Qdrant → your question is embedded and matched against them → the top chunks are fed to the LLM, which answers using *only* that context and cites its sources.

---

## 💻 Prerequisites by operating system

miniplat runs on **macOS, Linux, and Windows**. The easiest path on every OS is **Docker** (Option A) — it bundles Ollama and Qdrant for you, so the only thing you install on the host is Docker itself.

<details open>
<summary><b>🍎 macOS</b></summary>

```bash
# Install Homebrew first if you don't have it: https://brew.sh
brew install --cask docker     # then launch Docker Desktop once

# For the local (non-Docker) path you also want:
brew install uv ollama
```
Apple Silicon (M1/M2/M3) and Intel Macs are both supported.
</details>

<details>
<summary><b>🐧 Linux</b></summary>

```bash
# Docker Engine — see https://docs.docker.com/engine/install/ for your distro
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # log out/in so you can run docker without sudo

# For the local (non-Docker) path:
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv
curl -fsSL https://ollama.com/install.sh | sh     # ollama
```
> 🔥 Have an NVIDIA GPU? Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and Ollama will use it automatically for much faster inference.
</details>

<details>
<summary><b>🪟 Windows</b></summary>

The recommended setup is **WSL2 + Docker Desktop**:

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and enable the **WSL2 backend** (Settings → General → *Use the WSL 2 based engine*).
2. Run the commands from this README inside a **WSL2 Ubuntu** terminal (Start → "Ubuntu"), or in PowerShell.

For the local (non-Docker) path, in **PowerShell**:
```powershell
winget install astral-sh.uv      # uv
winget install Ollama.Ollama     # ollama (or download from https://ollama.com/download)
```
> 💡 Inside WSL2/PowerShell, `docker compose` and `curl` work the same as on macOS/Linux. Use `http://localhost:7860` and `:8000` in your Windows browser — Docker Desktop forwards the ports automatically.
</details>

> ℹ️ Once Docker is installed, the **Quick start** commands below are **identical on all three operating systems**.

---

## 🚀 Quick start

### Option A — Docker (recommended, one command)

Requires [Docker](https://www.docker.com/).

```bash
# 1. Start the whole stack (API + UI + Ollama + Qdrant)
docker compose up --build

# 2. In another terminal, pull the open models into the Ollama container (once)
docker compose exec ollama ollama pull llama3.2:3b        # chat / generation
docker compose exec ollama ollama pull nomic-embed-text   # embeddings
```

Then open:

| Service | URL |
|---|---|
| 💬 **Chat UI** | http://localhost:7860 |
| ⚡ **API docs (Swagger)** | http://localhost:8000/docs |

### Option B — Run locally with `uv`

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com). See [SetUP.MD](SetUP.MD) for full tooling install steps.

```bash
# Install dependencies
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Pull the models (once)
ollama serve &
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Start Qdrant (vector store)
docker run -d -p 6333:6333 -v qdrant:/qdrant/storage qdrant/qdrant

# Run the API
uvicorn miniplat.main:app --reload --port 8000

# Run the UI (in another terminal)
python -m miniplat.ui
```

---

## 🖱️ Using the app

1. Open the **Chat UI** at http://localhost:7860.
2. Pick an **embedding model** and an **answering model** from the dropdowns.
3. **Upload** one or more `.txt`, `.md`, or `.pdf` files and click **Add to index**.
4. Watch them appear in the **Indexed documents** table.
5. **Ask a question** in the chat — answers come back grounded in your docs, with sources.

---

## 🔌 API reference

The FastAPI backend is fully usable on its own. Interactive docs live at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/healthz` | Liveness check |
| `POST` | `/chat` | Plain chat with the LLM (no retrieval) |
| `POST` | `/ingest` | Embed and index documents |
| `POST` | `/search` | Return the most similar chunks for a query |
| `POST` | `/ask` | **RAG**: retrieve + answer with citations |
| `GET` | `/documents` | List currently indexed documents |
| `GET` | `/rag/models` | Available embedding / chat models |

**Example — ingest and ask:**

```bash
# Index a document
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"title": "notes.txt", "text": "miniplat is a local RAG platform."}]}'

# Ask a grounded question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is miniplat?"}'
```

```json
{
  "answer": "miniplat is a local RAG platform [1].",
  "sources": ["notes.txt"],
  "grounded": true,
  "embed_model": "nomic-embed-text",
  "chat_model": "llama3.2:3b"
}
```

---

## ⚙️ Configuration

Every setting is an environment variable prefixed with `MINIPLAT_` (also loadable from a `.env` file).

| Variable | Default | Description |
|---|---|---|
| `MINIPLAT_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `MINIPLAT_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `MINIPLAT_CHAT_MODEL` | `llama3.2:3b` | Default answering model |
| `MINIPLAT_EMBED_MODEL` | `nomic-embed-text` | Default embedding model |
| `MINIPLAT_CHAT_MODELS` | `llama3.2:3b` | Comma-separated chat models offered in the UI |
| `MINIPLAT_EMBED_MODELS` | `nomic-embed-text` | Comma-separated embedding models offered in the UI |
| `MINIPLAT_BACKEND_URL` | `http://localhost:8000` | Backend URL the UI talks to |

> 💡 Want more models? Pull them in Ollama (`ollama pull <model>`), then add them to `MINIPLAT_CHAT_MODELS` / `MINIPLAT_EMBED_MODELS`. Each embedding model gets its own Qdrant collection automatically.

---

## 🗂️ Project structure

```
miniplat/
├── main.py        # FastAPI app: /healthz, /chat, mounts the RAG router
├── rag.py         # RAG: chunking, embeddings, Qdrant store, /ingest /search /ask
├── llm.py         # Async Ollama client
├── config.py      # Env-based settings (MINIPLAT_*)
└── ui.py          # Gradio frontend
Dockerfile         # Containerizes the API/UI
docker-compose.yml # Full stack: app · ui · ollama · qdrant
```

---

## 🧪 Development

```bash
uv pip install -e ".[dev]"   # install dev tools (pytest, ruff)
pytest                       # run tests
ruff check .                 # lint
```

---

## 🛠️ Tech stack

| Layer | Tool |
|---|---|
| API | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| LLMs & embeddings | [Ollama](https://ollama.com) (open-weight models) |
| Vector store | [Qdrant](https://qdrant.tech) |
| UI | [Gradio](https://www.gradio.app/) |
| Packaging | [uv](https://docs.astral.sh/uv/) · [Docker](https://www.docker.com/) |

---

## 📄 License

Released under the [MIT License](LICENSE) — free to use, modify, and share.

---

<p align="center">Built with ❤️ as a hands-on, run-it-yourself AI platform. Star ⭐ the repo if it helped!</p>
```