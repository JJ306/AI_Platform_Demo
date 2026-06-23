"""Day 2 — a Gradio frontend for RAG (model selection + indexed-doc visibility).

Users upload documents (txt / md / pdf), pick an embedding model and an answering
model, see which documents are indexed, and chat with them. Uploads go into Qdrant
(persists across restarts); re-uploading the same file doesn't create duplicates.

The UI talks to the FastAPI backend (/ingest, /ask, /documents, /rag/models).
Run with:  python -m miniplat.ui
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import httpx

BACKEND = os.environ.get("MINIPLAT_BACKEND_URL", "http://localhost:8000")


def _models() -> tuple[list[str], list[str]]:
    try:
        d = httpx.get(f"{BACKEND}/rag/models", timeout=10).json()
        return d["embed_models"], d["chat_models"]
    except Exception:  # noqa: BLE001 - backend may not be up yet
        return ["nomic-embed-text"], ["llama3.2:3b"]


def _read_file(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return p.read_text(errors="ignore")


def ingest_files(files, embed_model: str) -> str:
    if not files:
        return "No files selected."
    documents = [{"title": os.path.basename(f.name), "text": _read_file(f.name)} for f in files]
    resp = httpx.post(f"{BACKEND}/ingest",
                      json={"documents": documents, "embed_model": embed_model}, timeout=600)
    resp.raise_for_status()
    d = resp.json()
    return (f"✅ Indexed {d['chunks_indexed']} chunk(s) into '{d['collection']}' "
            f"({d['embed_model']}). Total in this index: {d['total_in_collection']} "
            f"(duplicates are skipped).")


def list_docs(embed_model: str) -> list[list]:
    """Return [[title, chunks], ...] for the currently-indexed documents."""
    try:
        r = httpx.get(f"{BACKEND}/documents", params={"embed_model": embed_model}, timeout=60)
        d = r.json()
        return [[doc["title"], doc["chunks"]] for doc in d["documents"]]
    except Exception:  # noqa: BLE001
        return []


def ask(message: str, history, embed_model: str, chat_model: str) -> str:
    resp = httpx.post(f"{BACKEND}/ask",
                      json={"query": message, "embed_model": embed_model, "chat_model": chat_model},
                      timeout=600)
    resp.raise_for_status()
    d = resp.json()
    answer = d["answer"]
    if d.get("sources"):
        answer += "\n\n*Sources: " + ", ".join(d["sources"]) + "*"
    return answer


def build_ui() -> gr.Blocks:
    embed_choices, chat_choices = _models()
    with gr.Blocks(title="miniplat — Chat with your documents") as demo:
        gr.Markdown("# 📚 miniplat RAG\nUpload documents, pick your models, and ask questions.")
        embed_model = gr.Dropdown(embed_choices, value=embed_choices[0], label="Embedding model")
        chat_model = gr.Dropdown(chat_choices, value=chat_choices[0], label="Answering model")
        with gr.Row():
            files = gr.File(label="Upload documents (txt / md / pdf)", file_count="multiple")
            status = gr.Textbox(label="Index status", interactive=False)
        docs_table = gr.Dataframe(headers=["Document", "Chunks"], label="📑 Indexed documents",
                                  interactive=False)
        with gr.Row():
            add_btn = gr.Button("Add to index")
            refresh_btn = gr.Button("Refresh indexed list")
        add_btn.click(ingest_files, inputs=[files, embed_model], outputs=status) \
               .then(list_docs, inputs=embed_model, outputs=docs_table)
        refresh_btn.click(list_docs, inputs=embed_model, outputs=docs_table)
        embed_model.change(list_docs, inputs=embed_model, outputs=docs_table)
        gr.ChatInterface(ask, additional_inputs=[embed_model, chat_model],
                         title="Ask your documents")
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
