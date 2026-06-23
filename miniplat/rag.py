"""Day 2 — Retrieval-Augmented Generation (RAG) on Qdrant.

Ingest documents → embed with an open model → store vectors in **Qdrant** → retrieve
the most similar chunks → answer grounded in them, with sources.

Qdrant persists data to its own storage volume, so the index **accumulates** and is
**reusable across restarts** (Docker or Kubernetes).

Multi-model: the user picks the **embedding model** (each gets its own Qdrant
collection) and the **answering model**.

Dedup: each chunk's point id is a deterministic hash of (title + text), so
re-ingesting the same document **overwrites** instead of creating duplicates.

Visibility: ``GET /documents`` lists which documents are currently indexed.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from miniplat.config import get_settings
from miniplat.llm import OllamaClient

router = APIRouter()


def collection_for(embed_model: str) -> str:
    """One Qdrant collection per embedding model (stable name → reusable across restarts)."""
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", embed_model).strip("_")
    return f"miniplat_{safe}"


def point_id(payload: dict) -> str:
    """Deterministic id from content → re-ingesting the same chunk overwrites (dedup)."""
    key = f"{payload.get('title', '')}\x00{payload.get('text', '')}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


# ── Embeddings (open model via Ollama) ───────────────────────────────────────
class Embedder:
    def __init__(self, base_url: str | None = None, default_model: str | None = None) -> None:
        s = get_settings()
        self.base_url = (base_url or s.ollama_base_url).rstrip("/")
        self.default_model = default_model or s.embed_model

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        model = model or self.default_model
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                resp = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                )
                resp.raise_for_status()
                out.append(resp.json()["embedding"])
        return out


# ── Vector store: collection-aware (Qdrant + in-memory for tests) ────────────
@dataclass
class Hit:
    score: float
    title: str
    text: str


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class VectorStore(Protocol):
    async def ensure(self, collection: str, dim: int) -> None: ...
    async def add(
        self, collection: str, vectors: list[list[float]], payloads: list[dict]
    ) -> None: ...
    async def search(self, collection: str, vector: list[float], top_k: int) -> list[Hit]: ...
    async def count(self, collection: str) -> int: ...
    async def list_documents(self, collection: str) -> list[tuple[str, int]]: ...


@dataclass
class InMemoryVectorStore:
    """Collection-aware store for tests (mirrors Qdrant's dedup-by-id behavior)."""

    cols: dict[str, dict[str, tuple[list[float], dict]]] = field(default_factory=dict)

    async def ensure(self, collection: str, dim: int) -> None:
        self.cols.setdefault(collection, {})

    async def add(self, collection: str, vectors: list[list[float]], payloads: list[dict]) -> None:
        col = self.cols.setdefault(collection, {})
        for vec, payload in zip(vectors, payloads, strict=True):
            col[point_id(payload)] = (vec, payload)   # same id overwrites → dedup

    async def search(self, collection: str, vector: list[float], top_k: int) -> list[Hit]:
        items = self.cols.get(collection, {}).values()
        scored = [Hit(_cosine(vector, v), p.get("title", ""), p.get("text", "")) for v, p in items]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    async def count(self, collection: str) -> int:
        return len(self.cols.get(collection, {}))

    async def list_documents(self, collection: str) -> list[tuple[str, int]]:
        docs: dict[str, int] = {}
        for _vec, payload in self.cols.get(collection, {}).values():
            title = payload.get("title", "")
            docs[title] = docs.get(title, 0) + 1
        return sorted(docs.items())


class QdrantVectorStore:
    """Production store backed by Qdrant (data persists in Qdrant's storage volume)."""

    def __init__(self, url: str) -> None:
        from qdrant_client import AsyncQdrantClient

        self._c = AsyncQdrantClient(url=url)

    async def ensure(self, collection: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not await self._c.collection_exists(collection):
            await self._c.create_collection(
                collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    async def add(self, collection: str, vectors: list[list[float]], payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        # Deterministic ids → upserting the same chunk overwrites instead of duplicating.
        points = [
            PointStruct(id=point_id(p), vector=v, payload=p)
            for v, p in zip(vectors, payloads, strict=True)
        ]
        await self._c.upsert(collection, points=points)

    async def search(self, collection: str, vector: list[float], top_k: int) -> list[Hit]:
        res = await self._c.query_points(collection, query=vector, limit=top_k)
        return [Hit(p.score, (p.payload or {}).get("title", ""),
                    (p.payload or {}).get("text", "")) for p in res.points]

    async def count(self, collection: str) -> int:
        if not await self._c.collection_exists(collection):
            return 0
        return (await self._c.count(collection)).count

    async def list_documents(self, collection: str) -> list[tuple[str, int]]:
        if not await self._c.collection_exists(collection):
            return []
        docs: dict[str, int] = {}
        offset = None
        while True:
            points, offset = await self._c.scroll(
                collection, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            for p in points:
                title = (p.payload or {}).get("title", "")
                docs[title] = docs.get(title, 0) + 1
            if offset is None:
                break
        return sorted(docs.items())


def chunk(text: str, size: int = 500) -> list[str]:
    """Split text into <= `size`-char chunks.

    Paragraphs longer than `size` (common in PDFs) are HARD-split, so no single
    chunk can exceed the embedding model's context window (which would 500 Ollama).
    """
    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()] or [text]
    chunks, cur = [], ""
    for p in paras:
        # Hard-split an oversized paragraph into <= size pieces.
        while len(p) > size:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:size])
            p = p[size:]
        if not p:
            continue
        if len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


# ── Providers (overridden in tests) ──────────────────────────────────────────
@lru_cache
def get_store() -> VectorStore:
    return QdrantVectorStore(get_settings().qdrant_url)


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache
def get_generator() -> OllamaClient:
    return OllamaClient()


# ── Schemas ──────────────────────────────────────────────────────────────────
class Document(BaseModel):
    title: str
    text: str


class IngestRequest(BaseModel):
    documents: list[Document]
    embed_model: str | None = None


class IngestResponse(BaseModel):
    chunks_indexed: int
    total_in_collection: int
    collection: str
    embed_model: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 4
    embed_model: str | None = None


class SearchHit(BaseModel):
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class AskRequest(BaseModel):
    query: str
    top_k: int = 4
    embed_model: str | None = None
    chat_model: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    grounded: bool
    embed_model: str
    chat_model: str


class DocItem(BaseModel):
    title: str
    chunks: int


class DocsResponse(BaseModel):
    collection: str
    embed_model: str
    total_chunks: int
    documents: list[DocItem]


# ── Routes ───────────────────────────────────────────────────────────────────
@router.get("/rag/models")
async def rag_models() -> dict[str, list[str]]:
    """Models the user can choose for RAG (for the UI dropdowns)."""
    s = get_settings()
    return {"embed_models": s.embed_model_choices(), "chat_models": s.chat_model_choices()}


@router.get("/documents", response_model=DocsResponse)
async def documents(
    embed_model: str | None = None,
    store: VectorStore = Depends(get_store),
) -> DocsResponse:
    """List which documents are currently indexed (and how many chunks each has)."""
    em = embed_model or get_settings().embed_model
    collection = collection_for(em)
    docs = await store.list_documents(collection)
    return DocsResponse(
        collection=collection, embed_model=em, total_chunks=await store.count(collection),
        documents=[DocItem(title=t, chunks=c) for t, c in docs],
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    req: IngestRequest,
    store: VectorStore = Depends(get_store),
    embedder: Embedder = Depends(get_embedder),
) -> IngestResponse:
    embed_model = req.embed_model or get_settings().embed_model
    collection = collection_for(embed_model)
    chunks, payloads = [], []
    for doc in req.documents:
        for piece in chunk(doc.text):
            chunks.append(piece)
            payloads.append({"title": doc.title, "text": piece})
    if chunks:
        vectors = await embedder.embed(chunks, model=embed_model)
        await store.ensure(collection, len(vectors[0]))
        await store.add(collection, vectors, payloads)
    total = await store.count(collection)
    return IngestResponse(chunks_indexed=len(chunks), total_in_collection=total,
                          collection=collection, embed_model=embed_model)


@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    store: VectorStore = Depends(get_store),
    embedder: Embedder = Depends(get_embedder),
) -> SearchResponse:
    embed_model = req.embed_model or get_settings().embed_model
    collection = collection_for(embed_model)
    vec = (await embedder.embed([req.query], model=embed_model))[0]
    await store.ensure(collection, len(vec))
    hits = await store.search(collection, vec, req.top_k)
    return SearchResponse(
        hits=[SearchHit(title=h.title, score=round(h.score, 4), snippet=h.text[:200]) for h in hits]
    )


@router.post("/ask", response_model=AskResponse)
async def ask(
    req: AskRequest,
    store: VectorStore = Depends(get_store),
    embedder: Embedder = Depends(get_embedder),
    generator: OllamaClient = Depends(get_generator),
) -> AskResponse:
    s = get_settings()
    embed_model = req.embed_model or s.embed_model
    chat_model = req.chat_model or s.chat_model
    collection = collection_for(embed_model)
    vec = (await embedder.embed([req.query], model=embed_model))[0]
    await store.ensure(collection, len(vec))
    hits = await store.search(collection, vec, req.top_k)
    if not hits:
        return AskResponse(answer="I don't have any documents to answer that.",
                           sources=[], grounded=False,
                           embed_model=embed_model, chat_model=chat_model)
    context = "\n\n".join(f"[{i}] {h.title}: {h.text}" for i, h in enumerate(hits, 1))
    prompt = (
        "Answer the question using ONLY the context. Cite sources like [1]. "
        f"If the context is insufficient, say you don't know.\n\nContext:\n{context}\n\n"
        f"Question: {req.query}"
    )
    answer = await generator.chat(prompt, model=chat_model)
    return AskResponse(answer=answer, sources=[h.title for h in hits], grounded=True,
                       embed_model=embed_model, chat_model=chat_model)
