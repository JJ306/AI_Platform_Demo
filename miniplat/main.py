from __future__ import annotations

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from miniplat.llm import OllamaClient

app = FastAPI(title="miniplat", version="0.1.0")


def get_client() -> OllamaClient:
    """Dependency: the model client. Overridden in tests with a fake."""
    return OllamaClient()


class ChatRequest(BaseModel):
    message: str
    model: str | None = None  # optional override for the model to use (e.g. "llama3.2:3b")


class ChatResponse(BaseModel):
    reply: str
    model: str


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, client: OllamaClient = Depends(get_client)) -> ChatResponse:
    client.model = req.model or client.model
    reply = await client.chat(req.message)
    return ChatResponse(reply=reply, model=client.model)


