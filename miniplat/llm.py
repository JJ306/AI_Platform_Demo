"""A thin async client for an open-weight model served by Ollama."""

from __future__ import annotations

import httpx

from miniplat.config import get_settings


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.chat_model

    async def chat(self, message: str, model: str | None = None) -> str:
        """Send one user message to the model and return its text reply.

        Pass ``model`` to answer with a specific model (else the default).
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model or self.model,
                    "messages": [{"role": "user", "content": message}],
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return data.get("message", {}).get("content", "")
