"""A thin async client for an open-weight model served by Ollama.
"""

from __future__ import annotations

import httpx  # noqa: F401  (you'll use this when you implement chat)

from miniplat.config import get_settings


class OllamaClient:
	def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
		settings = get_settings()
		self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
		self.model = model or settings.chat_model

	async def chat(self, message: str) -> str:
		"""Send one user message to the model and return its text reply"""
		# here we can also add prompt to get better answer.
		url = f"{self.base_url}/api/chat"
		payload = {
			"model": self.model,
			"messages": [{"role": "user", "content": message}],
			"stream": False
		}
		
		# Using the requested 120.0-second timeout for long inference times
		async with httpx.AsyncClient(timeout=120.0) as client:
			response = await client.post(url, json=payload)
			response.raise_for_status()
			
			data = response.json()
			return data["message"]["content"]
