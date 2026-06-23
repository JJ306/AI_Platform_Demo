""" configuration. Every value comes from a MINIPLAT_* env var.

"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINIPLAT_", env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text"
    qdrant_url: str = "http://localhost:6333"  # persistent vector store (Qdrant) for RAG

    # Models the user may pick for RAG (comma-separated). Pull them in Ollama first.
    embed_models: str = "nomic-embed-text"
    chat_models: str = "llama3.2:3b"
    
    def embed_model_choices(self) -> list[str]:
        return _csv(self.embed_models, self.embed_model)

    def chat_model_choices(self) -> list[str]:
        return _csv(self.chat_models, self.chat_model)


def _csv(value: str, default: str) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    if default not in items:
        items.insert(0, default)
    return items

@lru_cache
def get_settings() -> Settings:
    return Settings()
