""" configuration. Every value comes from a MINIPLAT_* env var.

"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINIPLAT_", env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.2:3b"


@lru_cache
def get_settings() -> Settings:
    return Settings()
