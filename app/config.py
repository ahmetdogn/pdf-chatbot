from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "low"
    claude_max_tokens: int = 8000

    embedder: str = "fastembed"  # oder "hash" für tests
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    data_dir: Path = Path("data")
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 6
    max_upload_mb: int = 25


@lru_cache
def get_settings():
    return Settings()
