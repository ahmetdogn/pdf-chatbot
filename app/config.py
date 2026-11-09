from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "low"
    claude_max_tokens: int = 8000

    # wenn gesetzt muss man im frontend ein passwort eingeben bevor man chatten kann
    demo_password: str = ""

    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 6
    max_upload_mb: int = 4  # vercel erlaubt max 4.5 mb pro request

    # nur für die evaluation, im frontend läuft das gleiche modell mit transformers.js
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache
def get_settings():
    return Settings()
