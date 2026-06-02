from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://rag:rag_secret@localhost:5432/ragdb"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    embedding_model: str = "intfloat/multilingual-e5-large"
    llm_temperature: float = 0.0
    vector_store_path: Path = Path("./data/vector_store")
    upload_dir: Path = Path("./data/uploads")
    chunk_size: int = 512
    chunk_overlap: int = 64
    turbovec_bit_width: int = 4
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    hybrid_search_enabled: bool = True
    top_k: int = 8
    retrieval_min_k: int = 8

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_database_url(cls, v: str) -> str:
        """
        Render commonly injects DATABASE_URL as postgres://... (sync driver).
        SQLAlchemy async engine expects postgresql+asyncpg://...
        """
        if not isinstance(v, str):
            return v

        raw = v.strip()
        if raw.startswith("postgresql+asyncpg://"):
            return raw
        if raw.startswith("postgres://"):
            return "postgresql+asyncpg://" + raw.removeprefix("postgres://")
        if raw.startswith("postgresql://"):
            return "postgresql+asyncpg://" + raw.removeprefix("postgresql://")
        return raw


@lru_cache
def get_settings() -> Settings:
    return Settings()
