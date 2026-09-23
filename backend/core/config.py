"""
ResearchMind — core configuration
Loads from .env and exposes typed settings used everywhere.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    # LLM
    openai_api_key: str        = os.getenv("OPENAI_API_KEY", "")
    openai_model: str          = os.getenv("OPENAI_MODEL",      "gpt-4o")
    openai_fast_model: str     = os.getenv("OPENAI_FAST_MODEL", "gpt-4o-mini")

    # Vector DB
    qdrant_url: str            = os.getenv("QDRANT_URL",        "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY",    None)

    # External APIs
    semantic_scholar_key: str | None = os.getenv("SEMANTIC_SCHOLAR_API_KEY", None)
    youtube_api_key:      str | None = os.getenv("YOUTUBE_API_KEY",          None)

    # App
    frontend_url: str  = os.getenv("FRONTEND_URL", "http://localhost:5173")
    secret_key:   str  = os.getenv("SECRET_KEY",   "dev-secret")
    redis_url:    str  = os.getenv("REDIS_URL",     "redis://localhost:6379/0")

    # Paths
    upload_dir: Path   = BASE_DIR / os.getenv("UPLOAD_DIR",  "uploads")
    export_dir: Path   = BASE_DIR / os.getenv("EXPORT_DIR",  "exports")

    # RAG
    chunk_size:    int = 800
    chunk_overlap: int = 120
    top_k:         int = 6

    # Quiz
    pass_threshold: int = 70   # percent

    model_config = {"arbitrary_types_allowed": True}


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    s.export_dir.mkdir(parents=True, exist_ok=True)
    return s
