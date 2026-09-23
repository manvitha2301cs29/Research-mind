"""
ResearchMind — shared LLM + vector store clients
Single-instance objects reused across all agents.
"""
from __future__ import annotations

import logging
from functools import lru_cache  # still used by get_qdrant_client

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore as Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from core.config import get_settings

log = logging.getLogger(__name__)
cfg = get_settings()

# ── LLM clients ──────────────────────────────────────────────────────────────

class MissingAPIKeyError(RuntimeError):
    """Raised when neither the server nor the request supplies an OpenAI key."""


def _resolve_key() -> str:
    from core.request_context import get_request_api_key
    key = get_request_api_key() or cfg.openai_api_key
    if not key:
        raise MissingAPIKeyError(
            "No OpenAI API key: set OPENAI_API_KEY on the server or add your key in the app."
        )
    return key


def get_llm(fast: bool = False) -> ChatOpenAI:
    key = _resolve_key()
    model = cfg.openai_fast_model if fast else cfg.openai_model
    return ChatOpenAI(model=model, openai_api_key=key, temperature=0.2, streaming=True)


def get_embeddings() -> OpenAIEmbeddings:
    key = _resolve_key()
    return OpenAIEmbeddings(openai_api_key=key)


# ── Vector store ─────────────────────────────────────────────────────────────

COLLECTION = "researchmind_papers"
VECTOR_DIM  = 1536   # text-embedding-ada-002 / text-embedding-3-small


@lru_cache
def get_qdrant_client() -> QdrantClient:
    url = cfg.qdrant_url
    if not url or url == ":memory:" or "YOUR-CLUSTER-ID" in url:
        log.warning("QDRANT_URL not set — using in-memory Qdrant (RAG data won't persist).")
        return QdrantClient(":memory:")
    try:
        client = QdrantClient(url=url, api_key=cfg.qdrant_api_key or None, timeout=5, check_compatibility=False)
        client.get_collections()          # fail fast if the server isn't reachable
        return client
    except Exception as e:
        log.warning("Qdrant at %s unreachable (%s) — falling back to in-memory Qdrant.", url, e)
        return QdrantClient(":memory:")


def ensure_collection() -> None:
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def get_vectorstore(session_id: str) -> Qdrant:
    """Return a Qdrant vectorstore scoped to a session (via metadata filter)."""
    ensure_collection()
    return Qdrant(
        client=get_qdrant_client(),
        collection_name=COLLECTION,
        embeddings=get_embeddings(),
        metadata_payload_key="metadata",
    )
