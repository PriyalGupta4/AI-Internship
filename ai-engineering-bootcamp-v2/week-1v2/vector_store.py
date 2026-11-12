"""Pinecone vector store + OpenAI embeddings for the RAG extension of `/ask`.

All configuration comes from environment variables (see `.env.example`).
Nothing here runs at import time, so the API still starts — and `/ask` still
works — when Pinecone is not configured yet.
"""

import os
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone

# One embedding model for BOTH ingest and query. Vectors from different models
# live in different spaces, so mixing them silently ruins retrieval. This is a
# constant on purpose, not an env var, so the two sides can never drift apart.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # Your Pinecone index must be created with this dimension.
EMBEDDING_PRICE_PER_1K = 0.00002  # USD, text-embedding-3-small ($0.02 / 1M tokens)
EMBEDDING_BATCH_SIZE = 100  # Chunks per OpenAI embeddings request.

# Chunking defaults; override with CHUNK_SIZE / CHUNK_OVERLAP env vars (characters).
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# document_id becomes the prefix of every chunk ID ("handbook#chunk-0"), and
# Pinecone IDs/prefixes must be ASCII. "#" is reserved as the separator.
DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
CHUNK_ID_SEPARATOR = "#chunk-"


class VectorStoreConfigError(RuntimeError):
    """Raised when required env vars are missing or invalid."""


@dataclass(frozen=True)
class PineconeSettings:
    api_key: str
    index_name: str
    index_host: str | None
    namespace: str

    @classmethod
    def from_env(cls) -> "PineconeSettings":
        api_key = os.getenv("PINECONE_API_KEY", "").strip()
        index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
        missing = [
            name
            for name, value in (
                ("PINECONE_API_KEY", api_key),
                ("PINECONE_INDEX_NAME", index_name),
            )
            if not value
        ]
        if missing:
            raise VectorStoreConfigError(
                f"Missing required env var(s): {', '.join(missing)}"
            )
        return cls(
            api_key=api_key,
            index_name=index_name,
            index_host=os.getenv("PINECONE_INDEX_HOST", "").strip() or None,
            namespace=os.getenv("PINECONE_NAMESPACE", "").strip(),
        )


_openai: OpenAI | None = None
_index: Any = None
_settings: PineconeSettings | None = None


def get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI()  # Reads OPENAI_API_KEY from the environment.
    return _openai


def get_settings() -> PineconeSettings:
    global _settings
    if _settings is None:
        _settings = PineconeSettings.from_env()
    return _settings


def get_index():
    """Return a cached Pinecone data-plane client for the configured index."""

    global _index
    if _index is None:
        settings = get_settings()
        pc = Pinecone(api_key=settings.api_key)
        if settings.index_host:
            # Skips the extra control-plane lookup on every cold start.
            _index = pc.index(host=settings.index_host)
        else:
            _index = pc.index(name=settings.index_name)
    return _index


def embedding_cost_usd(tokens: int) -> float:
    return tokens / 1000 * EMBEDDING_PRICE_PER_1K


def chunk_settings() -> tuple[int, int]:
    """Read (chunk_size, chunk_overlap) from env, falling back to 800 / 100."""

    try:
        size = int(os.getenv("CHUNK_SIZE", "").strip() or DEFAULT_CHUNK_SIZE)
        overlap = int(os.getenv("CHUNK_OVERLAP", "").strip() or DEFAULT_CHUNK_OVERLAP)
    except ValueError as exc:
        raise VectorStoreConfigError("CHUNK_SIZE and CHUNK_OVERLAP must be integers") from exc
    if size <= 0 or overlap < 0 or overlap >= size:
        raise VectorStoreConfigError(
            f"Need CHUNK_SIZE > 0 and 0 <= CHUNK_OVERLAP < CHUNK_SIZE (got {size} / {overlap})"
        )
    return size, overlap


def chunk_text(text: str) -> list[str]:
    """Split text the same way as the Week 2 notebook: paragraphs, then lines, then words."""

    size, overlap = chunk_settings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


def embed_texts(texts: list[str]) -> tuple[list[list[float]], int]:
    """Embed texts with EMBEDDING_MODEL, in batches. Returns (vectors, tokens_used)."""

    vectors: list[list[float]] = []
    tokens = 0
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = get_openai().embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        tokens += response.usage.total_tokens if response.usage else 0
    return vectors, tokens


def upsert_chunks(chunks: list[dict[str, Any]]) -> tuple[int, int]:
    """Embed and store chunks shaped like {"id", "text", "metadata"?}.

    The chunk text is saved in metadata under "text" so queries can return it.
    Returns (vectors_upserted, embedding_tokens_used).
    """

    if not chunks:
        return 0, 0
    vectors, tokens = embed_texts([chunk["text"] for chunk in chunks])
    records = [
        {
            "id": chunk["id"],
            "values": vector,
            "metadata": {**chunk.get("metadata", {}), "text": chunk["text"]},
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    get_index().upsert(
        vectors=records,
        namespace=get_settings().namespace,
        batch_size=100,
        show_progress=False,
    )
    return len(records), tokens


def chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}{CHUNK_ID_SEPARATOR}{chunk_index}"


def ingest_document(
    document_id: str,
    text: str,
    source: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Chunk, embed and upsert one document; re-ingesting the same ID replaces it.

    Every chunk gets metadata document_id, chunk_index and source (plus any
    extra_metadata). New chunks are written first and leftovers from a longer
    previous version are deleted afterwards, so a failed upsert never wipes the
    old copy. Returns chunks_indexed, chunks_removed and embedding tokens.
    """

    chunks = chunk_text(text)
    records = [
        {
            "id": chunk_id(document_id, index),
            "text": chunk,
            "metadata": {
                **(extra_metadata or {}),
                "document_id": document_id,
                "chunk_index": index,
                "source": source,
            },
        }
        for index, chunk in enumerate(chunks)
    ]
    chunks_indexed, tokens = upsert_chunks(records)

    new_ids = {record["id"] for record in records}
    stale_ids = [
        item.id
        for page in get_index().list(
            prefix=f"{document_id}{CHUNK_ID_SEPARATOR}",
            namespace=get_settings().namespace,
        )
        for item in page.vectors
        if item.id not in new_ids
    ]
    for start in range(0, len(stale_ids), 1000):  # Pinecone deletes up to 1000 IDs per call.
        get_index().delete(
            ids=stale_ids[start : start + 1000], namespace=get_settings().namespace
        )

    return {
        "chunks_indexed": chunks_indexed,
        "chunks_removed": len(stale_ids),
        "embedding_tokens": tokens,
    }


def query_similar(question: str, top_k: int = 4) -> tuple[list[dict[str, Any]], int]:
    """Embed the question with the SAME model and return the top_k matches.

    Returns (matches, embedding_tokens_used); each match has id, score, text, metadata.
    """

    vectors, tokens = embed_texts([question])
    result = get_index().query(
        vector=vectors[0],
        top_k=top_k,
        namespace=get_settings().namespace,
        include_metadata=True,
    )
    matches = []
    for match in result.matches:
        metadata = dict(match.metadata or {})
        matches.append(
            {
                "id": match.id,
                "score": match.score,
                "text": metadata.pop("text", ""),
                "metadata": metadata,
            }
        )
    return matches, tokens


def pinecone_health() -> dict[str, Any]:
    """Check Pinecone without spending OpenAI tokens.

    Confirms the env vars are set, the API key works, the index exists, and its
    dimension matches EMBEDDING_DIMENSIONS. Never includes the API key in output.
    """

    report: dict[str, Any] = {
        "ok": False,
        "embedding_model": EMBEDDING_MODEL,
        "expected_dimension": EMBEDDING_DIMENSIONS,
    }
    try:
        settings = get_settings()
    except VectorStoreConfigError as exc:
        report["error"] = str(exc)
        return report

    report["index_name"] = settings.index_name
    report["namespace"] = settings.namespace or "__default__"
    start = time.perf_counter()
    try:
        stats = get_index().describe_index_stats()
    except Exception as exc:  # Surface auth / not-found / network errors as data.
        global _index
        _index = None  # Retry a fresh connection on the next call.
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    report["latency_ms"] = int((time.perf_counter() - start) * 1000)
    report["index_dimension"] = stats.dimension
    report["total_vector_count"] = stats.total_vector_count
    report["namespaces"] = {
        name: summary.vector_count for name, summary in (stats.namespaces or {}).items()
    }
    if stats.dimension != EMBEDDING_DIMENSIONS:
        report["error"] = (
            f"Index dimension is {stats.dimension}, but {EMBEDDING_MODEL} produces "
            f"{EMBEDDING_DIMENSIONS}. Recreate the index with dimension "
            f"{EMBEDDING_DIMENSIONS}."
        )
        return report

    report["ok"] = True
    return report
