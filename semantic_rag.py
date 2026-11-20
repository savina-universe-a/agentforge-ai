"""Semantic RAG using Gemini embeddings (google-genai) and MongoDB Atlas Vector Search."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from typing import Any, Literal

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import Field
from pymongo import MongoClient
from pymongo.errors import ConfigurationError, OperationFailure, PyMongoError

from database import (
    DEFAULT_TIMEOUT_MS,
    DatabaseConfigError,
    DatabaseError,
    _format_mongo_error,
    _read_dotenv_var,
    _redact_sensitive_text,
    compute_document_id,
    get_mongodb_db_name,
    get_mongodb_uri,
)
from planner import MissingAPIKeyError, _sanitize_error_text, get_api_key
from rag import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TOP_K,
    MAX_QUESTION_LENGTH,
    DocumentChunk,
    DocumentRecord,
    KnowledgeBaseIndex,
    RAGAnswerResult,
    RetrievedChunk,
    answer_question_from_knowledge_base,
    generate_grounded_answer,
)

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSIONS = 768
CHUNKS_COLLECTION = "document_chunks"
DEFAULT_VECTOR_INDEX_NAME = "chunk_embedding_index"
DEFAULT_EMBED_BATCH_SIZE = 16
DEFAULT_NUM_CANDIDATES = 40
DEFAULT_MIN_VECTOR_SCORE = 0.55

TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_TYPE_QUERY = "RETRIEVAL_QUERY"


class EmbeddingError(RuntimeError):
    """Raised when Gemini embedding generation fails or returns invalid vectors."""


class VectorIndexNotReadyError(DatabaseError):
    """Raised when the MongoDB Atlas Vector Search index is missing or not queryable."""


class SemanticRAGResult(RAGAnswerResult):
    """Grounded RAG answer annotated with the exact retrieval method used."""

    retrieval_mode: Literal["vector", "tfidf_fallback"] = "vector"
    fallback_reason: str | None = None
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    index_name: str = Field(default=DEFAULT_VECTOR_INDEX_NAME)


def get_embedding_model_name(
    override_model: str | None = None,
    env_path: Path | None = None,
) -> str:
    """Return GEMINI_EMBEDDING_MODEL from override, environment, or .env (default: gemini-embedding-001)."""
    if override_model and override_model.strip():
        return override_model.strip()

    configured = _read_dotenv_var("GEMINI_EMBEDDING_MODEL", env_path=env_path)
    return configured or DEFAULT_EMBEDDING_MODEL


def get_embedding_dimensions(
    override_dimensions: int | None = None,
    env_path: Path | None = None,
) -> int:
    """Return GEMINI_EMBEDDING_DIMENSIONS (default: 768)."""
    if override_dimensions is not None:
        if override_dimensions <= 0:
            raise ValueError("Embedding dimensions must be a positive integer.")
        return int(override_dimensions)

    raw = _read_dotenv_var("GEMINI_EMBEDDING_DIMENSIONS", env_path=env_path)
    if not raw:
        return DEFAULT_EMBEDDING_DIMENSIONS
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"GEMINI_EMBEDDING_DIMENSIONS must be an integer, got '{raw}'.") from exc
    if parsed <= 0:
        raise ValueError("GEMINI_EMBEDDING_DIMENSIONS must be a positive integer.")
    return parsed


def get_vector_index_name(
    override_index: str | None = None,
    env_path: Path | None = None,
) -> str:
    """Return MONGODB_VECTOR_INDEX_NAME (default: chunk_embedding_index)."""
    if override_index and override_index.strip():
        return override_index.strip()

    configured = _read_dotenv_var("MONGODB_VECTOR_INDEX_NAME", env_path=env_path)
    return configured or DEFAULT_VECTOR_INDEX_NAME


def get_vector_index_definition(dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> dict[str, Any]:
    """Return the exact MongoDB Atlas Vector Search JSON index definition."""
    return {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": int(dimensions),
                "similarity": "cosine",
            },
            {
                "type": "filter",
                "path": "doc_id",
            },
            {
                "type": "filter",
                "path": "doc_name",
            },
        ]
    }


def l2_normalize(vector: list[float]) -> list[float]:
    """L2-normalize an embedding vector (required for truncated Matryoshka Gemini embeddings)."""
    if not vector:
        raise EmbeddingError("Received an empty embedding vector from Gemini.")

    values = [float(v) for v in vector]
    if not all(math.isfinite(v) for v in values):
        raise EmbeddingError("Received a non-finite value in an embedding vector.")

    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0.0:
        raise EmbeddingError("Received a zero-norm embedding vector from Gemini.")

    return [v / norm for v in values]


def compute_chunk_hash(text: str, embedding_model: str, dimensions: int) -> str:
    """Compute a deterministic hash so unchanged chunks are never re-embedded."""
    payload = f"{embedding_model}|{dimensions}|{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _format_embedding_api_error(
    exc: genai_errors.APIError,
    model_name: str,
    api_key: str | None,
) -> str:
    """Convert a google-genai APIError into a clear, secret-free embedding error message."""
    code = getattr(exc, "code", None)
    if code in (401, 403):
        return "Gemini rejected GEMINI_API_KEY for embeddings (401/403). Verify key access."
    if code == 429:
        return "Gemini embedding quota or rate limit exceeded (429). Wait briefly or check billing/quota."
    if code == 404:
        return (
            f"Gemini embedding model '{model_name}' was not found or is unavailable (404). "
            f"Check GEMINI_EMBEDDING_MODEL in .env (default: '{DEFAULT_EMBEDDING_MODEL}')."
        )
    if code == 503:
        return f"Gemini embedding model '{model_name}' is temporarily experiencing high demand (503)."
    safe = _sanitize_error_text(str(exc), api_key)
    return f"Gemini embedding request failed: {safe}"


def embed_texts(
    texts: list[str],
    task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"] = TASK_TYPE_DOCUMENT,
    api_key: str | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    client: genai.Client | None = None,
    env_path: Path | None = None,
) -> list[list[float]]:
    """Generate L2-normalized Gemini embeddings in small batches with strict dimension validation."""
    if not texts:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    resolved_model = get_embedding_model_name(override_model=model, env_path=env_path)
    resolved_dims = get_embedding_dimensions(override_dimensions=dimensions, env_path=env_path)

    resolved_key: str | None = None
    if client is None:
        resolved_key = get_api_key(override_key=api_key, env_path=env_path)
        if not resolved_key:
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is not set. Add it to your local .env file or enter it "
                "privately in the sidebar under API Configuration."
            )

    try:
        genai_client = (
            client
            if client is not None
            else genai.Client(
                api_key=resolved_key,
                vertexai=False,
                http_options=types.HttpOptions(timeout=int(request_timeout_seconds * 1000)),
            )
        )
    except Exception as exc:  # noqa: BLE001
        safe = _sanitize_error_text(str(exc), resolved_key)
        raise EmbeddingError(f"Failed to initialize Gemini embedding client: {safe}") from exc

    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=resolved_dims,
    )

    all_vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            response = genai_client.models.embed_content(
                model=resolved_model,
                contents=batch,
                config=config,
            )
        except genai_errors.APIError as exc:
            raise EmbeddingError(_format_embedding_api_error(exc, resolved_model, resolved_key)) from exc
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise EmbeddingError(
                f"Gemini embedding request timed out after {request_timeout_seconds:.0f}s."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            safe = _sanitize_error_text(str(exc), resolved_key)
            raise EmbeddingError(f"Unexpected embedding error: {safe}") from exc

        embeddings = list(getattr(response, "embeddings", None) or [])
        if len(embeddings) != len(batch):
            raise EmbeddingError(
                f"Expected {len(batch)} embedding(s) from Gemini, received {len(embeddings)}."
            )

        for emb in embeddings:
            raw_values = list(getattr(emb, "values", None) or [])
            if len(raw_values) != resolved_dims:
                raise EmbeddingError(
                    f"Embedding dimension mismatch: expected {resolved_dims}, received {len(raw_values)}."
                )
            all_vectors.append(l2_normalize(raw_values))

    return all_vectors


class SemanticVectorStore:
    """Dedicated MongoDB Atlas Vector Search store for embedded document chunks."""

    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
        index_name: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        client: MongoClient[Any] | None = None,
        env_path: Path | None = None,
    ) -> None:
        self._env_path = env_path
        self._timeout_ms = max(int(timeout_ms), 500)
        self.db_name = get_mongodb_db_name(override_db=db_name, env_path=env_path)
        self.index_name = get_vector_index_name(override_index=index_name, env_path=env_path)
        self._uri: str | None = None
        self._owns_client = client is None

        if client is not None:
            self._client: MongoClient[Any] = client
        else:
            resolved_uri = get_mongodb_uri(override_uri=uri, env_path=env_path)
            if not resolved_uri:
                raise DatabaseConfigError(
                    "MONGODB_URI is not configured. Add MONGODB_URI to your local .env file "
                    "to use MongoDB Atlas Vector Search."
                )
            if not resolved_uri.startswith(("mongodb://", "mongodb+srv://")):
                raise DatabaseConfigError(
                    "MONGODB_URI must start with 'mongodb://' or 'mongodb+srv://'."
                )
            self._uri = resolved_uri
            try:
                self._client = MongoClient(
                    resolved_uri,
                    serverSelectionTimeoutMS=self._timeout_ms,
                    connectTimeoutMS=self._timeout_ms,
                    socketTimeoutMS=self._timeout_ms,
                    appname="AgentForgeAI",
                )
            except (ConfigurationError, PyMongoError, ValueError, OSError) as exc:
                raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc

        self._db = self._client[self.db_name]
        self._chunks = self._db[CHUNKS_COLLECTION]

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    def __enter__(self) -> SemanticVectorStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def count_indexed_chunks(self) -> int:
        """Return the number of embedded chunks currently stored in document_chunks."""
        try:
            return int(self._chunks.count_documents({}))
        except (PyMongoError, TimeoutError) as exc:
            raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc

    def check_vector_index_status(self) -> dict[str, Any]:
        """Inspect Atlas Vector Search index readiness on the document_chunks collection."""
        try:
            indexes = list(self._chunks.list_search_indexes(name=self.index_name))
        except TypeError:
            # Handle minimal mocks that do not accept keyword 'name'
            try:
                indexes = [
                    idx
                    for idx in list(self._chunks.list_search_indexes())
                    if idx.get("name") == self.index_name
                ]
            except (PyMongoError, TimeoutError) as exc:
                raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc
        except OperationFailure as exc:
            code = getattr(exc, "code", None)
            if code in (26, 59):  # NamespaceNotFound / CommandNotFound
                return {
                    "exists": False,
                    "queryable": False,
                    "status": "MISSING",
                    "index_name": self.index_name,
                    "collection": CHUNKS_COLLECTION,
                    "database": self.db_name,
                }
            raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc
        except (PyMongoError, TimeoutError) as exc:
            raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc

        matching = [idx for idx in indexes if idx.get("name") == self.index_name]
        if not matching:
            return {
                "exists": False,
                "queryable": False,
                "status": "MISSING",
                "index_name": self.index_name,
                "collection": CHUNKS_COLLECTION,
                "database": self.db_name,
            }

        idx_info = matching[0]
        status = str(idx_info.get("status", "UNKNOWN")).upper()
        queryable = bool(idx_info.get("queryable", status == "READY"))
        return {
            "exists": True,
            "queryable": queryable and status == "READY",
            "status": status,
            "index_name": self.index_name,
            "collection": CHUNKS_COLLECTION,
            "database": self.db_name,
        }

    def index_document_chunks(
        self,
        record: DocumentRecord,
        chunks: list[DocumentChunk],
        api_key: str | None = None,
        embedding_model: str | None = None,
        dimensions: int | None = None,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        genai_client: genai.Client | None = None,
    ) -> dict[str, Any]:
        """Idempotently embed and upsert changed chunks while deleting stale orphan chunks."""
        if not chunks:
            raise ValueError(f"Cannot embed '{record.doc_name}' without indexed chunks.")

        resolved_model = get_embedding_model_name(
            override_model=embedding_model, env_path=self._env_path
        )
        resolved_dims = get_embedding_dimensions(
            override_dimensions=dimensions, env_path=self._env_path
        )
        doc_id = compute_document_id(record.doc_name)

        try:
            existing_cursor = self._chunks.find(
                {"doc_id": doc_id},
                {"_id": 1, "chunk_hash": 1},
            )
            existing_hashes: dict[str, str] = {
                str(item["_id"]): str(item.get("chunk_hash", ""))
                for item in existing_cursor
            }
        except (PyMongoError, TimeoutError) as exc:
            raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc

        chunks_to_embed: list[tuple[DocumentChunk, str]] = []
        current_chunk_ids: set[str] = set()
        skipped_count = 0

        for chunk in chunks:
            current_chunk_ids.add(chunk.chunk_id)
            safe_text = _redact_sensitive_text(chunk.text, self._uri)
            chunk_hash = compute_chunk_hash(safe_text, resolved_model, resolved_dims)
            if existing_hashes.get(chunk.chunk_id) == chunk_hash:
                skipped_count += 1
            else:
                chunks_to_embed.append((chunk, chunk_hash))

        stale_chunk_ids = sorted(set(existing_hashes.keys()) - current_chunk_ids)

        if chunks_to_embed:
            texts_to_embed = [
                _redact_sensitive_text(chunk.text, self._uri)
                for chunk, _ in chunks_to_embed
            ]
            vectors = embed_texts(
                texts=texts_to_embed,
                task_type=TASK_TYPE_DOCUMENT,
                api_key=api_key,
                model=resolved_model,
                dimensions=resolved_dims,
                batch_size=batch_size,
                client=genai_client,
                env_path=self._env_path,
            )

            now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                for (chunk, chunk_hash), vector in zip(chunks_to_embed, vectors, strict=True):
                    payload: dict[str, Any] = {
                        "_id": chunk.chunk_id,
                        "doc_id": doc_id,
                        "chunk_id": chunk.chunk_id,
                        "doc_name": chunk.doc_name,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "citation_label": chunk.citation_label,
                        "text": _redact_sensitive_text(chunk.text, self._uri),
                        "chunk_hash": chunk_hash,
                        "embedding_model": resolved_model,
                        "dimensions": resolved_dims,
                        "embedding": vector,
                        "updated_at": now_iso,
                    }
                    self._chunks.replace_one({"_id": chunk.chunk_id}, payload, upsert=True)
            except (PyMongoError, TimeoutError) as exc:
                raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc

        deleted_stale = 0
        if stale_chunk_ids:
            try:
                res = self._chunks.delete_many(
                    {"doc_id": doc_id, "_id": {"$in": stale_chunk_ids}}
                )
                deleted_stale = int(getattr(res, "deleted_count", len(stale_chunk_ids)))
            except (PyMongoError, TimeoutError) as exc:
                raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc

        return {
            "doc_id": doc_id,
            "doc_name": record.doc_name,
            "embedded": len(chunks_to_embed),
            "skipped": skipped_count,
            "deleted_stale": deleted_stale,
            "embedding_model": resolved_model,
            "dimensions": resolved_dims,
        }

    def index_knowledge_base(
        self,
        index: KnowledgeBaseIndex,
        api_key: str | None = None,
        embedding_model: str | None = None,
        dimensions: int | None = None,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        genai_client: genai.Client | None = None,
    ) -> dict[str, Any]:
        """Idempotently embed and sync all documents currently loaded in KnowledgeBaseIndex."""
        totals = {
            "documents_processed": 0,
            "embedded": 0,
            "skipped": 0,
            "deleted_stale": 0,
            "embedding_model": get_embedding_model_name(
                override_model=embedding_model, env_path=self._env_path
            ),
            "dimensions": get_embedding_dimensions(
                override_dimensions=dimensions, env_path=self._env_path
            ),
        }
        for doc in index.documents:
            record, doc_chunks, _ = index.get_document_bundle(doc.doc_name)
            stats = self.index_document_chunks(
                record=record,
                chunks=doc_chunks,
                api_key=api_key,
                embedding_model=embedding_model,
                dimensions=dimensions,
                batch_size=batch_size,
                genai_client=genai_client,
            )
            totals["documents_processed"] += 1
            totals["embedded"] += stats["embedded"]
            totals["skipped"] += stats["skipped"]
            totals["deleted_stale"] += stats["deleted_stale"]
        return totals

    def vector_search(
        self,
        query: str,
        api_key: str | None = None,
        embedding_model: str | None = None,
        dimensions: int | None = None,
        top_k: int = DEFAULT_TOP_K,
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        min_score: float = DEFAULT_MIN_VECTOR_SCORE,
        genai_client: genai.Client | None = None,
    ) -> list[RetrievedChunk]:
        """Run MongoDB Atlas $vectorSearch with a RETRIEVAL_QUERY Gemini embedding."""
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise ValueError("Please enter a question to search the Knowledge Base.")
        if len(cleaned_query) > MAX_QUESTION_LENGTH:
            raise ValueError(
                f"Question exceeds the maximum length of {MAX_QUESTION_LENGTH} characters."
            )

        index_status = self.check_vector_index_status()
        if not index_status["exists"]:
            raise VectorIndexNotReadyError(
                f"Atlas Vector Search index '{self.index_name}' was not found on "
                f"'{self.db_name}.{CHUNKS_COLLECTION}'. Create it in MongoDB Atlas Search first."
            )
        if not index_status["queryable"]:
            raise VectorIndexNotReadyError(
                f"Atlas Vector Search index '{self.index_name}' is currently '{index_status['status']}' "
                "and not ready for queries yet. Wait for it to become READY in Atlas."
            )

        query_vectors = embed_texts(
            texts=[cleaned_query],
            task_type=TASK_TYPE_QUERY,
            api_key=api_key,
            model=embedding_model,
            dimensions=dimensions,
            batch_size=1,
            client=genai_client,
            env_path=self._env_path,
        )
        query_vector = query_vectors[0]

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": max(int(num_candidates), int(top_k) * 10),
                    "limit": int(top_k),
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "doc_name": 1,
                    "page_number": 1,
                    "chunk_index": 1,
                    "citation_label": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            raw_results = list(self._chunks.aggregate(pipeline))
        except OperationFailure as exc:
            err_lower = str(exc).lower()
            if "index" in err_lower or "vectorsearch" in err_lower or getattr(exc, "code", None) in (291, 40324):
                raise VectorIndexNotReadyError(
                    f"Atlas Vector Search index '{self.index_name}' could not be queried. "
                    "Verify the index exists, is READY, and matches 768 dimensions."
                ) from exc
            raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc
        except (PyMongoError, TimeoutError) as exc:
            raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc

        retrieved: list[RetrievedChunk] = []
        for item in raw_results:
            score = float(item.get("score", 0.0))
            if score < min_score:
                continue
            chunk = DocumentChunk(
                chunk_id=item["chunk_id"],
                doc_name=item["doc_name"],
                page_number=item.get("page_number"),
                chunk_index=int(item["chunk_index"]),
                citation_label=item["citation_label"],
                text=item["text"],
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=round(score, 4)))

        return retrieved


def answer_question_semantic(
    question: str,
    fallback_index: KnowledgeBaseIndex | None = None,
    allow_tfidf_fallback: bool = True,
    api_key: str | None = None,
    model: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_vector_score: float = DEFAULT_MIN_VECTOR_SCORE,
    vector_store: SemanticVectorStore | None = None,
    genai_client: genai.Client | None = None,
    env_path: Path | None = None,
) -> SemanticRAGResult:
    """Answer a question via Atlas Vector Search, with explicit, transparent TF-IDF fallback."""
    cleaned_question = (question or "").strip()
    if not cleaned_question:
        raise ValueError("Please enter a question before querying the Knowledge Base.")
    if len(cleaned_question) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"Question exceeds the maximum length of {MAX_QUESTION_LENGTH} characters."
        )

    resolved_embed_model = get_embedding_model_name(
        override_model=embedding_model, env_path=env_path
    )
    resolved_index_name = (
        vector_store.index_name
        if vector_store is not None
        else get_vector_index_name(env_path=env_path)
    )

    fallback_reason: str | None = None
    sources: list[RetrievedChunk] | None = None

    try:
        if vector_store is not None:
            sources = vector_store.vector_search(
                query=cleaned_question,
                api_key=api_key,
                embedding_model=resolved_embed_model,
                dimensions=dimensions,
                top_k=top_k,
                min_score=min_vector_score,
                genai_client=genai_client,
            )
        else:
            with SemanticVectorStore(env_path=env_path) as store:
                resolved_index_name = store.index_name
                sources = store.vector_search(
                    query=cleaned_question,
                    api_key=api_key,
                    embedding_model=resolved_embed_model,
                    dimensions=dimensions,
                    top_k=top_k,
                    min_score=min_vector_score,
                    genai_client=genai_client,
                )
    except (DatabaseConfigError, VectorIndexNotReadyError, DatabaseError, EmbeddingError) as exc:
        if allow_tfidf_fallback and fallback_index is not None and fallback_index.chunk_count > 0:
            fallback_reason = str(exc)
        else:
            raise

    if fallback_reason is not None and fallback_index is not None:
        tfidf_result = answer_question_from_knowledge_base(
            question=cleaned_question,
            index=fallback_index,
            api_key=api_key,
            model=model,
            top_k=top_k,
            client=genai_client,
            env_path=env_path,
        )
        return SemanticRAGResult(
            question=tfidf_result.question,
            model=tfidf_result.model,
            answer=tfidf_result.answer,
            sources=tfidf_result.sources,
            has_evidence=tfidf_result.has_evidence,
            retrieval_mode="tfidf_fallback",
            fallback_reason=fallback_reason,
            embedding_model=resolved_embed_model,
            index_name=resolved_index_name,
        )

    grounded = generate_grounded_answer(
        question=cleaned_question,
        sources=sources or [],
        api_key=api_key,
        model=model,
        client=genai_client,
        env_path=env_path,
    )
    return SemanticRAGResult(
        question=grounded.question,
        model=grounded.model,
        answer=grounded.answer,
        sources=grounded.sources,
        has_evidence=grounded.has_evidence,
        retrieval_mode="vector",
        fallback_reason=None,
        embedding_model=resolved_embed_model,
        index_name=resolved_index_name,
    )
