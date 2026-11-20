"""MongoDB Atlas persistence layer for AgentForge AI (Milestone 5)."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Literal

from dotenv import dotenv_values
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import (
    ConfigurationError,
    ConnectionFailure,
    DuplicateKeyError,
    NetworkTimeout,
    OperationFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)

from agent import AgentRunResult, ToolActivityEntry
import planner
from rag import (
    DocumentChunk,
    DocumentRecord,
    ExtractedPage,
    KnowledgeBaseIndex,
    _slugify_doc_name,
)

DEFAULT_DB_NAME = "agentforge_ai"
DOCUMENTS_COLLECTION = "documents"
AGENT_RUNS_COLLECTION = "agent_runs"
WORKFLOW_RUNS_COLLECTION = "workflow_runs"
DEFAULT_TIMEOUT_MS = 5_000

_MONGO_CREDENTIAL_PATTERN = re.compile(r"(mongodb(?:\+srv)?://)([^@\s/]+)@", re.IGNORECASE)
_GEMINI_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z\-_]{20,}")


class DatabaseConfigError(ValueError):
    """Raised when MongoDB configuration is missing or invalid."""


class DatabaseError(RuntimeError):
    """Raised when a MongoDB operation or connection fails."""


def _redact_sensitive_text(text: str, uri: str | None = None) -> str:
    """Strip MongoDB credentials and API key patterns from any text or error string."""
    cleaned = text
    if uri and uri in cleaned:
        cleaned = cleaned.replace(uri, "[REDACTED_MONGODB_URI]")
    cleaned = _MONGO_CREDENTIAL_PATTERN.sub(r"\1[REDACTED]@", cleaned)
    cleaned = _GEMINI_KEY_PATTERN.sub("[REDACTED_API_KEY]", cleaned)
    return cleaned


def _format_mongo_error(exc: Exception, uri: str | None = None) -> str:
    """Return a clear, user-friendly MongoDB error message without exposing credentials."""
    if isinstance(exc, ServerSelectionTimeoutError):
        return (
            "Could not reach MongoDB Atlas before the connection timeout. "
            "Check your network connection, Atlas IP Access List, and cluster status."
        )
    if isinstance(exc, NetworkTimeout):
        return "MongoDB Atlas operation timed out. Please retry shortly."
    if isinstance(exc, OperationFailure):
        code = getattr(exc, "code", None)
        if code in (18, 8000):
            return (
                "MongoDB Atlas authentication failed. Check the database username, "
                "password, and permissions in MONGODB_URI."
            )
        safe = _redact_sensitive_text(str(exc), uri)
        return f"MongoDB operation failed: {safe}"
    if isinstance(exc, ConfigurationError):
        safe = _redact_sensitive_text(str(exc), uri)
        return f"Invalid MongoDB configuration: {safe}"
    if isinstance(exc, ConnectionFailure):
        return "Unable to connect to MongoDB Atlas. Verify network access and cluster availability."

    safe = _redact_sensitive_text(str(exc), uri)
    return f"MongoDB error: {safe}"


def _read_dotenv_var(key: str, env_path: Path | None = None) -> str:
    """Read a single key from os.environ or local .env without mutating os.environ."""
    env_val = os.getenv(key, "").strip()
    if env_val:
        return env_val

    target_env = env_path if env_path is not None else planner.ENV_FILE
    if target_env.exists():
        file_val = (dotenv_values(target_env).get(key) or "").strip()
        if file_val:
            return file_val
    return ""


def get_mongodb_uri(
    override_uri: str | None = None,
    env_path: Path | None = None,
) -> str | None:
    """Return MONGODB_URI from an explicit override, environment, or local .env file."""
    if override_uri and override_uri.strip():
        return override_uri.strip()

    uri = _read_dotenv_var("MONGODB_URI", env_path=env_path)
    return uri or None


def get_mongodb_db_name(
    override_db: str | None = None,
    env_path: Path | None = None,
) -> str:
    """Return MONGODB_DB_NAME from override, environment, or local .env (default: agentforge_ai)."""
    if override_db and override_db.strip():
        return override_db.strip()

    db_name = _read_dotenv_var("MONGODB_DB_NAME", env_path=env_path)
    return db_name or DEFAULT_DB_NAME


def is_mongodb_configured(
    override_uri: str | None = None,
    env_path: Path | None = None,
) -> bool:
    """Return True when MONGODB_URI is set (without validating network connectivity)."""
    return bool(get_mongodb_uri(override_uri=override_uri, env_path=env_path))


def compute_document_id(doc_name: str) -> str:
    """Return a stable, deterministic MongoDB _id for an uploaded document name."""
    slug = _slugify_doc_name(doc_name)
    return f"doc:{slug}"


def compute_content_sha256(
    pages: list[ExtractedPage],
    chunks: list[DocumentChunk],
) -> str:
    """Compute a deterministic SHA-256 fingerprint of extracted text and chunk identifiers."""
    hasher = hashlib.sha256()
    for page in pages:
        hasher.update(f"p:{page.page_number}\n{page.text}\n".encode("utf-8"))
    for chunk in chunks:
        hasher.update(f"c:{chunk.chunk_id}\n{chunk.text}\n".encode("utf-8"))
    return hasher.hexdigest()


class MongoPersistence:
    """Small, safe MongoDB Atlas persistence wrapper for documents and agent activity."""

    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        client: MongoClient[Any] | None = None,
        env_path: Path | None = None,
    ) -> None:
        self._env_path = env_path
        self._timeout_ms = max(int(timeout_ms), 500)
        self.db_name = get_mongodb_db_name(override_db=db_name, env_path=env_path)
        self._uri: str | None = None
        self._owns_client = client is None

        if client is not None:
            self._client: MongoClient[Any] = client
        else:
            resolved_uri = get_mongodb_uri(override_uri=uri, env_path=env_path)
            if not resolved_uri:
                raise DatabaseConfigError(
                    "MONGODB_URI is not configured. Add MONGODB_URI to your local .env file "
                    "to enable MongoDB Atlas persistence."
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
        self._documents = self._db[DOCUMENTS_COLLECTION]
        self._agent_runs = self._db[AGENT_RUNS_COLLECTION]
        self._workflow_runs = self._db[WORKFLOW_RUNS_COLLECTION]

    def close(self) -> None:
        """Close the MongoClient if owned by this persistence instance."""
        if self._owns_client and self._client is not None:
            self._client.close()

    def __enter__(self) -> MongoPersistence:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def verify_connection(self) -> dict[str, Any]:
        """Verify MongoDB Atlas connectivity with a lightweight ping command."""
        try:
            self._client.admin.command("ping")
            return {
                "connected": True,
                "database": self.db_name,
            }
        except (PyMongoError, TimeoutError) as exc:
            raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc

    def save_document(
        self,
        record: DocumentRecord,
        chunks: list[DocumentChunk],
        pages: list[ExtractedPage] | None = None,
    ) -> dict[str, Any]:
        """Persist document metadata, extracted text, and exact chunks with duplicate detection."""
        if not chunks:
            raise ValueError(f"Cannot persist '{record.doc_name}' without indexed chunks.")

        doc_id = compute_document_id(record.doc_name)
        extracted_pages = list(pages) if pages else [
            ExtractedPage(page_number=chunk.page_number, text=chunk.text)
            for chunk in chunks
        ]
        content_hash = compute_content_sha256(extracted_pages, chunks)
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

        try:
            existing = self._documents.find_one(
                {"_id": doc_id},
                {"content_sha256": 1, "chunk_count": 1, "created_at": 1},
            )
            if (
                existing
                and existing.get("content_sha256") == content_hash
                and existing.get("chunk_count") == record.chunk_count
            ):
                return {
                    "doc_id": doc_id,
                    "doc_name": record.doc_name,
                    "status": "duplicate",
                }

            payload: dict[str, Any] = {
                "_id": doc_id,
                "doc_name": record.doc_name,
                "file_type": record.file_type,
                "size_bytes": record.size_bytes,
                "page_count": record.page_count,
                "chunk_count": record.chunk_count,
                "content_sha256": content_hash,
                "pages": [
                    {
                        "page_number": p.page_number,
                        "text": _redact_sensitive_text(p.text, self._uri),
                    }
                    for p in extracted_pages
                ],
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_name": c.doc_name,
                        "page_number": c.page_number,
                        "chunk_index": c.chunk_index,
                        "citation_label": c.citation_label,
                        "text": _redact_sensitive_text(c.text, self._uri),
                    }
                    for c in chunks
                ],
                "created_at": (existing or {}).get("created_at", now_iso),
                "updated_at": now_iso,
            }

            self._documents.replace_one({"_id": doc_id}, payload, upsert=True)
            status: Literal["inserted", "updated"] = "updated" if existing else "inserted"
            return {
                "doc_id": doc_id,
                "doc_name": record.doc_name,
                "status": status,
            }
        except DuplicateKeyError:
            return {
                "doc_id": doc_id,
                "doc_name": record.doc_name,
                "status": "duplicate",
            }
        except (PyMongoError, TimeoutError) as exc:
            raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc

    def save_knowledge_base(self, index: KnowledgeBaseIndex) -> dict[str, list[str]]:
        """Persist all documents currently indexed in a KnowledgeBaseIndex."""
        summary: dict[str, list[str]] = {
            "inserted": [],
            "updated": [],
            "duplicates": [],
        }
        for doc in index.documents:
            record, chunks, pages = index.get_document_bundle(doc.doc_name)
            outcome = self.save_document(record=record, chunks=chunks, pages=pages)
            if outcome["status"] == "inserted":
                summary["inserted"].append(doc.doc_name)
            elif outcome["status"] == "updated":
                summary["updated"].append(doc.doc_name)
            else:
                summary["duplicates"].append(doc.doc_name)
        return summary

    def load_documents(
        self,
    ) -> list[tuple[DocumentRecord, list[DocumentChunk], list[ExtractedPage]]]:
        """Load persisted documents, extracted pages, and exact chunk identifiers from MongoDB."""
        bundles: list[tuple[DocumentRecord, list[DocumentChunk], list[ExtractedPage]]] = []
        try:
            cursor = self._documents.find({}).sort("doc_name", ASCENDING)
            for raw in cursor:
                record = DocumentRecord(
                    doc_name=raw["doc_name"],
                    file_type=raw["file_type"],
                    size_bytes=int(raw["size_bytes"]),
                    page_count=int(raw["page_count"]),
                    chunk_count=int(raw["chunk_count"]),
                )
                chunks = [
                    DocumentChunk(
                        chunk_id=c["chunk_id"],
                        doc_name=c["doc_name"],
                        page_number=c.get("page_number"),
                        chunk_index=int(c["chunk_index"]),
                        citation_label=c["citation_label"],
                        text=c["text"],
                    )
                    for c in raw.get("chunks", [])
                ]
                pages = [
                    ExtractedPage(
                        page_number=p.get("page_number"),
                        text=p["text"],
                    )
                    for p in raw.get("pages", [])
                ]
                if chunks:
                    bundles.append((record, chunks, pages))
        except (PyMongoError, TimeoutError) as exc:
            raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise DatabaseError("Persisted document record in MongoDB has an unexpected format.") from exc

        return bundles

    def save_agent_run(self, result: AgentRunResult) -> str:
        """Persist a completed autonomous agent summary and safe tool activity log."""
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat(timespec="seconds")
        digest = hashlib.sha256(
            f"{now_utc.isoformat()}|{result.goal}|{result.iterations_used}|{len(result.activities)}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        run_id = f"run:{now_utc.strftime('%Y%m%dT%H%M%SZ')}:{digest}"

        safe_activities = [
            {
                "iteration": entry.iteration,
                "tool_name": entry.tool_name,
                "input_summary": _redact_sensitive_text(entry.input_summary, self._uri),
                "status": entry.status,
                "result_summary": _redact_sensitive_text(entry.result_summary, self._uri),
            }
            for entry in result.activities
        ]

        payload: dict[str, Any] = {
            "_id": run_id,
            "goal": _redact_sensitive_text(result.goal, self._uri),
            "model": result.model,
            "final_response": _redact_sensitive_text(result.final_response, self._uri),
            "iterations_used": result.iterations_used,
            "max_iterations": result.max_iterations,
            "stopped_reason": result.stopped_reason,
            "activities": safe_activities,
            "created_at": now_iso,
        }

        try:
            self._agent_runs.insert_one(payload)
            return run_id
        except (PyMongoError, TimeoutError) as exc:
            raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc

    def load_agent_runs(self, limit: int = 20) -> list[AgentRunResult]:
        """Load persisted autonomous agent runs and activity logs (newest first)."""
        bounded_limit = max(1, min(int(limit), 100))
        runs: list[AgentRunResult] = []
        try:
            cursor = self._agent_runs.find({}).sort("created_at", DESCENDING).limit(bounded_limit)
            for raw in cursor:
                activities = [
                    ToolActivityEntry(
                        iteration=int(item["iteration"]),
                        tool_name=item["tool_name"],
                        input_summary=item["input_summary"],
                        status=item["status"],
                        result_summary=item["result_summary"],
                    )
                    for item in raw.get("activities", [])
                ]
                runs.append(
                    AgentRunResult(
                        goal=raw["goal"],
                        model=raw["model"],
                        final_response=raw["final_response"],
                        activities=activities,
                        iterations_used=int(raw["iterations_used"]),
                        max_iterations=int(raw["max_iterations"]),
                        stopped_reason=raw.get("stopped_reason", "completed"),
                    )
                )
        except (PyMongoError, TimeoutError) as exc:
            raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise DatabaseError("Persisted agent run record in MongoDB has an unexpected format.") from exc

        return runs

    def save_workflow_run(self, workflow_payload: dict[str, Any]) -> str:
        """Persist a completed multi-agent LangGraph workflow trace and response safely."""
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat(timespec="seconds")
        task_text = str(workflow_payload.get("task", ""))
        route_text = str(workflow_payload.get("route", ""))
        trace_items = list(workflow_payload.get("trace", []))

        digest = hashlib.sha256(
            f"{now_utc.isoformat()}|{task_text}|{route_text}|{len(trace_items)}".encode("utf-8")
        ).hexdigest()[:16]
        workflow_id = f"wf:{now_utc.strftime('%Y%m%dT%H%M%SZ')}:{digest}"

        safe_trace = [
            {
                "step": int(item.get("step", idx)),
                "agent": str(item.get("agent", "unknown")),
                "action": _redact_sensitive_text(str(item.get("action", "")), self._uri),
                "status": str(item.get("status", "ok")),
                "detail": _redact_sensitive_text(str(item.get("detail", "")), self._uri),
            }
            for idx, item in enumerate(trace_items, start=1)
        ]

        safe_doc: dict[str, Any] = {
            "_id": workflow_id,
            "task": _redact_sensitive_text(task_text, self._uri),
            "model": str(workflow_payload.get("model", "")),
            "route": route_text,
            "retrieval_mode": str(workflow_payload.get("retrieval_mode", "not_needed")),
            "fallback_notes": [
                _redact_sensitive_text(str(note), self._uri)
                for note in workflow_payload.get("fallback_notes", [])
            ],
            "citations": [
                _redact_sensitive_text(str(c), self._uri)
                for c in workflow_payload.get("citations", [])
            ],
            "final_response": _redact_sensitive_text(
                str(workflow_payload.get("final_response", "")), self._uri
            ),
            "steps_taken": int(workflow_payload.get("steps_taken", 0)),
            "max_steps": int(workflow_payload.get("max_steps", 0)),
            "api_calls_used": int(workflow_payload.get("api_calls_used", 0)),
            "max_api_calls": int(workflow_payload.get("max_api_calls", 0)),
            "stopped_reason": str(workflow_payload.get("stopped_reason", "completed")),
            "trace": safe_trace,
            "created_at": now_iso,
        }

        try:
            self._workflow_runs.insert_one(safe_doc)
            return workflow_id
        except (PyMongoError, TimeoutError) as exc:
            raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc

    def load_workflow_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Load persisted multi-agent workflow runs (newest first)."""
        bounded_limit = max(1, min(int(limit), 100))
        runs: list[dict[str, Any]] = []
        try:
            cursor = self._workflow_runs.find({}).sort("created_at", DESCENDING).limit(bounded_limit)
            for raw in cursor:
                runs.append(
                    {
                        "workflow_id": str(raw.get("_id", "")),
                        "task": str(raw["task"]),
                        "model": str(raw["model"]),
                        "route": str(raw.get("route", "direct_write")),
                        "retrieval_mode": str(raw.get("retrieval_mode", "not_needed")),
                        "fallback_notes": list(raw.get("fallback_notes", [])),
                        "citations": list(raw.get("citations", [])),
                        "final_response": str(raw["final_response"]),
                        "steps_taken": int(raw.get("steps_taken", 0)),
                        "max_steps": int(raw.get("max_steps", 0)),
                        "api_calls_used": int(raw.get("api_calls_used", 0)),
                        "max_api_calls": int(raw.get("max_api_calls", 0)),
                        "stopped_reason": str(raw.get("stopped_reason", "completed")),
                        "trace": list(raw.get("trace", [])),
                        "created_at": str(raw.get("created_at", "")),
                    }
                )
        except (PyMongoError, TimeoutError) as exc:
            raise DatabaseError(_format_mongo_error(exc, self._uri)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise DatabaseError("Persisted workflow run record in MongoDB has an unexpected format.") from exc

        return runs
