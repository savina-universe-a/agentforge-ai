"""Mocked automated tests for Milestone 5 MongoDB Atlas persistence (no live DB calls)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from pymongo.errors import NetworkTimeout, OperationFailure, ServerSelectionTimeoutError

import agent
import database
import rag
from test_rag import _build_in_memory_pdf


class _InMemoryCursor:
    """Minimal sortable/limitable cursor stub for mocked PyMongo collections."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = list(docs)

    def sort(self, field: str, direction: int) -> _InMemoryCursor:
        self._docs.sort(key=lambda item: item.get(field, ""), reverse=(direction < 0))
        return self

    def limit(self, count: int) -> _InMemoryCursor:
        self._docs = self._docs[:count]
        return self

    def __iter__(self):
        return iter(self._docs)


class _InMemoryCollection:
    """Minimal in-memory PyMongo collection stub supporting find_one, replace_one, insert_one, find."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    def find_one(
        self,
        query: dict[str, Any],
        projection: dict[str, int] | None = None,
    ) -> dict[str, Any] | None:
        doc_id = query.get("_id")
        doc = self.docs.get(doc_id)
        if doc is None:
            return None
        if not projection:
            return dict(doc)
        return {k: doc[k] for k in projection if k in doc}

    def replace_one(
        self,
        query: dict[str, Any],
        replacement: dict[str, Any],
        upsert: bool = False,
    ) -> None:
        doc_id = query["_id"]
        if doc_id in self.docs or upsert:
            self.docs[doc_id] = dict(replacement)

    def insert_one(self, document: dict[str, Any]) -> None:
        self.docs[document["_id"]] = dict(document)

    def find(self, query: dict[str, Any]) -> _InMemoryCursor:
        return _InMemoryCursor(list(self.docs.values()))


class _FakeMongoClient:
    """Minimal in-memory MongoClient stub for unit and AppTest tests."""

    def __init__(self) -> None:
        self.admin = MagicMock()
        self.admin.command.return_value = {"ok": 1}
        self._dbs: dict[str, dict[str, _InMemoryCollection]] = {}

    def __getitem__(self, db_name: str) -> dict[str, _InMemoryCollection]:
        if db_name not in self._dbs:
            self._dbs[db_name] = {
                database.DOCUMENTS_COLLECTION: _InMemoryCollection(),
                database.AGENT_RUNS_COLLECTION: _InMemoryCollection(),
            }
        return self._dbs[db_name]

    def close(self) -> None:
        return None


class TestDatabaseConfig(unittest.TestCase):
    """Verify .env configuration loading and validation without leaking credentials."""

    def test_reads_mongodb_uri_and_default_db_name_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text(
                "MONGODB_URI=mongodb+srv://user:pass@cluster0.example.mongodb.net/\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(database.is_mongodb_configured(env_path=env_file))
                self.assertEqual(
                    database.get_mongodb_db_name(env_path=env_file),
                    "agentforge_ai",
                )

    def test_missing_or_invalid_uri_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_env = Path(tmp_dir) / ".env"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(database.DatabaseConfigError):
                    database.MongoPersistence(env_path=missing_env)

        with self.assertRaises(database.DatabaseConfigError):
            database.MongoPersistence(uri="https://not-a-mongo-uri.example.com")


class TestDatabasePersistence(unittest.TestCase):
    """Verify document persistence, duplicate handling, reload parity, and agent run history."""

    def test_verify_connection_pings_admin_and_returns_db_name(self) -> None:
        fake_client = _FakeMongoClient()
        db = database.MongoPersistence(client=fake_client, db_name="agentforge_test")

        info = db.verify_connection()
        fake_client.admin.command.assert_called_once_with("ping")
        self.assertEqual(info, {"connected": True, "database": "agentforge_test"})

    def test_save_duplicate_update_and_reload_preserves_chunks_and_tfidf_retrieval(self) -> None:
        fake_client = _FakeMongoClient()
        db = database.MongoPersistence(client=fake_client, db_name="agentforge_ai")

        original_index = rag.KnowledgeBaseIndex()
        original_index.add_document(
            "refund_policy.txt",
            b"Refund Policy: Full refunds are available within 14 days of purchase.",
        )
        original_index.add_document(
            "release_guide.pdf",
            _build_in_memory_pdf(
                [
                    "Release Schedule: Production releases happen every Friday at 14:00 UTC.",
                    "Rollback Guide: Revert to the previous tag if smoke tests fail.",
                ]
            ),
        )

        # First save inserts both documents under stable deterministic IDs
        first_summary = db.save_knowledge_base(original_index)
        self.assertEqual(sorted(first_summary["inserted"]), ["refund_policy.txt", "release_guide.pdf"])
        self.assertEqual(first_summary["duplicates"], [])
        stored_docs = fake_client["agentforge_ai"][database.DOCUMENTS_COLLECTION].docs
        self.assertIn("doc:refund-policy-txt", stored_docs)
        self.assertIn("doc:release-guide-pdf", stored_docs)

        # Second save with identical content detects duplicates cleanly
        second_summary = db.save_knowledge_base(original_index)
        self.assertEqual(second_summary["inserted"], [])
        self.assertEqual(sorted(second_summary["duplicates"]), ["refund_policy.txt", "release_guide.pdf"])

        # Updating a document replaces under the same stable _id
        original_index.add_document(
            "refund_policy.txt",
            b"Refund Policy: Full refunds are now available within 30 days of purchase.",
        )
        third_summary = db.save_knowledge_base(original_index)
        self.assertEqual(third_summary["updated"], ["refund_policy.txt"])
        self.assertEqual(third_summary["duplicates"], ["release_guide.pdf"])
        self.assertEqual(len(stored_docs), 2)

        # Reload into a brand-new KnowledgeBaseIndex and verify exact chunk IDs & TF-IDF parity
        expected_refund_hits = original_index.retrieve("How many days for a refund?", top_k=2)
        expected_pdf_hits = original_index.retrieve("When do production releases happen?", top_k=2)

        reloaded_index = rag.KnowledgeBaseIndex()
        loaded_count = reloaded_index.load_persisted_documents(db.load_documents())
        self.assertEqual(loaded_count, 2)
        self.assertEqual(
            sorted(c.chunk_id for c in reloaded_index.chunks),
            sorted(c.chunk_id for c in original_index.chunks),
        )
        self.assertEqual(
            sorted(c.citation_label for c in reloaded_index.chunks),
            sorted(c.citation_label for c in original_index.chunks),
        )

        reloaded_refund_hits = reloaded_index.retrieve("How many days for a refund?", top_k=2)
        reloaded_pdf_hits = reloaded_index.retrieve("When do production releases happen?", top_k=2)
        self.assertEqual(reloaded_refund_hits[0].chunk.chunk_id, expected_refund_hits[0].chunk.chunk_id)
        self.assertEqual(reloaded_refund_hits[0].score, expected_refund_hits[0].score)
        self.assertEqual(reloaded_pdf_hits[0].chunk.page_number, 1)
        self.assertEqual(reloaded_pdf_hits[0].chunk.citation_label, expected_pdf_hits[0].chunk.citation_label)

    def test_save_and_load_agent_runs_redacts_accidental_secrets(self) -> None:
        fake_client = _FakeMongoClient()
        db = database.MongoPersistence(client=fake_client, db_name="agentforge_ai")

        run_result = agent.AgentRunResult(
            goal="Calculate workshop cost",
            model="gemini-3.5-flash-lite",
            final_response=(
                "The total cost is 323. (Ignore leaked uri mongodb+srv://u:secretpass@cluster.mongodb.net/)"
            ),
            iterations_used=1,
            max_iterations=5,
            stopped_reason="completed",
            activities=[
                agent.ToolActivityEntry(
                    iteration=1,
                    tool_name="calculate",
                    input_summary="expression='6 * 48 + 35'",
                    status="success",
                    result_summary="6 * 48 + 35 = 323",
                )
            ],
        )

        run_id = db.save_agent_run(run_result)
        self.assertTrue(run_id.startswith("run:"))

        loaded_runs = db.load_agent_runs(limit=10)
        self.assertEqual(len(loaded_runs), 1)
        self.assertEqual(loaded_runs[0].goal, "Calculate workshop cost")
        self.assertEqual(loaded_runs[0].activities[0].tool_name, "calculate")
        self.assertNotIn("secretpass", loaded_runs[0].final_response)
        self.assertIn("[REDACTED]@", loaded_runs[0].final_response)

    def test_connection_timeout_and_auth_errors_never_expose_credentials(self) -> None:
        secret_uri = "mongodb+srv://devuser:TopSecretPass99@cluster0.example.mongodb.net/"

        timeout_client = MagicMock()
        timeout_client.admin.command.side_effect = ServerSelectionTimeoutError(
            f"Timed out connecting to {secret_uri}"
        )
        with patch("database.MongoClient", return_value=timeout_client):
            db = database.MongoPersistence(uri=secret_uri, timeout_ms=2000)
            with self.assertRaises(database.DatabaseError) as ctx:
                db.verify_connection()
        self.assertIn("timeout", str(ctx.exception).lower())
        self.assertNotIn("TopSecretPass99", str(ctx.exception))

        auth_client = MagicMock()
        auth_client.admin.command.side_effect = OperationFailure(
            f"Authentication failed for {secret_uri}",
            code=8000,
        )
        with patch("database.MongoClient", return_value=auth_client):
            db = database.MongoPersistence(uri=secret_uri, timeout_ms=2000)
            with self.assertRaises(database.DatabaseError) as auth_ctx:
                db.verify_connection()
        self.assertIn("authentication failed", str(auth_ctx.exception).lower())
        self.assertNotIn("TopSecretPass99", str(auth_ctx.exception))

        net_timeout_client = _FakeMongoClient()
        net_timeout_client["agentforge_ai"][database.DOCUMENTS_COLLECTION].find = MagicMock(
            side_effect=NetworkTimeout("socket timeout")
        )
        db = database.MongoPersistence(client=net_timeout_client)
        with self.assertRaises(database.DatabaseError) as net_ctx:
            db.load_documents()
        self.assertIn("timed out", str(net_ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
