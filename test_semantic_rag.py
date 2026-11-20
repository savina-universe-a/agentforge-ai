"""Mocked automated tests for Milestone 6 Gemini embeddings and Atlas Vector Search (no live calls)."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
from google.genai import errors as genai_errors

import planner
import rag
import semantic_rag
from test_agent import _make_text_response


def _unit_vec(dims: int, active_index: int, scale: float = 0.5) -> list[float]:
    """Return a non-normalized vector concentrated on active_index to verify L2 normalization."""
    vec = [0.0] * dims
    vec[active_index % dims] = scale
    return vec


class _FakeVectorCollection:
    """In-memory MongoDB collection stub supporting CRUD, list_search_indexes, and $vectorSearch."""

    def __init__(self, search_indexes: list[dict[str, Any]] | None = None) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.search_indexes: list[dict[str, Any]] = list(search_indexes or [])
        self.last_pipeline: list[dict[str, Any]] | None = None

    def count_documents(self, query: dict[str, Any]) -> int:
        return len(self.docs)

    def list_search_indexes(self, name: str | None = None) -> list[dict[str, Any]]:
        if name is None:
            return list(self.search_indexes)
        return [idx for idx in self.search_indexes if idx.get("name") == name]

    def find(
        self,
        query: dict[str, Any],
        projection: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        doc_id = query.get("doc_id")
        matched = [
            doc for doc in self.docs.values() if doc_id is None or doc.get("doc_id") == doc_id
        ]
        if not projection:
            return [dict(d) for d in matched]
        return [{k: d[k] for k in projection if k in d} for d in matched]

    def replace_one(
        self,
        query: dict[str, Any],
        replacement: dict[str, Any],
        upsert: bool = False,
    ) -> None:
        key = query["_id"]
        if key in self.docs or upsert:
            self.docs[key] = dict(replacement)

    def delete_many(self, query: dict[str, Any]) -> SimpleNamespace:
        doc_id = query.get("doc_id")
        id_in = set((query.get("_id") or {}).get("$in", []))
        removed = 0
        for key in list(self.docs.keys()):
            doc = self.docs[key]
            if (doc_id is None or doc.get("doc_id") == doc_id) and (not id_in or key in id_in):
                del self.docs[key]
                removed += 1
        return SimpleNamespace(deleted_count=removed)

    def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.last_pipeline = pipeline
        vs = pipeline[0]["$vectorSearch"]
        query_vec = vs["queryVector"]
        limit = int(vs["limit"])

        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self.docs.values():
            emb = doc.get("embedding", [])
            if len(emb) != len(query_vec):
                continue
            dot = sum(a * b for a, b in zip(query_vec, emb, strict=True))
            # MongoDB Atlas cosine similarity score maps [-1, 1] -> [0, 1] via (1 + cos) / 2
            score = (1.0 + dot) / 2.0
            scored.append((score, doc))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        output: list[dict[str, Any]] = []
        for score, doc in scored[:limit]:
            output.append(
                {
                    "chunk_id": doc["chunk_id"],
                    "doc_name": doc["doc_name"],
                    "page_number": doc.get("page_number"),
                    "chunk_index": doc["chunk_index"],
                    "citation_label": doc["citation_label"],
                    "text": doc["text"],
                    "score": score,
                }
            )
        return output


class _FakeVectorMongoClient:
    """In-memory MongoClient stub exposing document_chunks collection."""

    def __init__(self, search_indexes: list[dict[str, Any]] | None = None) -> None:
        self.collection = _FakeVectorCollection(search_indexes=search_indexes)

    def __getitem__(self, db_name: str) -> dict[str, _FakeVectorCollection]:
        return {semantic_rag.CHUNKS_COLLECTION: self.collection}

    def close(self) -> None:
        return None


class TestGeminiEmbeddings(unittest.TestCase):
    """Verify Gemini embedding batching, task types, dimensions, normalization, and error handling."""

    def test_embed_texts_batches_uses_task_type_and_l2_normalizes(self) -> None:
        mock_client = MagicMock()
        dims = 8

        def fake_embed_content(*, model: str, contents: list[str], config: Any) -> SimpleNamespace:
            self.assertEqual(model, "gemini-embedding-001")
            self.assertEqual(config.task_type, "RETRIEVAL_DOCUMENT")
            self.assertEqual(config.output_dimensionality, dims)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=_unit_vec(dims, idx, scale=0.5)) for idx in range(len(contents))]
            )

        mock_client.models.embed_content.side_effect = fake_embed_content

        vectors = semantic_rag.embed_texts(
            texts=["a", "b", "c", "d", "e"],
            task_type="RETRIEVAL_DOCUMENT",
            model="gemini-embedding-001",
            dimensions=dims,
            batch_size=2,
            client=mock_client,
        )

        # 5 texts with batch_size=2 -> 3 batched API calls (2, 2, 1)
        self.assertEqual(mock_client.models.embed_content.call_count, 3)
        self.assertEqual(len(vectors), 5)
        for vec in vectors:
            self.assertEqual(len(vec), dims)
            norm = math.sqrt(sum(v * v for v in vec))
            self.assertAlmostEqual(norm, 1.0, places=6)

    def test_embed_texts_rejects_dimension_mismatch_and_zero_vector(self) -> None:
        bad_dims_client = MagicMock()
        bad_dims_client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])]
        )
        with self.assertRaises(semantic_rag.EmbeddingError) as ctx:
            semantic_rag.embed_texts(
                texts=["hello"],
                dimensions=8,
                client=bad_dims_client,
            )
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

        zero_client = MagicMock()
        zero_client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.0] * 8)]
        )
        with self.assertRaises(semantic_rag.EmbeddingError):
            semantic_rag.embed_texts(texts=["hello"], dimensions=8, client=zero_client)

    def test_embed_texts_handles_quota_timeout_and_redacts_secret(self) -> None:
        secret = "super-secret-embed-key-123"
        quota_client = MagicMock()
        quota_client.models.embed_content.side_effect = genai_errors.APIError(
            429,
            {"error": {"message": f"Quota exceeded for {secret}"}},
        )

        with patch("semantic_rag.genai.Client", return_value=quota_client):
            with self.assertRaises(semantic_rag.EmbeddingError) as ctx:
                semantic_rag.embed_texts(texts=["hello"], api_key=secret, dimensions=8)

        self.assertIn("429", str(ctx.exception))
        self.assertNotIn(secret, str(ctx.exception))

        timeout_client = MagicMock()
        timeout_client.models.embed_content.side_effect = httpx.ReadTimeout("timeout")
        with self.assertRaises(semantic_rag.EmbeddingError) as t_ctx:
            semantic_rag.embed_texts(texts=["hello"], client=timeout_client, dimensions=8)
        self.assertIn("timed out", str(t_ctx.exception).lower())

    def test_vector_index_definition_matches_768_dimensions(self) -> None:
        definition = semantic_rag.get_vector_index_definition(768)
        vector_field = definition["fields"][0]
        self.assertEqual(vector_field["type"], "vector")
        self.assertEqual(vector_field["path"], "embedding")
        self.assertEqual(vector_field["numDimensions"], 768)
        self.assertEqual(vector_field["similarity"], "cosine")


class TestSemanticIndexingAndVectorRetrieval(unittest.TestCase):
    """Verify idempotent indexing, stale chunk deletion, vector retrieval, citations, and fallback."""

    def test_idempotent_indexing_skips_unchanged_and_deletes_stale_chunks(self) -> None:
        fake_mongo = _FakeVectorMongoClient()
        store = semantic_rag.SemanticVectorStore(client=fake_mongo, db_name="agentforge_ai")

        dims = 8
        mock_genai = MagicMock()

        def fake_embed(*, model: str, contents: list[str], config: Any) -> SimpleNamespace:
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=_unit_vec(dims, i + 1)) for i in range(len(contents))]
            )

        mock_genai.models.embed_content.side_effect = fake_embed

        record_v1 = rag.DocumentRecord(
            doc_name="policy.txt",
            file_type="txt",
            size_bytes=200,
            page_count=1,
            chunk_count=3,
        )
        chunks_v1 = [
            rag.DocumentChunk(
                chunk_id=f"policy-txt:c{i}",
                doc_name="policy.txt",
                page_number=None,
                chunk_index=i,
                citation_label=f"policy.txt | Chunk {i}",
                text=f"Policy section {i} content.",
            )
            for i in (1, 2, 3)
        ]

        # First run embeds all 3 chunks
        stats_1 = store.index_document_chunks(
            record_v1, chunks_v1, dimensions=dims, genai_client=mock_genai
        )
        self.assertEqual(stats_1["embedded"], 3)
        self.assertEqual(stats_1["skipped"], 0)
        self.assertEqual(stats_1["deleted_stale"], 0)
        self.assertEqual(mock_genai.models.embed_content.call_count, 1)
        self.assertEqual(store.count_indexed_chunks(), 3)

        # Second run with identical chunks skips all 3 without calling Gemini again
        stats_2 = store.index_document_chunks(
            record_v1, chunks_v1, dimensions=dims, genai_client=mock_genai
        )
        self.assertEqual(stats_2["embedded"], 0)
        self.assertEqual(stats_2["skipped"], 3)
        self.assertEqual(stats_2["deleted_stale"], 0)
        self.assertEqual(mock_genai.models.embed_content.call_count, 1)

        # Updated shorter document modifies chunk 1 and removes chunks 2 & 3
        record_v2 = rag.DocumentRecord(
            doc_name="policy.txt",
            file_type="txt",
            size_bytes=80,
            page_count=1,
            chunk_count=1,
        )
        chunks_v2 = [
            rag.DocumentChunk(
                chunk_id="policy-txt:c1",
                doc_name="policy.txt",
                page_number=None,
                chunk_index=1,
                citation_label="policy.txt | Chunk 1",
                text="Updated policy section 1 content.",
            )
        ]
        stats_3 = store.index_document_chunks(
            record_v2, chunks_v2, dimensions=dims, genai_client=mock_genai
        )
        self.assertEqual(stats_3["embedded"], 1)
        self.assertEqual(stats_3["skipped"], 0)
        self.assertEqual(stats_3["deleted_stale"], 2)
        self.assertEqual(store.count_indexed_chunks(), 1)
        self.assertEqual(list(fake_mongo.collection.docs.keys()), ["policy-txt:c1"])

    def test_vector_search_uses_retrieval_query_and_returns_grounded_citations(self) -> None:
        ready_index = [{"name": "chunk_embedding_index", "status": "READY", "queryable": True}]
        fake_mongo = _FakeVectorMongoClient(search_indexes=ready_index)
        store = semantic_rag.SemanticVectorStore(client=fake_mongo, db_name="agentforge_ai")

        dims = 8
        mock_genai = MagicMock()

        def fake_embed(*, model: str, contents: list[str], config: Any) -> SimpleNamespace:
            if config.task_type == "RETRIEVAL_DOCUMENT":
                # Chunk 1 -> axis 0 (refund), Chunk 2 -> axis 1 (deployment)
                return SimpleNamespace(
                    embeddings=[
                        SimpleNamespace(values=_unit_vec(dims, 0)),
                        SimpleNamespace(values=_unit_vec(dims, 1)),
                    ]
                )
            # Query about reimbursement -> axis 0
            self.assertEqual(config.task_type, "RETRIEVAL_QUERY")
            return SimpleNamespace(embeddings=[SimpleNamespace(values=_unit_vec(dims, 0))])

        mock_genai.models.embed_content.side_effect = fake_embed
        mock_genai.models.generate_content.return_value = _make_text_response(
            "Customers may request a full reimbursement within 14 days [billing.pdf | Page 2 | Chunk 1]."
        )

        record = rag.DocumentRecord(
            doc_name="billing.pdf",
            file_type="pdf",
            size_bytes=500,
            page_count=2,
            chunk_count=2,
        )
        chunks = [
            rag.DocumentChunk(
                chunk_id="billing-pdf:p2:c1",
                doc_name="billing.pdf",
                page_number=2,
                chunk_index=1,
                citation_label="billing.pdf | Page 2 | Chunk 1",
                text="Full reimbursement is guaranteed within 14 calendar days.",
            ),
            rag.DocumentChunk(
                chunk_id="billing-pdf:p1:c1",
                doc_name="billing.pdf",
                page_number=1,
                chunk_index=1,
                citation_label="billing.pdf | Page 1 | Chunk 1",
                text="Production deployments occur every Tuesday morning.",
            ),
        ]
        store.index_document_chunks(record, chunks, dimensions=dims, genai_client=mock_genai)

        result = semantic_rag.answer_question_semantic(
            question="Can I get my money back?",
            dimensions=dims,
            vector_store=store,
            genai_client=mock_genai,
        )

        self.assertEqual(result.retrieval_mode, "vector")
        self.assertIsNone(result.fallback_reason)
        self.assertTrue(result.has_evidence)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].chunk.citation_label, "billing.pdf | Page 2 | Chunk 1")
        self.assertIn("billing.pdf | Page 2 | Chunk 1", result.answer)

    def test_vector_index_not_ready_raises_without_fallback_and_reports_explicitly_with_fallback(self) -> None:
        building_index = [{"name": "chunk_embedding_index", "status": "BUILDING", "queryable": False}]
        fake_mongo = _FakeVectorMongoClient(search_indexes=building_index)
        store = semantic_rag.SemanticVectorStore(client=fake_mongo, db_name="agentforge_ai")

        # Without fallback -> raises VectorIndexNotReadyError
        with self.assertRaises(semantic_rag.VectorIndexNotReadyError) as ctx:
            semantic_rag.answer_question_semantic(
                question="What is the SLA?",
                allow_tfidf_fallback=False,
                vector_store=store,
            )
        self.assertIn("BUILDING", str(ctx.exception))

        # With fallback enabled and local TF-IDF chunks -> falls back EXPLICITLY (never silently!)
        local_index = rag.KnowledgeBaseIndex()
        local_index.add_document(
            "sla.txt",
            b"Support SLA: Critical incidents receive a response within 2 hours.",
        )

        mock_genai = MagicMock()
        mock_genai.models.generate_content.return_value = _make_text_response(
            "Critical incidents receive a response within 2 hours [sla.txt | Chunk 1]."
        )

        fallback_result = semantic_rag.answer_question_semantic(
            question="What is the support SLA?",
            fallback_index=local_index,
            allow_tfidf_fallback=True,
            vector_store=store,
            genai_client=mock_genai,
        )

        self.assertEqual(fallback_result.retrieval_mode, "tfidf_fallback")
        self.assertIsNotNone(fallback_result.fallback_reason)
        self.assertIn("BUILDING", fallback_result.fallback_reason or "")
        self.assertTrue(fallback_result.has_evidence)
        self.assertEqual(fallback_result.sources[0].chunk.citation_label, "sla.txt | Chunk 1")

    def test_vector_search_returns_no_evidence_when_below_threshold(self) -> None:
        ready_index = [{"name": "chunk_embedding_index", "status": "READY", "queryable": True}]
        fake_mongo = _FakeVectorMongoClient(search_indexes=ready_index)
        store = semantic_rag.SemanticVectorStore(client=fake_mongo, db_name="agentforge_ai")

        dims = 8
        mock_genai = MagicMock()
        mock_genai.models.embed_content.side_effect = [
            SimpleNamespace(embeddings=[SimpleNamespace(values=_unit_vec(dims, 0))]),
            SimpleNamespace(embeddings=[SimpleNamespace(values=_unit_vec(dims, 3))]),  # orthogonal -> score 0.50
        ]

        record = rag.DocumentRecord(
            doc_name="notes.txt", file_type="txt", size_bytes=50, page_count=1, chunk_count=1
        )
        chunks = [
            rag.DocumentChunk(
                chunk_id="notes-txt:c1",
                doc_name="notes.txt",
                page_number=None,
                chunk_index=1,
                citation_label="notes.txt | Chunk 1",
                text="Python setup instructions.",
            )
        ]
        store.index_document_chunks(record, chunks, dimensions=dims, genai_client=mock_genai)

        result = semantic_rag.answer_question_semantic(
            question="Completely unrelated topic?",
            dimensions=dims,
            min_vector_score=0.55,
            vector_store=store,
            genai_client=mock_genai,
        )

        self.assertEqual(result.retrieval_mode, "vector")
        self.assertFalse(result.has_evidence)
        self.assertEqual(result.sources, [])
        mock_genai.models.generate_content.assert_not_called()

    def test_missing_env_defaults_embedding_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_env = Path(tmp_dir) / ".env"
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    semantic_rag.get_embedding_model_name(env_path=missing_env),
                    "gemini-embedding-001",
                )
                self.assertEqual(
                    semantic_rag.get_embedding_dimensions(env_path=missing_env),
                    768,
                )
                self.assertEqual(
                    semantic_rag.get_vector_index_name(env_path=missing_env),
                    "chunk_embedding_index",
                )


if __name__ == "__main__":
    unittest.main()
