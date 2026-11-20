"""UI verification tests for AgentForge AI using Streamlit AppTest (mocked Gemini calls)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

import agent
import database
import planner
import rag
import semantic_rag
from test_agent import _make_function_call_response, _make_text_response
from test_database import _FakeMongoClient
from test_semantic_rag import _FakeVectorMongoClient, _unit_vec


MOCK_PLAN = planner.ActionPlan(
    summary="Execute the rollout in three focused steps.",
    steps=[
        planner.PlanStep(
            step_number=1,
            title="Finalize requirements",
            description="Confirm scope and acceptance checks before coding.",
            priority="High",
        ),
        planner.PlanStep(
            step_number=2,
            title="Implement core module",
            description="Build and unit-test the main planning service.",
            priority="Medium",
        ),
        planner.PlanStep(
            step_number=3,
            title="Update documentation",
            description="Add local setup and verification instructions.",
            priority="Low",
        ),
    ],
)


class TestAgentForgeApp(unittest.TestCase):
    """Verify dashboard rendering, navigation, and Milestone 2 AI planning flows."""

    def test_overview_renders_title_metrics_and_sidebar(self) -> None:
        at = AppTest.from_file("app.py").run()

        self.assertFalse(at.exception)
        self.assertEqual([t.value for t in at.title], ["AgentForge AI"])

        # Sidebar navigation items preserved
        self.assertEqual(len(at.sidebar.radio), 1)
        self.assertEqual(
            list(at.sidebar.radio[0].options),
            ["Overview", "AI Agent", "Knowledge Base", "Activity"],
        )

        # Overview metrics preserved
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(
            metrics,
            {
                "Agent Runs": "0",
                "Documents": "0",
                "Tasks Completed": "0",
            },
        )

        # Feature is labeled as AI planning, not autonomous execution
        subheaders = [s.value for s in at.subheader]
        self.assertIn("AI Planning", subheaders)

    def test_generate_plan_with_empty_goal_shows_warning(self) -> None:
        at = AppTest.from_file("app.py").run()

        at.button(key="generate_plan_btn").click().run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.warning), 1)
        self.assertIn("enter a goal", at.warning[0].value.lower())

    def test_generate_plan_with_missing_api_key_shows_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_env = Path(tmp_dir) / ".env"
            with (
                patch.object(planner, "ENV_FILE", missing_env),
                patch.dict(os.environ, {}, clear=True),
            ):
                at = AppTest.from_file("app.py").run()
                at.text_input(key="agent_goal").input("Build a release plan").run()
                at.button(key="generate_plan_btn").click().run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.warning), 1)
        self.assertIn("GEMINI_API_KEY", at.warning[0].value)

    def test_generate_plan_renders_mocked_gemini_steps_and_priorities(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = SimpleNamespace(
            parsed=MOCK_PLAN,
            text=MOCK_PLAN.model_dump_json(),
        )

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "mock-test-key"}, clear=False),
            patch("planner.genai.Client", return_value=mock_client),
        ):
            at = AppTest.from_file("app.py").run()
            at.text_input(key="agent_goal").input("Ship Milestone 2 cleanly").run()
            at.button(key="generate_plan_btn").click().run()

        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertFalse(at.warning)

        rendered_markdown = "\n".join(m.value for m in at.markdown)
        self.assertIn("Execute the rollout in three focused steps.", rendered_markdown)
        self.assertIn("Step 1: Finalize requirements", rendered_markdown)
        self.assertIn("High Priority", rendered_markdown)
        self.assertIn("Medium Priority", rendered_markdown)
        self.assertIn("Low Priority", rendered_markdown)

        # Metrics remain 0 (AI planning only, not autonomous execution)
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics["Agent Runs"], "0")
        self.assertEqual(metrics["Tasks Completed"], "0")

    def test_generate_plan_displays_api_error_cleanly(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Upstream timeout")

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "mock-test-key"}, clear=False),
            patch("planner.genai.Client", return_value=mock_client),
        ):
            at = AppTest.from_file("app.py").run()
            at.text_input(key="agent_goal").input("Ship Milestone 2 cleanly").run()
            at.button(key="generate_plan_btn").click().run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.error), 1)
        self.assertIn("Upstream timeout", at.error[0].value)

    def test_sidebar_private_session_key_enables_planning(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = SimpleNamespace(
            parsed=MOCK_PLAN,
            text=MOCK_PLAN.model_dump_json(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_env = Path(tmp_dir) / ".env"
            with (
                patch.object(planner, "ENV_FILE", missing_env),
                patch.dict(os.environ, {}, clear=True),
                patch("planner.genai.Client", return_value=mock_client) as mock_client_cls,
            ):
                at = AppTest.from_file("app.py").run()
                at.sidebar.text_input(key="sidebar_api_key_input").input("session-only-fake-key").run()
                at.sidebar.button(key="use_session_key_btn").click().run()
                at.text_input(key="agent_goal").input("Plan API integration").run()
                at.button(key="generate_plan_btn").click().run()

        self.assertFalse(at.exception)
        mock_client_cls.assert_called_once_with(api_key="session-only-fake-key", vertexai=False)
        rendered_markdown = "\n".join(m.value for m in at.markdown)
        self.assertIn("Step 1: Finalize requirements", rendered_markdown)

    def test_sidebar_placeholder_views_load_cleanly(self) -> None:
        at = AppTest.from_file("app.py").run()

        for section in ("AI Agent", "Knowledge Base", "Activity"):
            at.sidebar.radio[0].set_value(section).run()
            self.assertFalse(at.exception)
            self.assertEqual(at.subheader[0].value, section)
            self.assertEqual(len(at.info), 1)

    def test_run_autonomous_agent_with_empty_goal_shows_warning(self) -> None:
        at = AppTest.from_file("app.py").run()

        at.button(key="run_agent_btn").click().run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.warning), 1)
        self.assertIn("enter a goal", at.warning[0].value.lower())

    def test_run_autonomous_agent_renders_activity_log_and_final_response(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_function_call_response([("calculate", {"expression": "18 * 5"})]),
            _make_text_response(
                text="18 multiplied by 5 equals 90.",
                thought_text="HIDDEN REASONING SHOULD NOT APPEAR",
            ),
        ]

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "mock-secret-key-xyz"}, clear=False),
            patch("agent.genai.Client", return_value=mock_client),
            patch("database.is_mongodb_configured", return_value=False),
        ):
            at = AppTest.from_file("app.py").run()
            at.text_input(key="autonomous_goal").input("Compute 18 * 5").run()
            at.button(key="run_agent_btn").click().run()

        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertFalse(at.warning)

        rendered_markdown = "\n".join(m.value for m in at.markdown)
        self.assertIn("Activity Log", rendered_markdown)
        self.assertIn("calculate", rendered_markdown)
        self.assertIn("18 * 5 = 90", rendered_markdown)
        self.assertIn("18 multiplied by 5 equals 90.", rendered_markdown)
        self.assertNotIn("HIDDEN REASONING", rendered_markdown)
        self.assertNotIn("mock-secret-key-xyz", rendered_markdown)

        # Overview metrics reflect completed autonomous run in current session
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics["Agent Runs"], "1")
        self.assertEqual(metrics["Tasks Completed"], "1")

    def test_run_autonomous_agent_displays_error_gracefully(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Service unreachable")

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "mock-secret-key-xyz"}, clear=False),
            patch("agent.genai.Client", return_value=mock_client),
            patch("database.is_mongodb_configured", return_value=False),
        ):
            at = AppTest.from_file("app.py").run()
            at.text_input(key="autonomous_goal").input("Compute 18 * 5").run()
            at.button(key="run_agent_btn").click().run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.error), 1)
        self.assertIn("Service unreachable", at.error[0].value)
        self.assertNotIn("mock-secret-key-xyz", at.error[0].value)

    def test_knowledge_base_empty_query_warns_user(self) -> None:
        at = AppTest.from_file("app.py").run()
        at.sidebar.radio[0].set_value("Knowledge Base").run()
        at.text_input(key="kb_question").input("What is the refund policy?").run()
        at.button(key="kb_ask_btn").click().run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.warning), 1)
        self.assertIn("upload and index", at.warning[0].value.lower())

    def test_knowledge_base_grounded_answer_sources_and_clear_flow(self) -> None:
        kb_index = rag.KnowledgeBaseIndex()
        kb_index.add_document(
            "warranty.txt",
            b"Hardware Warranty: All devices include a 24-month limited hardware warranty.",
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_text_response(
            text="All devices include a 24-month limited hardware warranty [warranty.txt | Chunk 1]."
        )

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "mock-rag-key"}, clear=False),
            patch("rag.genai.Client", return_value=mock_client),
        ):
            at = AppTest.from_file("app.py").run()
            at.session_state["kb_index"] = kb_index
            at.run()

            # Overview Documents metric reflects in-memory Knowledge Base count
            metrics = {m.label: m.value for m in at.metric}
            self.assertEqual(metrics["Documents"], "1")

            # Switch to Knowledge Base and ask a grounded question
            at.sidebar.radio[0].set_value("Knowledge Base").run()
            at.text_input(key="kb_question").input("How long is the hardware warranty?").run()
            at.button(key="kb_ask_btn").click().run()

            self.assertFalse(at.exception)
            self.assertFalse(at.error)

            rendered_markdown = "\n".join(m.value for m in at.markdown)
            self.assertIn("24-month limited hardware warranty", rendered_markdown)
            self.assertIn("warranty.txt | Chunk 1", rendered_markdown)

            # Expandable Sources section contains the retrieved passage
            self.assertTrue(any("Sources" in exp.label for exp in at.expander))

            # Clear Knowledge Base removes documents from session state
            at.button(key="kb_clear_btn").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state["kb_index"].document_count, 0)

    def test_knowledge_base_missing_evidence_shows_explicit_notice(self) -> None:
        kb_index = rag.KnowledgeBaseIndex()
        kb_index.add_document(
            "warranty.txt",
            b"Hardware Warranty: All devices include a 24-month limited hardware warranty.",
        )

        at = AppTest.from_file("app.py").run()
        at.session_state["kb_index"] = kb_index
        at.sidebar.radio[0].set_value("Knowledge Base").run()
        at.text_input(key="kb_question").input("What is the cafeteria pizza price?").run()
        at.button(key="kb_ask_btn").click().run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.warning), 1)
        self.assertIn("do not contain", at.warning[0].value.lower())

    def test_knowledge_base_mongodb_save_clear_and_reload_flow(self) -> None:
        fake_mongo_client = _FakeMongoClient()

        kb_index = rag.KnowledgeBaseIndex()
        kb_index.add_document(
            "sla.txt",
            b"Support SLA: Priority 1 incidents receive an initial response within 2 hours.",
        )

        with (
            patch.dict(
                os.environ,
                {"MONGODB_URI": "mongodb+srv://user:pass@cluster0.example.mongodb.net/"},
                clear=False,
            ),
            patch("database.MongoClient", return_value=fake_mongo_client),
        ):
            at = AppTest.from_file("app.py").run()
            at.session_state["kb_index"] = kb_index
            at.sidebar.radio[0].set_value("Knowledge Base").run()

            # Save indexed document to mocked MongoDB Atlas
            at.button(key="kb_mongo_save_btn").click().run()
            self.assertFalse(at.exception)
            self.assertTrue(any("saved 1 new" in s.value.lower() for s in at.success))

            # Clear session Knowledge Base
            at.button(key="kb_clear_btn").click().run()
            self.assertEqual(at.session_state["kb_index"].document_count, 0)

            # Reload saved document from MongoDB Atlas into local TF-IDF index
            at.button(key="kb_mongo_load_btn").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state["kb_index"].document_count, 1)
            hits = at.session_state["kb_index"].retrieve("Priority 1 SLA response time")
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].chunk.citation_label, "sla.txt | Chunk 1")

    def test_knowledge_base_session_mode_handles_missing_mongodb_uri_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_env = Path(tmp_dir) / ".env"
            with (
                patch.object(planner, "ENV_FILE", missing_env),
                patch.dict(os.environ, {"MONGODB_URI": ""}, clear=False),
            ):
                at = AppTest.from_file("app.py").run()
                at.sidebar.radio[0].set_value("Knowledge Base").run()
                at.button(key="kb_mongo_verify_btn").click().run()

        self.assertFalse(at.exception)
        self.assertTrue(any("MONGODB_URI" in w.value for w in at.warning))

    def test_autonomous_agent_persists_run_and_activity_view_reloads_it(self) -> None:
        fake_mongo_client = _FakeMongoClient()

        mock_gemini = MagicMock()
        mock_gemini.models.generate_content.side_effect = [
            _make_function_call_response([("calculate", {"expression": "10 * 7"})]),
            _make_text_response("10 multiplied by 7 equals 70."),
        ]

        with (
            patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "mock-secret-key",
                    "MONGODB_URI": "mongodb+srv://user:pass@cluster0.example.mongodb.net/",
                },
                clear=False,
            ),
            patch("agent.genai.Client", return_value=mock_gemini),
            patch("database.MongoClient", return_value=fake_mongo_client),
        ):
            at = AppTest.from_file("app.py").run()
            at.text_input(key="autonomous_goal").input("Compute 10 * 7").run()
            at.button(key="run_agent_btn").click().run()

            self.assertFalse(at.exception)
            self.assertTrue(any("saved to mongodb atlas" in s.value.lower() for s in at.success))

            # Simulate a fresh session by clearing in-memory history, then reload from Activity page
            at.session_state["activity_history"] = []
            at.sidebar.radio[0].set_value("Activity").run()
            at.button(key="activity_mongo_load_btn").click().run()

            self.assertFalse(at.exception)
            self.assertEqual(len(at.session_state["activity_history"]), 1)
            rendered_markdown = "\n".join(m.value for m in at.markdown)
            self.assertIn("10 * 7 = 70", rendered_markdown)
            self.assertIn("10 multiplied by 7 equals 70.", rendered_markdown)

    def test_knowledge_base_semantic_indexing_and_vector_search_flow(self) -> None:
        ready_index = [{"name": "chunk_embedding_index", "status": "READY", "queryable": True}]
        fake_vec_client = _FakeVectorMongoClient(search_indexes=ready_index)

        mock_genai = MagicMock()
        mock_genai.models.embed_content.side_effect = [
            SimpleNamespace(embeddings=[SimpleNamespace(values=_unit_vec(768, 0))]),
            SimpleNamespace(embeddings=[SimpleNamespace(values=_unit_vec(768, 0))]),
        ]
        mock_genai.models.generate_content.return_value = _make_text_response(
            "Priority 1 incidents have a 2-hour SLA [sla.txt | Chunk 1]."
        )

        kb_index = rag.KnowledgeBaseIndex()
        kb_index.add_document(
            "sla.txt",
            b"Support SLA: Priority 1 incidents receive an initial response within 2 hours.",
        )

        with (
            patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "mock-secret-key",
                    "MONGODB_URI": "mongodb+srv://user:pass@cluster0.example.mongodb.net/",
                },
                clear=False,
            ),
            patch("semantic_rag.MongoClient", return_value=fake_vec_client),
            patch("semantic_rag.genai.Client", return_value=mock_genai),
            patch("rag.genai.Client", return_value=mock_genai),
        ):
            at = AppTest.from_file("app.py").run()
            at.session_state["kb_index"] = kb_index
            at.sidebar.radio[0].set_value("Knowledge Base").run()

            # Explicitly index embeddings to Atlas
            at.button(key="kb_vector_index_btn").click().run()
            self.assertFalse(at.exception)
            self.assertTrue(any("embedded 1 new/changed" in s.value.lower() for s in at.success))

            # Switch to Semantic Search mode and ask a question
            at.radio(key="kb_retrieval_mode").set_value("Semantic Search (Atlas Vector)").run()
            at.text_input(key="kb_question").input("How fast is urgent incident response?").run()
            at.button(key="kb_ask_btn").click().run()

            self.assertFalse(at.exception)
            self.assertFalse(at.error)
            self.assertFalse(at.warning)
            rendered_markdown = "\n".join(m.value for m in at.markdown)
            self.assertIn("2-hour SLA", rendered_markdown)
            self.assertIn("sla.txt | Chunk 1", rendered_markdown)

    def test_knowledge_base_semantic_fallback_warns_explicitly(self) -> None:
        # Atlas Vector Search index is MISSING
        fake_vec_client = _FakeVectorMongoClient(search_indexes=[])
        mock_genai = MagicMock()
        mock_genai.models.generate_content.return_value = _make_text_response(
            "Priority 1 incidents have a 2-hour SLA [sla.txt | Chunk 1]."
        )

        kb_index = rag.KnowledgeBaseIndex()
        kb_index.add_document(
            "sla.txt",
            b"Support SLA: Priority 1 incidents receive an initial response within 2 hours.",
        )

        with (
            patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "mock-secret-key",
                    "MONGODB_URI": "mongodb+srv://user:pass@cluster0.example.mongodb.net/",
                },
                clear=False,
            ),
            patch("semantic_rag.MongoClient", return_value=fake_vec_client),
            patch("rag.genai.Client", return_value=mock_genai),
        ):
            at = AppTest.from_file("app.py").run()
            at.session_state["kb_index"] = kb_index
            at.sidebar.radio[0].set_value("Knowledge Base").run()
            at.radio(key="kb_retrieval_mode").set_value("Semantic Search (Atlas Vector)").run()
            at.text_input(key="kb_question").input("What is the support SLA?").run()
            at.button(key="kb_ask_btn").click().run()

            self.assertFalse(at.exception)
            self.assertTrue(
                any("local tf-idf fallback" in w.value.lower() for w in at.warning)
            )


if __name__ == "__main__":
    unittest.main()
