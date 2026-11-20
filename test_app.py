"""UI verification tests for AgentForge AI using Streamlit AppTest (mocked Gemini calls)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

import planner


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
        mock_client_cls.assert_called_once_with(api_key="session-only-fake-key")
        rendered_markdown = "\n".join(m.value for m in at.markdown)
        self.assertIn("Step 1: Finalize requirements", rendered_markdown)

    def test_sidebar_placeholder_views_load_cleanly(self) -> None:
        at = AppTest.from_file("app.py").run()

        for section in ("AI Agent", "Knowledge Base", "Activity"):
            at.sidebar.radio[0].set_value(section).run()
            self.assertFalse(at.exception)
            self.assertEqual(at.subheader[0].value, section)
            self.assertEqual(len(at.info), 1)


if __name__ == "__main__":
    unittest.main()
