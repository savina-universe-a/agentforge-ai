"""Unit tests for planner.py using mocked google-genai responses (no live API calls)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from google.genai import errors as genai_errors

import planner


SAMPLE_PLAN_JSON = """
{
  "summary": "Break the launch into scoping, implementation, and verification phases.",
  "steps": [
    {
      "step_number": 1,
      "title": "Define scope and success criteria",
      "description": "Write down core user flows, target metrics, and release constraints.",
      "priority": "High"
    },
    {
      "step_number": 2,
      "title": "Build core prototype",
      "description": "Implement the primary workflow and validate with local unit tests.",
      "priority": "Medium"
    },
    {
      "step_number": 3,
      "title": "Document onboarding steps",
      "description": "Add setup notes and troubleshooting tips to the README.",
      "priority": "Low"
    }
  ]
}
"""


class TestPlanner(unittest.TestCase):
    """Verify Gemini planning logic, dotenv loading, and error handling without live calls."""

    def test_generate_action_plan_returns_structured_steps_and_priorities(self) -> None:
        parsed_plan = planner.ActionPlan.model_validate_json(SAMPLE_PLAN_JSON)
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = SimpleNamespace(
            parsed=parsed_plan,
            text=SAMPLE_PLAN_JSON,
        )

        with patch("planner.genai.Client", return_value=mock_client) as mock_client_cls:
            result = planner.generate_action_plan(
                goal="Launch MVP dashboard",
                api_key="fake-test-key",
            )

        mock_client_cls.assert_called_once_with(api_key="fake-test-key", vertexai=False)
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], planner.DEFAULT_MODEL)
        self.assertIn("Launch MVP dashboard", call_kwargs["contents"])

        self.assertEqual(result.summary, parsed_plan.summary)
        self.assertEqual(len(result.steps), 3)
        self.assertEqual(
            [step.priority for step in result.steps],
            ["High", "Medium", "Low"],
        )

    def test_generate_action_plan_parses_json_text_when_parsed_attr_is_none(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = SimpleNamespace(
            parsed=None,
            text=SAMPLE_PLAN_JSON,
        )

        result = planner.generate_action_plan(
            goal="Ship beta",
            client=mock_client,
        )

        self.assertEqual(len(result.steps), 3)
        self.assertEqual(result.steps[0].title, "Define scope and success criteria")

    def test_generate_action_plan_rejects_empty_goal(self) -> None:
        with self.assertRaises(ValueError):
            planner.generate_action_plan(goal="   ", api_key="fake-test-key")

    def test_generate_action_plan_raises_missing_api_key_when_unconfigured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_env = Path(tmp_dir) / ".env"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(planner.MissingAPIKeyError):
                    planner.generate_action_plan(
                        goal="Create a study schedule",
                        env_path=missing_env,
                    )

    def test_get_api_key_reads_from_local_dotenv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text("GEMINI_API_KEY=local-dotenv-test-key\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                loaded = planner.get_api_key(env_path=env_file)

            self.assertEqual(loaded, "local-dotenv-test-key")

    def test_get_model_name_reads_from_dotenv_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(planner.get_model_name(env_path=env_file), "gemini-3.5-flash-lite")

            env_file.write_text("GEMINI_MODEL=gemini-3.5-flash\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(planner.get_model_name(env_path=env_file), "gemini-3.5-flash")

    def test_generate_action_plan_handles_503_unavailable(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = genai_errors.APIError(
            503,
            {"error": {"message": "High demand"}},
        )

        with patch("planner.genai.Client", return_value=mock_client):
            with self.assertRaises(planner.PlannerError) as ctx:
                planner.generate_action_plan(goal="Plan sprint", api_key="fake-test-key")

        self.assertIn("503", str(ctx.exception))

    def test_save_api_key_to_env_persists_key_and_validates_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            with patch.dict(os.environ, {}, clear=True):
                saved_path = planner.save_api_key_to_env("saved-test-key", env_path=env_file)
                self.assertEqual(saved_path, env_file)
                self.assertEqual(os.environ.get("GEMINI_API_KEY"), "saved-test-key")
                self.assertIn("GEMINI_API_KEY", env_file.read_text(encoding="utf-8"))

            with self.assertRaises(ValueError):
                planner.save_api_key_to_env("   ", env_path=env_file)
            with self.assertRaises(ValueError):
                planner.save_api_key_to_env("line1\nline2", env_path=env_file)

    def test_generate_action_plan_handles_api_error_without_exposing_key(self) -> None:
        secret = "super-secret-fake-key-123"
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = genai_errors.APIError(
            401,
            {"error": {"message": f"Invalid key: {secret}"}},
        )

        with patch("planner.genai.Client", return_value=mock_client):
            with self.assertRaises(planner.PlannerError) as ctx:
                planner.generate_action_plan(goal="Plan sprint", api_key=secret)

        self.assertIn("rejected", str(ctx.exception).lower())
        self.assertNotIn(secret, str(ctx.exception))

    def test_generate_action_plan_redacts_secret_in_unexpected_errors(self) -> None:
        secret = "another-secret-fake-key-456"
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError(
            f"Connection failed while using {secret}"
        )

        with patch("planner.genai.Client", return_value=mock_client):
            with self.assertRaises(planner.PlannerError) as ctx:
                planner.generate_action_plan(goal="Plan sprint", api_key=secret)

        self.assertNotIn(secret, str(ctx.exception))
        self.assertIn("[REDACTED]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
