"""Automated tests for Milestone 3 safe tools and bounded autonomous Gemini agent (fully mocked)."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
from google.genai import errors as genai_errors
from google.genai import types

import agent
import planner
import tools


def _make_function_call_response(
    calls: list[tuple[str, dict[str, object]]],
    thought_text: str | None = None,
) -> SimpleNamespace:
    """Build a mock google-genai response requesting one or more function calls."""
    fn_calls = [
        SimpleNamespace(id=f"call_{idx}", name=name, args=args)
        for idx, (name, args) in enumerate(calls, start=1)
    ]
    parts: list[Any] = []
    if thought_text:
        parts.append(SimpleNamespace(text=thought_text, thought=True))
    for fc in fn_calls:
        parts.append(types.Part.from_function_call(name=fc.name, args=dict(fc.args)))

    candidate = SimpleNamespace(content=types.Content(role="model", parts=parts))
    return SimpleNamespace(function_calls=fn_calls, candidates=[candidate], text="")


def _make_text_response(text: str, thought_text: str | None = None) -> SimpleNamespace:
    """Build a mock google-genai final text response (optionally with hidden thought part)."""
    parts: list[Any] = []
    if thought_text:
        parts.append(SimpleNamespace(text=thought_text, thought=True))
    parts.append(SimpleNamespace(text=text, thought=False))
    candidate = SimpleNamespace(content=types.Content(role="model", parts=parts))
    return SimpleNamespace(function_calls=[], candidates=[candidate], text=text)


class TestSafeTools(unittest.TestCase):
    """Verify safe Python tools and strict arithmetic parser protections."""

    def test_calculate_evaluates_valid_arithmetic(self) -> None:
        self.assertEqual(tools.calculate("(12 + 8) * 3 / 4")["result"], 15)
        self.assertEqual(tools.calculate("2 ** 8")["result"], 256)
        self.assertEqual(tools.calculate("-5 + 10.5")["result"], 5.5)
        self.assertEqual(tools.calculate("17 // 5")["result"], 3)
        self.assertEqual(tools.calculate("17 % 5")["result"], 2)

    def test_calculate_blocks_unsafe_expressions_and_code_execution(self) -> None:
        unsafe_inputs = [
            "__import__('os').system('echo hi')",
            "open('README.md').read()",
            "eval('1 + 1')",
            "exec('x = 1')",
            "(lambda x: x)(5)",
            "a + b",
            "True + 1",
            "9 ** 999",
            "10 / 0",
            "",
        ]
        for expr in unsafe_inputs:
            with self.subTest(expr=expr):
                with self.assertRaises(ValueError):
                    tools.calculate(expr)

    def test_get_current_datetime_returns_expected_fields(self) -> None:
        result = tools.get_current_datetime()
        for key in ("local_datetime", "local_date", "local_time", "weekday", "timezone", "utc_datetime"):
            self.assertIn(key, result)
            self.assertTrue(result[key])

    def test_create_action_plan_returns_structured_steps(self) -> None:
        result = tools.create_action_plan("Prepare release notes")
        self.assertEqual(result["goal"], "Prepare release notes")
        self.assertGreaterEqual(len(result["steps"]), 3)
        for step in result["steps"]:
            self.assertIn(step["priority"], ("High", "Medium", "Low"))
            self.assertTrue(step["title"])
            self.assertTrue(step["description"])

    def test_execute_tool_rejects_disallowed_tool_names(self) -> None:
        result = tools.execute_tool("run_shell", {"command": "dir"})
        self.assertEqual(result["status"], "error")
        self.assertIn("unknown or disallowed tool", result["error"].lower())

    def test_execute_tool_handles_timeout_gracefully(self) -> None:
        def slow_tool() -> dict[str, str]:
            time.sleep(0.3)
            return {"ok": "done"}

        result = tools.execute_tool(
            "get_current_datetime",
            {},
            timeout_seconds=0.05,
            registry={"get_current_datetime": slow_tool},
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("timed out", result["error"].lower())

    def test_summarize_tool_args_redacts_secret_keys(self) -> None:
        summary = tools.summarize_tool_args({"expression": "2 + 2", "api_key": "secret-123"})
        self.assertIn("expression='2 + 2'", summary)
        self.assertIn("api_key=[REDACTED]", summary)
        self.assertNotIn("secret-123", summary)


class TestAutonomousAgent(unittest.TestCase):
    """Verify Gemini function-calling agent loop without live API credits."""

    def test_tool_selection_execution_and_result_feedback(self) -> None:
        mock_client = MagicMock()
        captured_contents: list[list[types.Content]] = []

        responses = [
            _make_function_call_response([("calculate", {"expression": "(15 * 4) + 12"})]),
            _make_text_response("The calculated total is 72."),
        ]

        def fake_generate_content(**kwargs: Any) -> SimpleNamespace:
            captured_contents.append(list(kwargs["contents"]))
            return responses.pop(0)

        mock_client.models.generate_content.side_effect = fake_generate_content

        with patch("agent.genai.Client", return_value=mock_client) as mock_client_cls:
            result = agent.run_autonomous_agent(
                goal="What is (15 * 4) + 12?",
                api_key="fake-test-key",
                model="gemini-3.5-flash-lite",
            )

        self.assertEqual(mock_client_cls.call_args.kwargs["api_key"], "fake-test-key")
        self.assertIs(mock_client_cls.call_args.kwargs["vertexai"], False)

        # Verify tool result (72) was sent back to Gemini on turn 2
        self.assertEqual(len(captured_contents), 2)
        turn2_last_message = captured_contents[1][-1]
        fn_resp = turn2_last_message.parts[0].function_response
        self.assertEqual(fn_resp.name, "calculate")
        self.assertEqual(fn_resp.response["status"], "success")
        self.assertEqual(fn_resp.response["output"]["result"], 72)

        self.assertEqual(result.stopped_reason, "completed")
        self.assertEqual(result.iterations_used, 1)
        self.assertEqual(len(result.activities), 1)
        self.assertEqual(result.activities[0].tool_name, "calculate")
        self.assertEqual(result.activities[0].status, "success")
        self.assertIn("72", result.activities[0].result_summary)
        self.assertEqual(result.final_response, "The calculated total is 72.")

    def test_multi_step_tool_execution(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_function_call_response([("get_current_datetime", {})]),
            _make_function_call_response([("calculate", {"expression": "250 * 4"})]),
            _make_function_call_response([("create_action_plan", {"goal": "Q4 budget review"})]),
            _make_text_response("Completed date lookup, budget calculation (1000), and action plan."),
        ]

        result = agent.run_autonomous_agent(
            goal="Check today's date, compute 250 * 4, and create a plan for Q4 budget review.",
            client=mock_client,
            model="gemini-3.5-flash-lite",
        )

        self.assertEqual(result.stopped_reason, "completed")
        self.assertEqual(result.iterations_used, 3)
        self.assertEqual(
            [entry.tool_name for entry in result.activities],
            ["get_current_datetime", "calculate", "create_action_plan"],
        )
        self.assertTrue(all(entry.status == "success" for entry in result.activities))
        self.assertIn("1000", result.final_response)

    def test_iteration_limit_stops_after_five_iterations(self) -> None:
        mock_client = MagicMock()
        # Keep requesting tools beyond the 5-iteration safety limit
        mock_client.models.generate_content.side_effect = [
            _make_function_call_response([("calculate", {"expression": f"{i} + 1"})])
            for i in range(1, 8)
        ]

        result = agent.run_autonomous_agent(
            goal="Keep calculating forever",
            client=mock_client,
            model="gemini-3.5-flash-lite",
        )

        self.assertEqual(result.stopped_reason, "max_iterations_reached")
        self.assertEqual(result.iterations_used, 5)
        self.assertEqual(len(result.activities), 5)
        # 5 executed tool turns + 1 final check turn that hits the bound
        self.assertEqual(mock_client.models.generate_content.call_count, 6)
        self.assertIn("safety limit of 5", result.final_response.lower())

    def test_invalid_tool_call_is_recorded_and_handled_gracefully(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_function_call_response([("run_shell", {"cmd": "rm -rf /"})]),
            _make_text_response("I cannot run shell commands, so I cannot execute that action."),
        ]

        result = agent.run_autonomous_agent(
            goal="Run a shell command",
            client=mock_client,
            model="gemini-3.5-flash-lite",
        )

        self.assertEqual(result.stopped_reason, "completed")
        self.assertEqual(len(result.activities), 1)
        self.assertEqual(result.activities[0].tool_name, "run_shell")
        self.assertEqual(result.activities[0].status, "error")
        self.assertIn("disallowed", result.activities[0].result_summary.lower())
        self.assertIn("cannot run shell commands", result.final_response.lower())

    def test_hidden_reasoning_parts_are_never_shown(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_text_response(
            text="Here is the clean answer.",
            thought_text="HIDDEN INTERNAL REASONING DO NOT SHOW",
        )

        result = agent.run_autonomous_agent(
            goal="Answer briefly",
            client=mock_client,
            model="gemini-3.5-flash-lite",
        )

        self.assertEqual(result.final_response, "Here is the clean answer.")
        self.assertNotIn("HIDDEN INTERNAL REASONING", result.final_response)

    def test_missing_api_key_raises_missing_api_key_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_env = Path(tmp_dir) / ".env"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(planner.MissingAPIKeyError):
                    agent.run_autonomous_agent(goal="Check date", env_path=missing_env)

    def test_request_timeout_raises_clean_agent_error(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = httpx.ReadTimeout("timed out")

        with self.assertRaises(agent.AgentError) as ctx:
            agent.run_autonomous_agent(
                goal="Calculate 2 + 2",
                client=mock_client,
                request_timeout_seconds=10.0,
            )

        self.assertIn("timed out", str(ctx.exception).lower())

    def test_api_error_redacts_api_key_and_raises_agent_error(self) -> None:
        secret = "super-secret-agent-key-999"
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = genai_errors.APIError(
            503,
            {"error": {"message": f"Overloaded for {secret}"}},
        )

        with patch("agent.genai.Client", return_value=mock_client):
            with self.assertRaises(agent.AgentError) as ctx:
                agent.run_autonomous_agent(goal="Calculate 2 + 2", api_key=secret)

        self.assertIn("503", str(ctx.exception))
        self.assertNotIn(secret, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
