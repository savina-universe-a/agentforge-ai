"""UI verification tests for AgentForge AI using Streamlit AppTest."""

import unittest

from streamlit.testing.v1 import AppTest


class TestAgentForgeApp(unittest.TestCase):
    """Verify Milestone 1 dashboard rendering and Run Agent button behavior."""

    def test_overview_renders_title_metrics_and_sidebar(self) -> None:
        at = AppTest.from_file("app.py").run()

        self.assertFalse(at.exception)
        self.assertEqual([t.value for t in at.title], ["AgentForge AI"])

        # Sidebar navigation items
        self.assertEqual(len(at.sidebar.radio), 1)
        self.assertEqual(
            list(at.sidebar.radio[0].options),
            ["Overview", "AI Agent", "Knowledge Base", "Activity"],
        )

        # Overview metrics
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(
            metrics,
            {
                "Agent Runs": "0",
                "Documents": "0",
                "Tasks Completed": "0",
            },
        )

    def test_run_agent_button_reports_ai_integration_coming(self) -> None:
        at = AppTest.from_file("app.py").run()

        at.text_input(key="agent_goal").input("Draft a launch checklist").run()
        at.button[0].click().run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.info), 1)
        message = at.info[0].value.lower()
        self.assertIn("ai integration is coming", message)
        self.assertIn("not connected", message)

        # Metrics must remain at 0 (no simulated or faked run)
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics["Agent Runs"], "0")
        self.assertEqual(metrics["Tasks Completed"], "0")

    def test_run_agent_button_with_empty_goal_reports_ai_integration_coming(self) -> None:
        at = AppTest.from_file("app.py").run()

        at.button[0].click().run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.info), 1)
        self.assertIn("ai integration is coming", at.info[0].value.lower())

    def test_sidebar_placeholder_views_load_cleanly(self) -> None:
        at = AppTest.from_file("app.py").run()

        for section in ("AI Agent", "Knowledge Base", "Activity"):
            at.sidebar.radio[0].set_value(section).run()
            self.assertFalse(at.exception)
            self.assertEqual(at.subheader[0].value, section)
            self.assertEqual(len(at.info), 1)


if __name__ == "__main__":
    unittest.main()
