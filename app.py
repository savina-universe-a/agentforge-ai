"""AgentForge AI — Milestone 3 Streamlit dashboard with AI planning & autonomous tool-calling agent."""

from __future__ import annotations

import streamlit as st

from agent import (
    MAX_TOOL_ITERATIONS,
    AgentError,
    AgentRunResult,
    run_autonomous_agent,
)
from planner import (
    ActionPlan,
    MissingAPIKeyError,
    PlannerError,
    generate_action_plan,
    get_api_key,
    get_model_name,
    save_api_key_to_env,
)

NAV_ITEMS = ("Overview", "AI Agent", "Knowledge Base", "Activity")

PRIORITY_BADGES = {
    "High": "🔴 High Priority",
    "Medium": "🟡 Medium Priority",
    "Low": "🟢 Low Priority",
}


def _get_session_override_key() -> str | None:
    """Return an API key entered privately in the current browser session, if any."""
    saved_session_key = st.session_state.get("session_api_key", "")
    typed_key = st.session_state.get("sidebar_api_key_input", "")
    candidate = (saved_session_key or typed_key or "").strip()
    return candidate or None


def render_sidebar() -> str:
    """Render sidebar navigation and private local API key controls."""
    with st.sidebar:
        st.markdown("## AgentForge AI")
        st.caption("Milestone 3 · Autonomous Tool-Calling Agent")
        st.divider()

        selection = st.radio(
            "Navigation",
            options=NAV_ITEMS,
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        model_name = get_model_name()
        key_configured = bool(get_api_key(override_key=_get_session_override_key()))
        if key_configured:
            st.caption(f"Gemini API: Ready (`{model_name}`)")
        else:
            st.caption("Gemini API: GEMINI_API_KEY not configured")

        with st.expander("API Key Setup (Private)", expanded=False):
            st.caption(
                "Enter your Gemini API key locally. It is masked, never logged, "
                "and never shared outside your machine."
            )
            entered_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="Paste GEMINI_API_KEY privately...",
                key="sidebar_api_key_input",
            )

            col_session, col_env = st.columns(2)
            with col_session:
                if st.button("Use Session", key="use_session_key_btn", use_container_width=True):
                    if entered_key.strip():
                        st.session_state["session_api_key"] = entered_key.strip()
                        st.success("Key active for this session.")
                    else:
                        st.warning("Enter an API key first.")

            with col_env:
                if st.button("Save to .env", key="save_env_key_btn", use_container_width=True):
                    try:
                        save_api_key_to_env(entered_key)
                        st.session_state["session_api_key"] = entered_key.strip()
                        st.success("Saved to local .env (gitignored).")
                    except ValueError as exc:
                        st.warning(str(exc))

    return selection


def render_action_plan(plan: ActionPlan, goal: str) -> None:
    """Display the structured Gemini action plan with clear steps and priorities."""
    with st.container(border=True):
        st.subheader("Generated Action Plan")
        st.caption(
            f"Goal: {goal} · AI planning output only (no autonomous tool execution)."
        )
        st.markdown(plan.summary)
        st.divider()

        for step in plan.steps:
            badge = PRIORITY_BADGES.get(step.priority, step.priority)
            with st.container(border=True):
                st.markdown(f"**Step {step.step_number}: {step.title}** — `{badge}`")
                st.write(step.description)


def render_agent_result(result: AgentRunResult) -> None:
    """Display the autonomous agent's tool activity log and visible final response."""
    with st.container(border=True):
        st.subheader("Autonomous Agent Result")
        st.caption(
            f"Goal: {result.goal} · Model: `{result.model}` · "
            f"Tool iterations: {result.iterations_used}/{result.max_iterations}"
        )

        if result.stopped_reason == "max_iterations_reached":
            st.warning(
                f"Reached the safety limit of {result.max_iterations} tool iteration(s)."
            )

        st.markdown("#### Activity Log")
        if not result.activities:
            st.caption("No tools were called for this request.")
        else:
            for entry in result.activities:
                status_badge = "✅ Success" if entry.status == "success" else "⚠️ Error"
                with st.container(border=True):
                    st.markdown(
                        f"**Iteration {entry.iteration} · Tool:** `{entry.tool_name}` — `{status_badge}`"
                    )
                    st.markdown(f"**Input:** `{entry.input_summary}`")
                    st.markdown(f"**Result:** {entry.result_summary}")

        st.divider()
        st.markdown("#### Final Response")
        st.markdown(result.final_response)


def render_autonomous_agent_controls(
    input_key: str = "autonomous_goal",
    button_key: str = "run_agent_btn",
    show_header: bool = True,
) -> None:
    """Render the Run Autonomous Agent form, execution flow, and latest run output."""
    with st.container(border=True):
        if show_header:
            st.subheader("Autonomous Agent (Tool Calling)")
            st.caption(
                f"Gemini autonomously selects and executes safe Python tools "
                f"(`calculate`, `get_current_datetime`, `create_action_plan`) across up to "
                f"{MAX_TOOL_ITERATIONS} bounded iterations."
            )

        agent_goal = st.text_input(
            "Agent Task",
            placeholder="e.g., Check today's date, calculate 18 * 45 + 120, and create a sprint plan...",
            key=input_key,
        )

        if st.button("Run Autonomous Agent", type="primary", key=button_key):
            if not agent_goal.strip():
                st.warning("Please enter a goal before running the autonomous agent.")
            else:
                try:
                    with st.spinner("Running autonomous Gemini tool-calling agent..."):
                        result = run_autonomous_agent(
                            goal=agent_goal,
                            api_key=_get_session_override_key(),
                        )
                    st.session_state["last_agent_result"] = result
                    st.session_state["agent_runs"] = st.session_state.get("agent_runs", 0) + 1
                    if result.stopped_reason == "completed":
                        st.session_state["tasks_completed"] = (
                            st.session_state.get("tasks_completed", 0) + 1
                        )
                    history: list[AgentRunResult] = list(
                        st.session_state.get("activity_history", [])
                    )
                    history.insert(0, result)
                    st.session_state["activity_history"] = history
                    st.rerun()
                except MissingAPIKeyError as exc:
                    st.warning(str(exc))
                except ValueError as exc:
                    st.warning(str(exc))
                except AgentError as exc:
                    st.error(str(exc))

    last_result = st.session_state.get("last_agent_result")
    if isinstance(last_result, AgentRunResult):
        st.write("")
        render_agent_result(last_result)


def render_overview() -> None:
    """Render the Overview dashboard with metrics, AI planning, and Autonomous Agent."""
    agent_runs = st.session_state.get("agent_runs", 0)
    tasks_completed = st.session_state.get("tasks_completed", 0)

    col_runs, col_docs, col_tasks = st.columns(3)

    with col_runs:
        with st.container(border=True):
            st.metric(label="Agent Runs", value=agent_runs)

    with col_docs:
        with st.container(border=True):
            st.metric(label="Documents", value=0)

    with col_tasks:
        with st.container(border=True):
            st.metric(label="Tasks Completed", value=tasks_completed)

    st.write("")

    with st.container(border=True):
        st.subheader("AI Planning")
        st.caption(
            "Enter a goal to generate a prioritized action plan with Gemini. "
            "This is AI planning only — autonomous tool execution is handled separately below."
        )

        goal = st.text_input(
            "Goal",
            placeholder="Describe what you want a prioritized action plan for...",
            key="agent_goal",
        )

        if st.button("Generate Action Plan", type="primary", key="generate_plan_btn"):
            if not goal.strip():
                st.warning("Please enter a goal before generating an action plan.")
            else:
                try:
                    with st.spinner("Generating prioritized action plan with Gemini..."):
                        plan = generate_action_plan(
                            goal=goal,
                            api_key=_get_session_override_key(),
                        )
                    st.session_state["last_plan"] = plan
                    st.session_state["last_plan_goal"] = goal.strip()
                except MissingAPIKeyError as exc:
                    st.warning(str(exc))
                except ValueError as exc:
                    st.warning(str(exc))
                except PlannerError as exc:
                    st.error(str(exc))

    last_plan = st.session_state.get("last_plan")
    last_goal = st.session_state.get("last_plan_goal", "")
    if isinstance(last_plan, ActionPlan):
        st.write("")
        render_action_plan(last_plan, last_goal)

    st.write("")
    render_autonomous_agent_controls(
        input_key="autonomous_goal",
        button_key="run_agent_btn",
        show_header=True,
    )


def render_ai_agent_view() -> None:
    """Render the dedicated AI Agent workspace."""
    with st.container(border=True):
        st.subheader("AI Agent")
        st.info(
            "Autonomous Gemini tool-calling agent (bounded to 5 iterations). "
            "Available safe tools: `calculate`, `get_current_datetime`, and `create_action_plan`."
        )

    st.write("")
    render_autonomous_agent_controls(
        input_key="ai_agent_view_goal",
        button_key="ai_agent_view_run_btn",
        show_header=False,
    )


def render_activity_view() -> None:
    """Render the Activity view showing autonomous agent runs from the current session."""
    history: list[AgentRunResult] = st.session_state.get("activity_history", [])
    with st.container(border=True):
        st.subheader("Activity")
        if not history:
            st.info(
                "No autonomous agent runs recorded in this session yet. "
                "Run the Autonomous Agent from Overview or AI Agent to see tool activity here."
            )
            return

    for run in history:
        st.write("")
        render_agent_result(run)


def render_placeholder(title: str, description: str) -> None:
    """Render a minimal placeholder view for upcoming platform sections."""
    with st.container(border=True):
        st.subheader(title)
        st.info(description)


def main() -> None:
    """Configure the page and render the selected section."""
    st.set_page_config(
        page_title="AgentForge AI",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    selected_section = render_sidebar()

    st.title("AgentForge AI")
    st.caption(
        "AI agent platform for Gemini AI planning, autonomous tool calling, and future RAG workflows."
    )
    st.divider()

    if selected_section == "Overview":
        render_overview()
    elif selected_section == "AI Agent":
        render_ai_agent_view()
    elif selected_section == "Knowledge Base":
        render_placeholder(
            "Knowledge Base",
            "Coming soon: Upload and index documents for retrieval-augmented generation (RAG) with pgvector.",
        )
    elif selected_section == "Activity":
        render_activity_view()


if __name__ == "__main__":
    main()
