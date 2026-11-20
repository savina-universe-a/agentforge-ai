"""AgentForge AI — Milestone 2 Streamlit dashboard with Gemini AI planning."""

from __future__ import annotations

import streamlit as st

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
        st.caption("Milestone 2 · Gemini AI Planning")
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
            st.caption(f"AI Planning: Ready (`{model_name}`)")
        else:
            st.caption("AI Planning: GEMINI_API_KEY not configured")

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


def render_overview() -> None:
    """Render the Overview dashboard with baseline metrics and Gemini AI planning."""
    col_runs, col_docs, col_tasks = st.columns(3)

    with col_runs:
        with st.container(border=True):
            st.metric(label="Agent Runs", value=0)

    with col_docs:
        with st.container(border=True):
            st.metric(label="Documents", value=0)

    with col_tasks:
        with st.container(border=True):
            st.metric(label="Tasks Completed", value=0)

    st.write("")

    with st.container(border=True):
        st.subheader("AI Planning")
        st.caption(
            "Enter a goal to generate a prioritized action plan with Gemini. "
            "This is AI planning only — autonomous tool execution is not enabled yet."
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
        "AI agent platform for Gemini planning, future tool calling, RAG, and multi-agent workflows."
    )
    st.divider()

    if selected_section == "Overview":
        render_overview()
    elif selected_section == "AI Agent":
        render_placeholder(
            "AI Agent",
            "Coming soon: Autonomous tool calling and structured agent execution. "
            "For Milestone 2, AI planning is available on the Overview page.",
        )
    elif selected_section == "Knowledge Base":
        render_placeholder(
            "Knowledge Base",
            "Coming soon: Upload and index documents for retrieval-augmented generation (RAG) with pgvector.",
        )
    elif selected_section == "Activity":
        render_placeholder(
            "Activity",
            "Coming soon: Inspect saved runs, tool call traces, and task completion history.",
        )


if __name__ == "__main__":
    main()
