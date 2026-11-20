"""AgentForge AI — Milestone 1 Streamlit dashboard."""

import streamlit as st

NAV_ITEMS = ("Overview", "AI Agent", "Knowledge Base", "Activity")


def render_sidebar() -> str:
    """Render sidebar navigation and return the selected section."""
    with st.sidebar:
        st.markdown("## AgentForge AI")
        st.caption("Milestone 1 · Platform Foundation")
        st.divider()

        selection = st.radio(
            "Navigation",
            options=NAV_ITEMS,
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("AI Engine: Not connected (Gemini integration coming soon)")

    return selection


def render_overview() -> None:
    """Render the Overview dashboard with baseline metrics and goal input."""
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
        st.subheader("Run an Agent")
        st.caption(
            "Enter a goal to launch an agent workflow once Gemini integration is connected."
        )

        goal = st.text_input(
            "Goal",
            placeholder="Describe what you want the agent to accomplish...",
            key="agent_goal",
        )

        if st.button("Run Agent", type="primary"):
            if goal.strip():
                st.info(
                    "AI integration is coming soon. Gemini is not connected yet, "
                    "so no agent run was executed for your goal."
                )
            else:
                st.info(
                    "AI integration is coming soon. Gemini is not connected yet — "
                    "once connected, you will be able to enter a goal and run live agent workflows."
                )


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
        "AI agent platform foundation for Gemini tool calling, RAG, and multi-agent workflows."
    )
    st.divider()

    if selected_section == "Overview":
        render_overview()
    elif selected_section == "AI Agent":
        render_placeholder(
            "AI Agent",
            "Coming soon: Configure Gemini models, tool calling, and structured agent execution here.",
        )
    elif selected_section == "Knowledge Base":
        render_placeholder(
            "Knowledge Base",
            "Coming soon: Upload and index documents for retrieval-augmented generation (RAG) with pgvector.",
        )
    elif selected_section == "Activity":
        render_placeholder(
            "Activity",
            "Coming soon: Inspect agent runs, tool call traces, and task completion history.",
        )


if __name__ == "__main__":
    main()
