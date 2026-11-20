"""AgentForge AI — Milestone 4 Streamlit dashboard (Planning, Autonomous Agent & Local RAG)."""

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
from rag import (
    MAX_FILE_SIZE_BYTES,
    DocumentIngestionError,
    KnowledgeBaseIndex,
    RAGAnswerResult,
    RAGError,
    answer_question_from_knowledge_base,
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


def _get_kb_index() -> KnowledgeBaseIndex:
    """Return the session-only in-memory KnowledgeBaseIndex."""
    index = st.session_state.get("kb_index")
    if not isinstance(index, KnowledgeBaseIndex):
        index = KnowledgeBaseIndex()
        st.session_state["kb_index"] = index
    return index


def render_sidebar() -> str:
    """Render sidebar navigation and private local API key controls."""
    with st.sidebar:
        st.markdown("## AgentForge AI")
        st.caption("Milestone 4 · Planning, Agent & Document RAG")
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


def render_rag_answer(result: RAGAnswerResult) -> None:
    """Display the grounded RAG answer and expandable retrieved Sources."""
    with st.container(border=True):
        st.subheader("Grounded Answer")
        st.caption(
            f"Question: {result.question} · Model: `{result.model}` · "
            f"Retrieved passages: {len(result.sources)}"
        )

        if not result.has_evidence:
            st.warning(result.answer)
        else:
            st.markdown(result.answer)

        with st.expander(f"Sources ({len(result.sources)} retrieved passage(s))", expanded=False):
            if not result.sources:
                st.caption("No document passages met the minimum relevance threshold.")
            else:
                for item in result.sources:
                    with st.container(border=True):
                        st.markdown(
                            f"**[{item.chunk.citation_label}]** — Relevance: `{item.score:.2f}`"
                        )
                        st.write(item.chunk.text)


def render_overview() -> None:
    """Render the Overview dashboard with metrics, AI planning, and Autonomous Agent."""
    agent_runs = st.session_state.get("agent_runs", 0)
    tasks_completed = st.session_state.get("tasks_completed", 0)
    document_count = _get_kb_index().document_count

    col_runs, col_docs, col_tasks = st.columns(3)

    with col_runs:
        with st.container(border=True):
            st.metric(label="Agent Runs", value=agent_runs)

    with col_docs:
        with st.container(border=True):
            st.metric(label="Documents", value=document_count)

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


def render_knowledge_base_view() -> None:
    """Render the Milestone 4 local TF-IDF Document RAG Knowledge Base view."""
    kb_index = _get_kb_index()

    with st.container(border=True):
        st.subheader("Knowledge Base")
        if kb_index.document_count == 0:
            st.info(
                "No documents uploaded yet. Upload PDF or TXT files below to build an "
                "in-memory TF-IDF knowledge base for this session (files are never written to disk)."
            )
        else:
            col_summary, col_clear = st.columns([3, 1])
            with col_summary:
                st.caption(
                    f"Indexed in session memory: **{kb_index.document_count}** document(s), "
                    f"**{kb_index.chunk_count}** overlapping chunk(s)."
                )
            with col_clear:
                if st.button("Clear Knowledge Base", key="kb_clear_btn", use_container_width=True):
                    kb_index.clear()
                    st.session_state.pop("kb_last_answer", None)
                    st.session_state.pop("kb_upload_feedback", None)
                    st.session_state["kb_uploader_nonce"] = (
                        st.session_state.get("kb_uploader_nonce", 0) + 1
                    )
                    st.rerun()

            for doc in kb_index.documents:
                size_kb = max(doc.size_bytes / 1024, 0.1)
                page_info = f"{doc.page_count} page(s)" if doc.file_type == "pdf" else "TXT"
                st.markdown(
                    f"- **{doc.doc_name}** (`{page_info}`, {doc.chunk_count} chunk(s), {size_kb:.1f} KB)"
                )

    st.write("")

    with st.container(border=True):
        st.markdown("#### Upload Documents")
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        st.caption(
            f"Upload PDF or TXT files (max {max_mb:.0f} MB each). Text and retrieval index live "
            "only in your browser session memory."
        )

        uploader_nonce = st.session_state.get("kb_uploader_nonce", 0)
        uploaded_files = st.file_uploader(
            "Select PDF or TXT documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key=f"kb_uploader_{uploader_nonce}",
        )

        if st.button("Index Uploaded Documents", key="kb_index_btn"):
            if not uploaded_files:
                st.warning("Select at least one PDF or TXT file before indexing.")
            else:
                indexed_names: list[str] = []
                errors: list[str] = []
                for uploaded in uploaded_files:
                    try:
                        record = kb_index.add_document(uploaded.name, uploaded.getvalue())
                        indexed_names.append(f"{record.doc_name} ({record.chunk_count} chunks)")
                    except DocumentIngestionError as exc:
                        errors.append(str(exc))

                if indexed_names:
                    st.session_state.pop("kb_last_answer", None)
                st.session_state["kb_upload_feedback"] = {
                    "indexed": indexed_names,
                    "errors": errors,
                }
                st.rerun()

        feedback = st.session_state.get("kb_upload_feedback")
        if isinstance(feedback, dict):
            indexed_list = feedback.get("indexed") or []
            error_list = feedback.get("errors") or []
            if indexed_list:
                st.success(f"Indexed: {', '.join(indexed_list)}")
            for err in error_list:
                st.error(err)

    st.write("")

    with st.container(border=True):
        st.markdown("#### Ask a Question")
        st.caption(
            "Local TF-IDF retrieves the most relevant passages and sends only those excerpts to "
            "Gemini for a grounded, cited answer."
        )

        question = st.text_input(
            "Question",
            placeholder="Ask a question about your uploaded documents...",
            key="kb_question",
        )

        if st.button("Ask Knowledge Base", type="primary", key="kb_ask_btn"):
            if not question.strip():
                st.warning("Please enter a question before querying the Knowledge Base.")
            elif kb_index.chunk_count == 0:
                st.warning("Upload and index at least one PDF or TXT document first.")
            else:
                try:
                    with st.spinner("Retrieving passages and generating grounded answer..."):
                        result = answer_question_from_knowledge_base(
                            question=question,
                            index=kb_index,
                            api_key=_get_session_override_key(),
                        )
                    st.session_state["kb_last_answer"] = result
                except MissingAPIKeyError as exc:
                    st.warning(str(exc))
                except ValueError as exc:
                    st.warning(str(exc))
                except RAGError as exc:
                    st.error(str(exc))

    last_answer = st.session_state.get("kb_last_answer")
    if isinstance(last_answer, RAGAnswerResult):
        st.write("")
        render_rag_answer(last_answer)


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
        "AI agent platform for Gemini AI planning, autonomous tool calling, and local document RAG."
    )
    st.divider()

    if selected_section == "Overview":
        render_overview()
    elif selected_section == "AI Agent":
        render_ai_agent_view()
    elif selected_section == "Knowledge Base":
        render_knowledge_base_view()
    elif selected_section == "Activity":
        render_activity_view()


if __name__ == "__main__":
    main()
