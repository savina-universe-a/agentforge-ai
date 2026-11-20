# AgentForge AI

A clean, modular AI agent platform built with Python, Streamlit, `google-genai`, MongoDB Atlas, local & semantic RAG, and LangGraph multi-agent orchestration.

**Milestone 7** adds **LangGraph multi-agent orchestration** (`StateGraph` with typed `WorkflowState`) featuring a dynamic **Supervisor** router and three specialized roles (**Research Agent**, **Analysis Agent**, and **Writer Agent**) with strict step/API budgets, explicit retry/fallback handling, and MongoDB `workflow_runs` persistence.

## Features by Milestone

- **Milestone 1 — Dashboard Foundation:** Minimal Streamlit UI with `Overview`, `AI Agent`, `Knowledge Base`, and `Activity` navigation and baseline metrics.
- **Milestone 2 — Gemini AI Planning:** Generates structured, prioritized action plans using `google-genai` and `python-dotenv` with private local `GEMINI_API_KEY` setup.
- **Milestone 3 — Autonomous Tool-Calling Agent:** Bounded (max 5 iterations) Gemini function-calling loop over three safe allowlisted Python tools (`calculate` via restricted AST parsing, `get_current_datetime`, and `create_action_plan`), showing an **Activity Log** and **Final Response**.
- **Milestone 4 — Local Document RAG Knowledge Base:** In-memory PDF (`pypdf`) and TXT extraction with overlapping chunking, deterministic source IDs, local `scikit-learn` TF-IDF retrieval, prompt-injection defenses, inline citations, and expandable **Sources**.
- **Milestone 5 — MongoDB Atlas Persistence (`pymongo`):** Persists Knowledge Base documents (`documents`) and autonomous agent run logs (`agent_runs`) under stable IDs with duplicate detection and session-only fallback.
- **Milestone 6 — Semantic RAG (Gemini Embeddings + Atlas Vector Search):** Embeds chunks with `gemini-embedding-001` (`768` dims, L2-normalized) into `document_chunks` and queries via Atlas `$vectorSearch` (`chunk_embedding_index`), with idempotent syncing and explicit TF-IDF fallback.
- **Milestone 7 — LangGraph Multi-Agent Orchestration (`langgraph`):**
  - Typed `WorkflowState` and bounded `StateGraph` with dynamic intent routing (`research`, `analysis`, `research_and_analysis`, or `direct_write`) — only runs the agents needed for each task.
  - **Supervisor:** Classifies intent, manages handoffs, enforces max graph steps (default `8`) and Gemini API call budgets (default `6`), and performs at most one explicit broadened-query retry when initial document retrieval returns zero passages.
  - **Research Agent:** Reuses Atlas Vector Search (`semantic_rag.py`) with explicit, transparent fallback to local TF-IDF (`rag.py`); treats uploaded text strictly as untrusted data.
  - **Analysis Agent:** Reuses existing safe tools (`tools.py`: `calculate`, `get_current_datetime`, `create_action_plan`) for exact arithmetic, date/time lookups, and structured planning.
  - **Writer Agent:** Produces a grounded final response with inline citations (`[doc | Page X | Chunk Y]`) and explicitly reports missing evidence instead of fabricating facts.
  - Persists workflow traces and outputs to a dedicated MongoDB `workflow_runs` collection without affecting `agent_runs`, and supports full local-only execution when MongoDB is unavailable.

## Prerequisites

- Windows 10/11
- Python 3.11 or newer
- A Google Gemini API key (from Google AI Studio)
- *(Optional for Milestones 5–7)* A MongoDB Atlas cluster with Vector Search enabled

## Windows Setup (PowerShell)

1. **Create and activate a virtual environment** (skip creation if `.venv` already exists):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(If PowerShell blocks script execution, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and retry.)*

2. **Install dependencies**:
   ```powershell
   python -m pip install -r requirements.txt
   ```

3. **Configure local `.env` privately (never paste secrets in chat or commit `.env`)**:
   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

4. **Run the Streamlit app**:
   ```powershell
   python -m streamlit run app.py
   ```
   Open `http://localhost:8501`. *(Using `python -m streamlit` avoids Windows Application Control blocks on generated `.venv\Scripts\streamlit.exe` wrappers.)*

## Configuration

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes (for live AI & embeddings) | — | Your private Gemini API key |
| `GEMINI_MODEL` | No | `gemini-3.5-flash-lite` | Gemini model for planning, agent tool calling, RAG, and multi-agent workflow |
| `GEMINI_EMBEDDING_MODEL` | No | `gemini-embedding-001` | Gemini embedding model for Semantic RAG |
| `GEMINI_EMBEDDING_DIMENSIONS` | No | `768` | Output dimensionality (L2-normalized) matching the Atlas Vector index |
| `MONGODB_URI` | No | — | MongoDB Atlas SRV URI (`mongodb+srv://...`); omit for local-only mode |
| `MONGODB_DB_NAME` | No | `agentforge_ai` | MongoDB database (`documents`, `document_chunks`, `agent_runs`, `workflow_runs`) |
| `MONGODB_VECTOR_INDEX_NAME` | No | `chunk_embedding_index` | Atlas Vector Search index name on `document_chunks` |

## MongoDB Atlas Vector Search Index Setup

1. Upload/index at least one document in **Knowledge Base** and click **Index Embeddings to Atlas** (creates `agentforge_ai.document_chunks`).
2. In MongoDB Atlas → **Search & Vector Search** → **Create Vector Search Index** → **JSON Editor**, select `agentforge_ai.document_chunks`, name the index `chunk_embedding_index`, and paste:
   ```json
   {
     "fields": [
       {
         "type": "vector",
         "path": "embedding",
         "numDimensions": 768,
         "similarity": "cosine"
       },
       {
         "type": "filter",
         "path": "doc_id"
       },
       {
         "type": "filter",
         "path": "doc_name"
       }
     ]
   }
   ```
3. Wait until the index status becomes **READY**, then verify via **Check Vector Index** in AgentForge AI.

## How to Verify & Test

### 1. Automated Tests (Mocked — Zero Live API or Atlas Calls)

```powershell
python -m unittest discover -v
```

### 2. Live Multi-Agent Workflow Test (Browser)

1. Open `http://localhost:8501`.
2. Ensure at least one document is indexed in **Knowledge Base** (either in session memory or in Atlas `document_chunks`).
3. Open the **AI Agent** page → **Multi-Agent Workflow (LangGraph)**.
4. Test **dynamic multi-agent handoff** (`research_and_analysis`):
   > Look up our SLA or release policy in the Knowledge Base, calculate the cost of 3 incidents at 250 dollars each, and write a concise summary with citations.
   - Click **Run Multi-Agent Workflow**.
   - Verify the **Agent Execution Trace** shows `Supervisor → Research Agent → Supervisor → Analysis Agent → Supervisor → Writer Agent`, inline citations in **Final Response**, and retrieved passages in **Citations & Retrieved Evidence**.
5. Test **minimal routing** (`analysis` only):
   > Check today's date and calculate 18 * 45 + 120.
   - Verify **Research Agent** is skipped in the execution trace.
6. Refresh the browser, open **Activity**, and click **Load Saved Workflows from MongoDB** to confirm the workflow trace persisted cleanly in `workflow_runs`.

## Project Structure

```text
agentforge-ai/
├── app.py               # Streamlit dashboard (Planning, Multi-Agent & Single Agent, RAG, Activity)
├── multi_agent.py       # Bounded LangGraph StateGraph (Supervisor, Research, Analysis, Writer)
├── semantic_rag.py      # Gemini embeddings (768d L2-normalized) & MongoDB Atlas Vector Search
├── database.py          # MongoDB Atlas persistence (documents, agent_runs, workflow_runs)
├── rag.py               # PDF/TXT extraction, overlapping chunking, TF-IDF & grounded Q&A
├── agent.py             # Bounded Gemini single-agent tool-calling loop (max 5 iterations)
├── tools.py             # Safe allowlisted Python tools (AST calculate, datetime, action plan)
├── planner.py           # Standalone Gemini AI planning service
├── test_app.py          # Streamlit AppTest UI tests (mocked Gemini & MongoDB)
├── test_multi_agent.py  # LangGraph routing, handoffs, retries, budgets & persistence tests (mocked)
├── test_semantic_rag.py # Embedding, dimension, idempotent indexing & Vector Search tests (mocked)
├── test_database.py     # MongoDB persistence, reload parity & error-handling tests (mocked)
├── test_rag.py          # Extraction, chunking, TF-IDF retrieval & grounded RAG tests (mocked)
├── test_agent.py        # Safe tools & autonomous agent loop unit tests (mocked)
├── test_planner.py      # Planner unit tests (mocked)
├── requirements.txt     # Python dependencies (including langgraph)
├── .env.example         # Template for local environment variables (no secrets)
├── .gitignore           # Excludes .venv, caches, .env, secrets, and local DB files
└── README.md            # Setup, architecture, and verification instructions
```
