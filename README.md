# AgentForge AI

A clean, modular AI agent platform built with Python, Streamlit, `google-genai`, local TF-IDF RAG, and MongoDB Atlas persistence.

**Milestone 5** adds optional **MongoDB Atlas persistence** (`pymongo`) for Knowledge Base documents and autonomous agent activity history while keeping local session-only usage and local `scikit-learn` TF-IDF retrieval working seamlessly (no vector search or embeddings yet).

## Features by Milestone

- **Milestone 1 — Dashboard Foundation:** Minimal Streamlit UI with `Overview`, `AI Agent`, `Knowledge Base`, and `Activity` navigation and baseline metrics.
- **Milestone 2 — Gemini AI Planning:** Generates structured, prioritized action plans using `google-genai` and `python-dotenv` with private local `GEMINI_API_KEY` setup.
- **Milestone 3 — Autonomous Tool-Calling Agent:** Bounded (max 5 iterations) Gemini function-calling loop over three safe allowlisted Python tools (`calculate` via restricted AST parsing, `get_current_datetime`, and `create_action_plan`), showing an **Activity Log** and **Final Response**.
- **Milestone 4 — Local Document RAG Knowledge Base:** In-memory PDF (`pypdf`) and TXT extraction with overlapping chunking, deterministic source IDs, local `scikit-learn` TF-IDF retrieval, prompt-injection defenses, inline citations, and expandable **Sources**.
- **Milestone 5 — MongoDB Atlas Persistence (`pymongo`):**
  - Connects via `MONGODB_URI` and `MONGODB_DB_NAME` (default `agentforge_ai`) from local `.env`.
  - Saves document metadata, extracted page text, and exact chunk identifiers (`documents` collection) under stable document IDs (`doc:<slug>`) with duplicate detection — reloading into the local TF-IDF index preserves identical source names, page/chunk IDs, and retrieval scores.
  - Persists completed autonomous agent run summaries and safe tool activity logs (`agent_runs` collection) without storing API keys, connection strings, or hidden reasoning.
  - Gracefully falls back to session-only mode whenever `MONGODB_URI` is unset or Atlas is unreachable.

## Prerequisites

- Windows 10/11
- Python 3.11 or newer
- A Google Gemini API key (from Google AI Studio) for live AI calls
- *(Optional)* A MongoDB Atlas cluster URI for persistence across browser restarts

## Windows Setup (PowerShell)

1. **Create a virtual environment** in the repository root (skip if `.venv` already exists):
   ```powershell
   python -m venv .venv
   ```
   *(Or `py -3.11 -m venv .venv` if using the Windows Python Launcher.)*

2. **Activate the virtual environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   *(If PowerShell blocks script execution, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and retry.)*

3. **Install dependencies**:
   ```powershell
   python -m pip install -r requirements.txt
   ```

4. **Configure local `.env` privately (never paste secrets in chat or commit `.env`)**:
   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```
   Set:
   - `GEMINI_API_KEY=` — your private Gemini API key (or enter it privately in the sidebar under **API Key Setup (Private)**).
   - `GEMINI_MODEL=gemini-3.5-flash-lite`
   - `MONGODB_URI=` — *(optional)* your MongoDB Atlas SRV connection string (`mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority`).
   - `MONGODB_DB_NAME=agentforge_ai`

   *(In MongoDB Atlas, make sure your current IP address is allowed under **Network Access** and your database user has read/write permissions on `agentforge_ai`.)*

5. **Run the Streamlit app**:
   ```powershell
   python -m streamlit run app.py
   ```
   Streamlit starts a local server at `http://localhost:8501`.
   *(Using `python -m streamlit` also avoids Windows Application Control blocks on generated `.venv\Scripts\streamlit.exe` wrappers.)*

## Configuration

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes (for live AI calls) | — | Your private Gemini API key |
| `GEMINI_MODEL` | No | `gemini-3.5-flash-lite` | Available Gemini model used across planning, agent, and RAG |
| `MONGODB_URI` | No | — | MongoDB Atlas connection string (`mongodb+srv://...`); omit for session-only mode |
| `MONGODB_DB_NAME` | No | `agentforge_ai` | MongoDB database name (`documents` and `agent_runs` collections) |

## How to Verify & Test

### 1. Automated Tests (Mocked — Zero Live DB or Gemini API Calls)

```powershell
python -m unittest discover -v
```

### 2. Live MongoDB Atlas Persistence Test

1. Add your `MONGODB_URI` and `MONGODB_DB_NAME=agentforge_ai` to your local `.env` file, then restart Streamlit (`python -m streamlit run app.py`).
2. Open `http://localhost:8501` → **Knowledge Base** and click **Verify Connection** — confirm it reports connected to `agentforge_ai`.
3. Upload and index a small `.txt` or `.pdf` file, click **Save to MongoDB**, then click **Clear Knowledge Base** (or reload the browser tab).
4. Click **Load from MongoDB** and ask a question about the document — confirm the reloaded citations (`doc | Chunk 1` or `doc | Page X | Chunk Y`) and **Sources** match identically.
5. Run one prompt in **Autonomous Agent (Tool Calling)**, then refresh the browser, open **Activity**, and click **Load Saved Activity from MongoDB** to confirm the run summary and tool log persisted.

## Project Structure

```text
agentforge-ai/
├── app.py            # Streamlit dashboard (Planning, Agent, Knowledge Base, Activity)
├── database.py       # MongoDB Atlas persistence wrapper (pymongo, connection check, CRUD)
├── rag.py            # Local in-memory PDF/TXT extraction, chunking, TF-IDF & Gemini RAG
├── agent.py          # Bounded Gemini function-calling agent loop (max 5 iterations)
├── tools.py          # Safe allowlisted Python tools (AST calculate, datetime, action plan)
├── planner.py        # Standalone Gemini AI planning service
├── test_app.py       # Streamlit AppTest UI tests (mocked Gemini & MongoDB)
├── test_database.py  # MongoDB Atlas persistence, reload parity & error-handling tests (mocked)
├── test_rag.py       # Document extraction, chunking, TF-IDF retrieval & RAG tests (mocked)
├── test_agent.py     # Safe tools & autonomous agent loop unit tests (mocked)
├── test_planner.py   # Planner unit tests (mocked)
├── requirements.txt  # Python dependencies
├── .env.example      # Template for local environment variables (no secrets)
├── .gitignore        # Excludes .venv, caches, .env, secrets, and local DB files
└── README.md         # Setup, architecture, and verification instructions
```
