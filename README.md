# AgentForge AI

A clean, modular AI agent platform built with Python, Streamlit, and the official `google-genai` SDK.

**Milestone 4** adds a local **Document RAG Knowledge Base** (`pypdf` + `scikit-learn` TF-IDF + Gemini grounded Q&A) alongside Milestone 2's **AI Planning** and Milestone 3's **Autonomous Tool-Calling Agent** (no vector database, Supabase, LangChain, or LangGraph yet).

## Features by Milestone

- **Milestone 1 — Dashboard Foundation:** Minimal Streamlit UI with `Overview`, `AI Agent`, `Knowledge Base`, and `Activity` navigation and baseline metrics.
- **Milestone 2 — Gemini AI Planning:** Generates structured, prioritized action plans using `google-genai` and `python-dotenv` with private local `GEMINI_API_KEY` setup.
- **Milestone 3 — Autonomous Tool-Calling Agent:** Runs a bounded (max 5 iterations) Gemini function-calling loop over three safe allowlisted Python tools (`calculate` via restricted AST parsing, `get_current_datetime`, and `create_action_plan`), displaying a transparent **Activity Log** and **Final Response**.
- **Milestone 4 — Local Document RAG Knowledge Base:**
  - Upload **PDF** and **TXT** files (up to 5 MB each; rejects empty, encrypted, binary, or unreadable files).
  - Extracts text strictly in memory (`pypdf` for PDFs with page numbers preserved; UTF-8 for TXT) — **never writes uploaded documents to disk or Git**.
  - Splits text into overlapping chunks with deterministic source IDs (`doc.pdf | Page 2 | Chunk 1` or `notes.txt | Chunk 1`).
  - Indexes chunks locally in Streamlit session state using `scikit-learn` **TF-IDF + cosine similarity**.
  - Retrieves only the top matching passages and instructs Gemini to answer **strictly from retrieved context**, cite sources inline, treat document content as untrusted data (prompt-injection protection), and explicitly say when documents do not contain the answer.
  - Displays retrieved passages inside an expandable **Sources** section and provides a **Clear Knowledge Base** button.

## Prerequisites

- Windows 10/11
- Python 3.11 or newer
- A Google Gemini API key (from Google AI Studio) for live runs

## Windows Setup (PowerShell)

1. **Create a virtual environment** in the repository root (skip if `.venv` already exists):
   ```powershell
   python -m venv .venv
   ```
   *(If you use the Python Launcher on Windows, you can run `py -3.11 -m venv .venv`.)*

2. **Activate the virtual environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   *(If PowerShell blocks script execution, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and retry.)*

3. **Install dependencies**:
   ```powershell
   python -m pip install -r requirements.txt
   ```

4. **Configure `GEMINI_API_KEY` privately (never paste your key in chat or commit it)**:
   - **Option A — Local `.env` file (recommended):**
     ```powershell
     Copy-Item .env.example .env
     notepad .env
     ```
     Paste your key after `GEMINI_API_KEY=` and save. `.env` is ignored by `.gitignore`.
   - **Option B — Private sidebar input in the app:**
     Start the app, open **API Key Setup (Private)** in the sidebar, paste your masked key, and click **Use Session** (memory only) or **Save to .env** (writes to your gitignored local `.env`).

5. **Run the Streamlit app**:
   ```powershell
   python -m streamlit run app.py
   ```
   Streamlit starts a local server at `http://localhost:8501`.
   *(Using `python -m streamlit` also avoids Windows Application Control blocks on generated `.venv\Scripts\streamlit.exe` wrappers.)*

## Configuration

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes (for live runs) | — | Your private Gemini API key |
| `GEMINI_MODEL` | No | `gemini-3.5-flash-lite` | Available Gemini model used for AI planning, agent tool calling, and RAG Q&A |

## How to Test

### 1. Automated Tests (Mocked — Zero API Credits)

All automated tests mock `google-genai` responses and verify Milestones 1–4 without making live API calls:

```powershell
python -m unittest discover -v
```

### 2. Live Browser Tests

- **Document RAG (`Knowledge Base` tab):**
  1. Upload a small `.txt` or `.pdf` file (e.g., project notes or a short policy document) and click **Index Uploaded Documents**.
  2. Ask a question answered by the file and click **Ask Knowledge Base** — verify inline citations and expand **Sources** to inspect the retrieved passages.
  3. Ask a completely unrelated question — verify the app explicitly states that the uploaded documents do not contain the answer.
  4. Click **Clear Knowledge Base** to remove all documents and the TF-IDF index from session memory.
- **Autonomous Agent (`Overview` or `AI Agent` tab):**
  Enter a multi-step prompt (e.g., check today's date, calculate a cost expression, and create an action plan) and click **Run Autonomous Agent**.

## Project Structure

```text
agentforge-ai/
├── app.py            # Streamlit dashboard (AI Planning, Autonomous Agent, Knowledge Base, Activity)
├── rag.py            # Local in-memory PDF/TXT extraction, overlapping chunking, TF-IDF & Gemini RAG
├── agent.py          # Bounded Gemini function-calling agent loop (max 5 iterations)
├── tools.py          # Safe allowlisted Python tools (AST calculate, datetime, action plan)
├── planner.py        # Standalone Gemini AI planning service
├── test_app.py       # Streamlit AppTest UI verification tests (mocked Gemini)
├── test_rag.py       # Document extraction, chunking, TF-IDF retrieval & RAG tests (mocked Gemini)
├── test_agent.py     # Safe tools & autonomous agent loop unit tests (mocked Gemini)
├── test_planner.py   # Planner unit tests (mocked Gemini, dotenv, error handling)
├── requirements.txt  # Python dependencies (streamlit, google-genai, python-dotenv, pypdf, scikit-learn)
├── .env.example      # Template for local environment variables (no secrets)
├── .gitignore        # Excludes .venv, caches, .env, secrets, and local DB files
└── README.md         # Setup, architecture, and testing instructions
```
