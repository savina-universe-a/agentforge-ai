# AgentForge AI

A clean, modular AI agent platform built with Python, Streamlit, `google-genai`, MongoDB Atlas, local TF-IDF RAG, and semantic Vector Search RAG.

**Milestone 6** adds **Semantic RAG** using Gemini embeddings (`gemini-embedding-001`, 768 dimensions, L2-normalized) and **MongoDB Atlas Vector Search** (`$vectorSearch` on `document_chunks`), while keeping local TF-IDF retrieval as a transparent, explicit fallback.

## Features by Milestone

- **Milestone 1 — Dashboard Foundation:** Minimal Streamlit UI with `Overview`, `AI Agent`, `Knowledge Base`, and `Activity` navigation and baseline metrics.
- **Milestone 2 — Gemini AI Planning:** Generates structured, prioritized action plans using `google-genai` and `python-dotenv` with private local `GEMINI_API_KEY` setup.
- **Milestone 3 — Autonomous Tool-Calling Agent:** Bounded (max 5 iterations) Gemini function-calling loop over three safe allowlisted Python tools (`calculate` via restricted AST parsing, `get_current_datetime`, and `create_action_plan`), showing an **Activity Log** and **Final Response**.
- **Milestone 4 — Local Document RAG Knowledge Base:** In-memory PDF (`pypdf`) and TXT extraction with overlapping chunking, deterministic source IDs, local `scikit-learn` TF-IDF retrieval, prompt-injection defenses, inline citations, and expandable **Sources**.
- **Milestone 5 — MongoDB Atlas Persistence (`pymongo`):** Persists Knowledge Base documents (`documents` collection) and autonomous agent run logs (`agent_runs` collection) under stable IDs with duplicate detection and session-only fallback.
- **Milestone 6 — Semantic RAG (Gemini Embeddings + Atlas Vector Search):**
  - Embeds chunks using `gemini-embedding-001` (`output_dimensionality=768`, L2-normalized to unit norm) with `RETRIEVAL_DOCUMENT` for document chunks and `RETRIEVAL_QUERY` for user questions.
  - Stores chunk embeddings in a dedicated `document_chunks` collection retaining `doc_id`, `chunk_id`, `doc_name`, `page_number`, `citation_label`, and `text`.
  - **Idempotent & cost-conscious:** Only runs when you click **Index Embeddings to Atlas**, batches in groups of 16, skips unchanged chunks via SHA-256 hashing, and removes stale orphan chunks when a document is updated.
  - Adds **Semantic Search (Atlas Vector)** mode alongside **Local TF-IDF** mode, reusing the prompt-injection-hardened grounded Gemini answer pipeline and showing an explicit warning banner whenever TF-IDF fallback is used.

## Prerequisites

- Windows 10/11
- Python 3.11 or newer
- A Google Gemini API key (from Google AI Studio)
- *(Optional for Milestones 5–6)* A MongoDB Atlas cluster with Vector Search enabled

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
| `GEMINI_MODEL` | No | `gemini-3.5-flash-lite` | Gemini model for planning, agent tool calling, and grounded RAG answers |
| `GEMINI_EMBEDDING_MODEL` | No | `gemini-embedding-001` | Gemini embedding model for Semantic RAG |
| `GEMINI_EMBEDDING_DIMENSIONS` | No | `768` | Output dimensionality (L2-normalized) matching the Atlas Vector index |
| `MONGODB_URI` | No | — | MongoDB Atlas SRV URI (`mongodb+srv://...`); omit for session-only mode |
| `MONGODB_DB_NAME` | No | `agentforge_ai` | MongoDB database name (`documents`, `document_chunks`, `agent_runs`) |
| `MONGODB_VECTOR_INDEX_NAME` | No | `chunk_embedding_index` | Atlas Vector Search index name on `document_chunks` |

## MongoDB Atlas Vector Search Index Setup

1. First upload/index at least one document in **Knowledge Base** and click **Index Embeddings to Atlas** (this creates the `agentforge_ai.document_chunks` collection automatically).
2. In the MongoDB Atlas UI, open your cluster → **Atlas Search** (or **Search & Vector Search**) → **Create Search Index** → **Atlas Vector Search** → **JSON Editor**.
3. Select database **`agentforge_ai`** and collection **`document_chunks`**.
4. Set the **Index Name** to **`chunk_embedding_index`**.
5. Paste this exact JSON definition (matching `gemini-embedding-001` at `768` dimensions):
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
6. Click **Create Vector Search Index** and wait until its status becomes **Active / READY** (usually 15–60 seconds). Click **Check Vector Index** in AgentForge AI to confirm readiness.

## How to Verify & Test

### 1. Automated Tests (Mocked — Zero Live API or Atlas Calls)

```powershell
python -m unittest discover -v
```

### 2. Live Semantic Search Test

1. Ensure `GEMINI_API_KEY` and `MONGODB_URI` are set in your local `.env`, then open `http://localhost:8501` → **Knowledge Base**.
2. Upload a small `.txt` or `.pdf` file, click **Index Uploaded Documents**, and click **Index Embeddings to Atlas**.
   - Click **Index Embeddings to Atlas** a second time to verify idempotency (`embedded 0 new/changed, skipped N unchanged`).
3. Create the `chunk_embedding_index` index in MongoDB Atlas using the JSON above, then click **Check Vector Index** until it reports `READY`.
4. Under **Ask a Question**, select **Semantic Search (Atlas Vector)** and ask a paraphrased question using synonyms rather than exact words from the document (for example, ask *"How quickly do you respond to urgent outages?"* for a document mentioning *"Priority 1 incidents receive an initial response within 2 hours"*).
5. Verify the answer shows `Retrieval: Atlas Vector Search (chunk_embedding_index, gemini-embedding-001)`, cites the exact document/page/chunk, and displays the retrieved passage in **Sources**.

## Project Structure

```text
agentforge-ai/
├── app.py               # Streamlit dashboard (Planning, Agent, TF-IDF & Semantic RAG, Activity)
├── semantic_rag.py      # Gemini embeddings (768d L2-normalized) & MongoDB Atlas Vector Search
├── database.py          # MongoDB Atlas persistence wrapper (documents & agent_runs)
├── rag.py               # PDF/TXT extraction, overlapping chunking, TF-IDF & grounded Q&A
├── agent.py             # Bounded Gemini function-calling agent loop (max 5 iterations)
├── tools.py             # Safe allowlisted Python tools (AST calculate, datetime, action plan)
├── planner.py           # Standalone Gemini AI planning service
├── test_app.py          # Streamlit AppTest UI tests (mocked Gemini & MongoDB)
├── test_semantic_rag.py # Embedding, dimension, idempotent indexing & Vector Search tests (mocked)
├── test_database.py     # MongoDB persistence, reload parity & error-handling tests (mocked)
├── test_rag.py          # Extraction, chunking, TF-IDF retrieval & grounded RAG tests (mocked)
├── test_agent.py        # Safe tools & autonomous agent loop unit tests (mocked)
├── test_planner.py      # Planner unit tests (mocked)
├── requirements.txt     # Python dependencies
├── .env.example         # Template for local environment variables (no secrets)
├── .gitignore           # Excludes .venv, caches, .env, secrets, and local DB files
└── README.md            # Setup, Atlas Vector Search guide, and testing instructions
```
