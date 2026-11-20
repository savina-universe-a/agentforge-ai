# AgentForge AI

A clean, modular AI agent platform built with Python and Streamlit.

**Milestone 2** connects the official `google-genai` Python SDK so you can enter a goal and generate a prioritized **AI action plan** (planning only — autonomous tool execution, RAG, and multi-agent workflows are planned for later milestones).

## Prerequisites

- Windows 10/11
- Python 3.11 or newer
- A Google Gemini API key (from Google AI Studio) for live AI planning

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

4. **Configure `GEMINI_API_KEY` privately (choose either local method — never paste your key in chat or commit it)**:
   - **Option A — Local `.env` file (recommended):**
     ```powershell
     Copy-Item .env.example .env
     notepad .env
     ```
     Paste your key after `GEMINI_API_KEY=` and save. `.env` is already ignored by `.gitignore`.
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
| `GEMINI_API_KEY` | Yes (for live planning) | — | Your private Gemini API key |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Supported Gemini model used for AI planning |

## Running Automated Tests

All automated tests mock `google-genai` responses and do **not** make paid API calls:

```powershell
python -m unittest discover -v
```

## Project Structure

```text
agentforge-ai/
├── app.py            # Streamlit dashboard UI and private sidebar key setup
├── planner.py        # Gemini AI planning service (google-genai + python-dotenv)
├── test_app.py       # Streamlit AppTest UI verification tests (mocked Gemini)
├── test_planner.py   # Planner unit tests (mocked Gemini, dotenv, error handling)
├── requirements.txt  # Python dependencies (streamlit, google-genai, python-dotenv)
├── .env.example      # Template for local environment variables (no secrets)
├── .gitignore        # Excludes .venv, caches, .env, secrets, and local DB files
└── README.md         # Setup, configuration, and testing instructions
```
