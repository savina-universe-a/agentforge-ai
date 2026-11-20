# AgentForge AI

A clean, modular AI agent platform foundation built with Python and Streamlit.

**Milestone 1** establishes the core dashboard shell and project setup ahead of upcoming integrations:
- Official `google-genai` SDK for Gemini tool calling
- Supabase PostgreSQL + `pgvector` for retrieval-augmented generation (RAG)
- Optional multi-agent workflows with LangGraph when needed

## Prerequisites

- Windows 10/11
- Python 3.11 or newer

## Windows Setup (PowerShell)

1. **Create a virtual environment** in the repository root:
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

4. **Run the Streamlit app**:
   ```powershell
   python -m streamlit run app.py
   ```
   Streamlit will start a local server at `http://localhost:8501`.
   *(Using `python -m streamlit` also avoids Windows Application Control blocks on generated `.venv\Scripts\streamlit.exe` wrappers.)*

## Running Tests

Run the Streamlit `AppTest` verification suite:

```powershell
python -m unittest test_app.py -v
```

## Project Structure

```text
agentforge-ai/
├── app.py            # Streamlit dashboard application
├── test_app.py       # Streamlit AppTest UI verification tests
├── requirements.txt  # Minimal Python dependencies (Streamlit)
├── .env.example      # Template for future API/database environment variables
├── .gitignore        # Excludes .venv, caches, secrets, and local DB files
└── README.md         # Setup and usage instructions
```
