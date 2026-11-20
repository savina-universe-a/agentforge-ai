# AgentForge AI

A clean, modular AI agent platform built with Python, Streamlit, and the official `google-genai` SDK.

**Milestone 3** adds a genuine **autonomous Gemini tool-calling agent** alongside Milestone 2's **AI Planning** feature (no LangGraph, database, or RAG yet).

## What Makes This a Real Autonomous Agent?

Unlike a single prompt-and-response chat completion (or Milestone 2's standalone **AI Planning**), the **Autonomous Agent** in Milestone 3 runs a genuine **observe → decide → act → observe** tool-calling loop powered by Gemini function calling:

1. **Tool Selection:** Gemini receives your goal plus structured schemas (`FunctionDeclaration`) for three allowlisted Python tools and autonomously decides whether any tool is needed — and which arguments to pass.
2. **Safe Local Execution:** Python validates and executes the requested tool locally (with timeout protection and no `eval`/`exec`, shell commands, web browsing, or file-system edits):
   - `calculate(expression)` — Evaluates basic arithmetic via a restricted Python `ast` parser.
   - `get_current_datetime()` — Returns the current local date, time, weekday, timezone, and UTC timestamp via `datetime`.
   - `create_action_plan(goal)` — Returns structured, prioritized planning steps for a goal.
3. **Feedback Loop:** Python sends each tool result back to Gemini as a `FunctionResponse` part. Gemini inspects the real execution output and decides whether to call another tool or synthesize the final answer.
4. **Safety Bounds & Transparency:** The loop is capped at **5 tool iterations**, invalid tool calls and timeouts are handled gracefully, and every run displays a clear **Activity Log** (selected tool, safe input summary, execution result, and final response — never API keys or hidden reasoning).

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

4. **Configure `GEMINI_API_KEY` privately (choose either local method — never paste your key in chat or commit it)**:
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
| `GEMINI_MODEL` | No | `gemini-3.5-flash-lite` | Available Gemini model used for AI planning and tool calling |

## How to Test

### 1. Automated Tests (Mocked — Zero API Credits)

All automated tests mock `google-genai` responses and verify UI rendering, safe tools, tool selection, tool-result feedback, multi-step execution, the 5-iteration safety limit, timeouts, invalid tool calls, and API error handling:

```powershell
python -m unittest discover -v
```

### 2. Live Autonomous Agent Test (In the Browser)

1. Open `http://localhost:8501` with your `GEMINI_API_KEY` configured locally.
2. In **Overview → Autonomous Agent (Tool Calling)** (or the **AI Agent** sidebar tab), enter a multi-step prompt such as:
   > Check today's date, calculate the total cost of 6 workshop seats at 48 dollars each plus a 35 dollar room fee, and create a prioritized action plan to prepare the workshop.
3. Click **Run Autonomous Agent**.
4. Confirm the **Activity Log** shows Gemini calling `get_current_datetime`, `calculate`, and `create_action_plan` with safe inputs and execution results, followed by a synthesized **Final Response**.

## Project Structure

```text
agentforge-ai/
├── app.py            # Streamlit dashboard (AI Planning + Autonomous Agent + Activity Log)
├── agent.py          # Bounded Gemini function-calling agent loop (max 5 iterations)
├── tools.py          # Safe allowlisted Python tools (AST calculate, datetime, action plan)
├── planner.py        # Milestone 2 standalone Gemini AI planning service
├── test_app.py       # Streamlit AppTest UI verification tests (mocked Gemini)
├── test_agent.py     # Safe tools & autonomous agent loop unit tests (mocked Gemini)
├── test_planner.py   # Planner unit tests (mocked Gemini, dotenv, error handling)
├── requirements.txt  # Python dependencies (streamlit, google-genai, python-dotenv)
├── .env.example      # Template for local environment variables (no secrets)
├── .gitignore        # Excludes .venv, caches, .env, secrets, and local DB files
└── README.md         # Setup, architecture, and testing instructions
```
