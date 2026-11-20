"""Bounded autonomous Gemini tool-calling agent for AgentForge AI (Milestone 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

from planner import (
    DEFAULT_MODEL,
    MissingAPIKeyError,
    _sanitize_error_text,
    get_api_key,
    get_model_name,
)
from tools import (
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    execute_tool,
    get_gemini_tools,
    summarize_tool_args,
    summarize_tool_result,
)

MAX_TOOL_ITERATIONS = 5
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0

AGENT_SYSTEM_INSTRUCTION = (
    "You are AgentForge AI, an autonomous tool-calling assistant. "
    "Use the provided safe tools whenever they help answer the user's request accurately: "
    "'calculate' for exact arithmetic, 'get_current_datetime' for the current date and time, "
    "and 'create_action_plan' for structured planning steps. "
    "You may call tools across multiple steps when needed. "
    "Once you have the information required, provide a clear, well-structured final response. "
    "If a tool returns an error, adapt cleanly or explain the limitation. "
    "Never claim to run shell commands, browse the web, or modify local files."
)


class AgentError(RuntimeError):
    """Raised when the autonomous agent encounters a Gemini API, timeout, or runtime error."""


class ToolActivityEntry(BaseModel):
    """Record of one tool invocation during an autonomous agent run."""

    iteration: int = Field(..., ge=1)
    tool_name: str
    input_summary: str
    status: Literal["success", "error"]
    result_summary: str
    result_data: dict[str, Any] = Field(default_factory=dict)


class AgentRunResult(BaseModel):
    """Final output and tool activity log of a bounded autonomous agent run."""

    goal: str
    model: str
    final_response: str
    activities: list[ToolActivityEntry] = Field(default_factory=list)
    iterations_used: int = 0
    max_iterations: int = MAX_TOOL_ITERATIONS
    stopped_reason: Literal["completed", "max_iterations_reached"] = "completed"


def _extract_visible_text(response: Any) -> str:
    """Extract only visible model text parts, excluding any hidden reasoning/thought parts."""
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        visible_chunks = [
            part.text
            for part in parts
            if getattr(part, "text", None) and not getattr(part, "thought", False)
        ]
        if visible_chunks:
            return "\n".join(chunk.strip() for chunk in visible_chunks if chunk.strip()).strip()

    fallback_text = getattr(response, "text", None)
    return fallback_text.strip() if isinstance(fallback_text, str) else ""


def _build_model_turn_content(response: Any, function_calls: list[Any]) -> types.Content:
    """Preserve the model's exact Content (including Gemini 3.x thought signatures) when available."""
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        candidate_content = getattr(candidates[0], "content", None)
        if candidate_content is not None and getattr(candidate_content, "parts", None):
            return candidate_content

    rebuilt_parts = [
        types.Part.from_function_call(
            name=getattr(call, "name", "") or "unknown_tool",
            args=dict(getattr(call, "args", None) or {}),
        )
        for call in function_calls
    ]
    return types.Content(role="model", parts=rebuilt_parts)


def _build_function_response_part(call: Any, tool_payload: dict[str, Any]) -> types.Part:
    """Build a FunctionResponse Part carrying tool execution output back to Gemini."""
    call_name = (getattr(call, "name", None) or "unknown_tool").strip() or "unknown_tool"
    call_id = getattr(call, "id", None)
    if call_id:
        return types.Part(
            function_response=types.FunctionResponse(
                id=call_id,
                name=call_name,
                response=tool_payload,
            )
        )
    return types.Part.from_function_response(name=call_name, response=tool_payload)


def _format_api_error(exc: genai_errors.APIError, resolved_model: str, resolved_key: str | None) -> str:
    """Convert a google-genai APIError into a clear, secret-free error message."""
    status_code = getattr(exc, "code", None)
    if status_code in (401, 403):
        return "Gemini rejected the API key (401/403). Check that GEMINI_API_KEY is valid and active."
    if status_code == 429:
        return "Gemini API rate limit or quota was exceeded (429). Wait a moment or check your quota."
    if status_code == 404:
        return (
            f"Gemini model '{resolved_model}' was not found or is unavailable (404). "
            f"Set GEMINI_MODEL in .env to an available model such as '{DEFAULT_MODEL}'."
        )
    if status_code == 503:
        return (
            f"Gemini model '{resolved_model}' is temporarily experiencing high demand (503). "
            "Please try again shortly or switch GEMINI_MODEL in .env."
        )

    safe_detail = _sanitize_error_text(str(exc), resolved_key)
    return f"Gemini API request failed: {safe_detail}"


def run_autonomous_agent(
    goal: str,
    api_key: str | None = None,
    model: str | None = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    client: genai.Client | None = None,
    env_path: Path | None = None,
    tool_registry: dict[str, Callable[..., dict[str, Any]]] | None = None,
) -> AgentRunResult:
    """Run a bounded autonomous Gemini tool-calling loop (up to 5 tool iterations)."""
    cleaned_goal = goal.strip()
    if not cleaned_goal:
        raise ValueError("Please enter a goal before running the autonomous agent.")

    if max_iterations < 1 or max_iterations > MAX_TOOL_ITERATIONS:
        raise ValueError(f"max_iterations must be between 1 and {MAX_TOOL_ITERATIONS}.")

    resolved_key: str | None = None
    if client is None:
        resolved_key = get_api_key(override_key=api_key, env_path=env_path)
        if not resolved_key:
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is not set. Add it to your local .env file or enter it "
                "privately in the sidebar under API Configuration."
            )

    resolved_model = (model or get_model_name(env_path=env_path)).strip() or DEFAULT_MODEL

    try:
        genai_client = (
            client
            if client is not None
            else genai.Client(
                api_key=resolved_key,
                vertexai=False,
                http_options=types.HttpOptions(timeout=int(request_timeout_seconds * 1000)),
            )
        )
    except Exception as exc:  # noqa: BLE001
        safe_detail = _sanitize_error_text(str(exc), resolved_key)
        raise AgentError(f"Failed to initialize Gemini client: {safe_detail}") from exc

    config = types.GenerateContentConfig(
        system_instruction=AGENT_SYSTEM_INSTRUCTION,
        tools=get_gemini_tools(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.2,
    )

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=cleaned_goal)])
    ]
    activities: list[ToolActivityEntry] = []
    iterations_used = 0

    while True:
        try:
            response = genai_client.models.generate_content(
                model=resolved_model,
                contents=contents,
                config=config,
            )
        except genai_errors.APIError as exc:
            raise AgentError(_format_api_error(exc, resolved_model, resolved_key)) from exc
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise AgentError(
                f"Gemini request timed out after {request_timeout_seconds:.0f}s. Please try again."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            safe_detail = _sanitize_error_text(str(exc), resolved_key)
            raise AgentError(f"Unexpected error while running the agent: {safe_detail}") from exc

        function_calls = list(getattr(response, "function_calls", None) or [])
        if not function_calls:
            final_text = _extract_visible_text(response)
            if not final_text:
                raise AgentError("Gemini returned an empty final response. Please try again.")
            return AgentRunResult(
                goal=cleaned_goal,
                model=resolved_model,
                final_response=final_text,
                activities=activities,
                iterations_used=iterations_used,
                max_iterations=max_iterations,
                stopped_reason="completed",
            )

        if iterations_used >= max_iterations:
            partial_text = _extract_visible_text(response)
            limit_notice = (
                f"Stopped after reaching the safety limit of {max_iterations} tool iteration(s). "
                "Review the activity log above for completed tool results or try a narrower request."
            )
            final_text = f"{partial_text}\n\n{limit_notice}".strip() if partial_text else limit_notice
            return AgentRunResult(
                goal=cleaned_goal,
                model=resolved_model,
                final_response=final_text,
                activities=activities,
                iterations_used=iterations_used,
                max_iterations=max_iterations,
                stopped_reason="max_iterations_reached",
            )

        iterations_used += 1
        contents.append(_build_model_turn_content(response, function_calls))

        response_parts: list[types.Part] = []
        for call in function_calls:
            call_name = (getattr(call, "name", None) or "").strip() or "unknown_tool"
            raw_args = getattr(call, "args", None)
            args_dict = dict(raw_args) if isinstance(raw_args, dict) else {}

            tool_payload = execute_tool(
                tool_name=call_name,
                raw_args=args_dict,
                timeout_seconds=tool_timeout_seconds,
                registry=tool_registry,
            )

            status: Literal["success", "error"] = (
                "success" if tool_payload.get("status") == "success" else "error"
            )
            activities.append(
                ToolActivityEntry(
                    iteration=iterations_used,
                    tool_name=call_name,
                    input_summary=summarize_tool_args(args_dict),
                    status=status,
                    result_summary=summarize_tool_result(call_name, tool_payload),
                    result_data=tool_payload,
                )
            )
            response_parts.append(_build_function_response_part(call, tool_payload))

        contents.append(types.Content(role="user", parts=response_parts))
