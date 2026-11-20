"""Gemini AI planning service for AgentForge AI (Milestone 2)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv, set_key
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

DEFAULT_MODEL = "gemini-3.5-flash-lite"
ENV_FILE = Path(__file__).resolve().parent / ".env"

Priority = Literal["High", "Medium", "Low"]

SYSTEM_INSTRUCTION = (
    "You are an expert technical project planner inside AgentForge AI. "
    "Given a user goal, produce a clear, practical action plan with 3 to 7 ordered steps. "
    "Assign each step a realistic priority ('High', 'Medium', or 'Low') and concise, actionable guidance. "
    "Note that you are only creating a plan and are not executing external tools or actions."
)


class MissingAPIKeyError(ValueError):
    """Raised when GEMINI_API_KEY is not configured."""


class PlannerError(RuntimeError):
    """Raised when Gemini fails to generate or return a valid action plan."""


class PlanStep(BaseModel):
    """A single prioritized step in an AI-generated action plan."""

    step_number: int = Field(..., ge=1, description="Sequential step number starting at 1.")
    title: str = Field(..., min_length=1, description="Short, actionable step title.")
    description: str = Field(..., min_length=1, description="Concrete details for completing the step.")
    priority: Priority = Field(..., description="Priority level: High, Medium, or Low.")


class ActionPlan(BaseModel):
    """Structured action plan returned by Gemini."""

    summary: str = Field(..., min_length=1, description="Brief overview of the recommended approach.")
    steps: list[PlanStep] = Field(..., min_length=1, description="Ordered list of prioritized steps.")


def _sanitize_error_text(message: str, api_key: str | None) -> str:
    """Ensure an API key is never exposed in an error message."""
    if api_key and api_key in message:
        return message.replace(api_key, "[REDACTED]")
    return message


def get_api_key(override_key: str | None = None, env_path: Path | None = None) -> str | None:
    """Return the Gemini API key from a session override, environment, or local .env file."""
    if override_key and override_key.strip():
        return override_key.strip()

    target_env = env_path if env_path is not None else ENV_FILE
    if target_env.exists():
        load_dotenv(dotenv_path=target_env, override=False)

    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    return env_key or None


def get_model_name(env_path: Path | None = None) -> str:
    """Return the configured Gemini model name, defaulting to gemini-3.5-flash-lite."""
    target_env = env_path if env_path is not None else ENV_FILE
    if target_env.exists():
        load_dotenv(dotenv_path=target_env, override=False)

    configured = os.getenv("GEMINI_MODEL", "").strip()
    return configured or DEFAULT_MODEL


def save_api_key_to_env(api_key: str, env_path: Path | None = None) -> Path:
    """Privately save GEMINI_API_KEY to the local gitignored .env file."""
    cleaned_key = api_key.strip()
    if not cleaned_key:
        raise ValueError("API key cannot be empty.")
    if "\n" in cleaned_key or "\r" in cleaned_key:
        raise ValueError("API key must be a single line.")

    target_env = env_path if env_path is not None else ENV_FILE
    if not target_env.exists():
        target_env.touch()

    set_key(str(target_env), "GEMINI_API_KEY", cleaned_key)
    os.environ["GEMINI_API_KEY"] = cleaned_key
    return target_env


def generate_action_plan(
    goal: str,
    api_key: str | None = None,
    model: str | None = None,
    client: genai.Client | None = None,
    env_path: Path | None = None,
) -> ActionPlan:
    """Generate a prioritized action plan for a goal using the official google-genai SDK."""
    cleaned_goal = goal.strip()
    if not cleaned_goal:
        raise ValueError("Please enter a goal before generating an action plan.")

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
        # Pin vertexai=False so stray Vertex AI env vars cannot override Gemini Developer API mode.
        genai_client = (
            client
            if client is not None
            else genai.Client(api_key=resolved_key, vertexai=False)
        )
        response = genai_client.models.generate_content(
            model=resolved_model,
            contents=f"Create a prioritized action plan for the following goal:\n\n{cleaned_goal}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=ActionPlan,
                temperature=0.3,
            ),
        )
    except genai_errors.APIError as exc:
        status_code = getattr(exc, "code", None)
        if status_code in (401, 403):
            raise PlannerError(
                "Gemini rejected the API key (401/403). Check that GEMINI_API_KEY is valid and active."
            ) from exc
        if status_code == 429:
            raise PlannerError(
                "Gemini API rate limit or quota was exceeded (429). Wait a moment or check your billing/quota settings."
            ) from exc
        if status_code == 404:
            raise PlannerError(
                f"Gemini model '{resolved_model}' was not found or is unavailable (404). "
                f"Set GEMINI_MODEL in .env to an available model such as '{DEFAULT_MODEL}'."
            ) from exc
        if status_code == 503:
            raise PlannerError(
                f"Gemini model '{resolved_model}' is temporarily experiencing high demand (503). "
                "Please try again shortly or switch GEMINI_MODEL in .env."
            ) from exc

        safe_detail = _sanitize_error_text(str(exc), resolved_key)
        raise PlannerError(f"Gemini API request failed: {safe_detail}") from exc
    except Exception as exc:  # noqa: BLE001
        safe_detail = _sanitize_error_text(str(exc), resolved_key)
        raise PlannerError(f"Unexpected error while calling Gemini: {safe_detail}") from exc

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, ActionPlan):
        return parsed

    raw_text = (getattr(response, "text", None) or "").strip()
    if not raw_text:
        raise PlannerError("Gemini returned an empty response. Please try again.")

    try:
        return ActionPlan.model_validate_json(raw_text)
    except ValidationError as exc:
        raise PlannerError(
            "Gemini returned a response that could not be parsed into a structured action plan."
        ) from exc
