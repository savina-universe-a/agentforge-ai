"""Safe Python tools and Gemini function declarations for AgentForge AI (Milestone 3)."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
import math
import operator
from typing import Any, Callable

from google.genai import types

MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 50
MAX_EXPONENT = 100
MAX_POWER_BASE = 1_000_000
DEFAULT_TOOL_TIMEOUT_SECONDS = 5.0

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_arithmetic_node(node: ast.AST) -> int | float:
    """Safely evaluate a restricted arithmetic AST node (no eval/exec)."""
    if isinstance(node, ast.Expression):
        return _eval_arithmetic_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric literals are allowed in arithmetic expressions.")
        if not math.isfinite(node.value):
            raise ValueError("Expression contains a non-finite number.")
        return node.value

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError("Unsupported unary operator.")
        return op_func(_eval_arithmetic_node(node.operand))

    if isinstance(node, ast.BinOp):
        op_func = _BINARY_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError("Unsupported binary operator.")

        left = _eval_arithmetic_node(node.left)
        right = _eval_arithmetic_node(node.right)

        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("Division by zero is not allowed.")

        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT or abs(left) > MAX_POWER_BASE:
                raise ValueError(
                    f"Exponentiation exceeds safe limits (|base| <= {MAX_POWER_BASE}, |exponent| <= {MAX_EXPONENT})."
                )

        result = op_func(left, right)
        if not isinstance(result, (int, float)) or not math.isfinite(result):
            raise ValueError("Calculation resulted in a non-finite value.")
        return result

    raise ValueError(
        "Unsupported expression syntax. Only numbers, parentheses, and +, -, *, /, //, %, ** are allowed."
    )


def calculate(expression: str) -> dict[str, Any]:
    """Evaluate a basic arithmetic expression using a restricted AST parser."""
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")

    cleaned = expression.strip()
    if not cleaned:
        raise ValueError("Expression cannot be empty.")
    if len(cleaned) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Expression exceeds maximum length of {MAX_EXPRESSION_LENGTH} characters.")

    try:
        parsed = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid arithmetic syntax.") from exc

    if sum(1 for _ in ast.walk(parsed)) > MAX_AST_NODES:
        raise ValueError("Expression is too complex.")

    value = _eval_arithmetic_node(parsed)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, float):
        value = round(value, 10)

    return {
        "expression": cleaned,
        "result": value,
    }


def get_current_datetime() -> dict[str, Any]:
    """Return the current local and UTC date/time using Python's datetime."""
    now_local = datetime.now().astimezone()
    now_utc = now_local.astimezone(timezone.utc)
    return {
        "local_datetime": now_local.isoformat(timespec="seconds"),
        "local_date": now_local.strftime("%Y-%m-%d"),
        "local_time": now_local.strftime("%H:%M:%S"),
        "weekday": now_local.strftime("%A"),
        "timezone": now_local.tzname() or "local",
        "utc_datetime": now_utc.isoformat(timespec="seconds"),
    }


def create_action_plan(goal: str) -> dict[str, Any]:
    """Return a structured, prioritized action plan scaffold for a goal."""
    if not isinstance(goal, str):
        raise ValueError("Goal must be a string.")

    cleaned_goal = goal.strip()
    if not cleaned_goal:
        raise ValueError("Goal cannot be empty.")
    if len(cleaned_goal) > 500:
        cleaned_goal = cleaned_goal[:500].rstrip() + "..."

    steps = [
        {
            "step_number": 1,
            "title": "Clarify scope and success criteria",
            "description": f"Define concrete deliverables, constraints, and deadline for: {cleaned_goal}",
            "priority": "High",
        },
        {
            "step_number": 2,
            "title": "Break work into prioritized tasks",
            "description": "Split the goal into small, verifiable milestones ordered by dependency and impact.",
            "priority": "High",
        },
        {
            "step_number": 3,
            "title": "Execute core deliverable",
            "description": "Focus on completing the highest-priority milestone first while tracking blockers.",
            "priority": "Medium",
        },
        {
            "step_number": 4,
            "title": "Verify results and review risks",
            "description": "Check outputs against the original criteria and resolve any gaps before finishing.",
            "priority": "Medium",
        },
        {
            "step_number": 5,
            "title": "Document outcomes and next steps",
            "description": "Record final notes, lessons learned, and follow-up actions for future reference.",
            "priority": "Low",
        },
    ]

    return {
        "goal": cleaned_goal,
        "summary": f"Structured 5-step prioritized plan for: {cleaned_goal}",
        "steps": steps,
    }


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "calculate": calculate,
    "get_current_datetime": get_current_datetime,
    "create_action_plan": create_action_plan,
}

_REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "calculate": ("expression",),
    "get_current_datetime": (),
    "create_action_plan": ("goal",),
}


def get_gemini_tools() -> list[types.Tool]:
    """Build official google-genai FunctionDeclaration schemas for the safe tools."""
    declarations = [
        types.FunctionDeclaration(
            name="calculate",
            description=(
                "Evaluate a safe arithmetic expression using numbers, parentheses, and "
                "basic operators (+, -, *, /, //, %, **). Use this for exact math calculations."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate, e.g. '(120 * 4) + 35'.",
                    }
                },
                "required": ["expression"],
            },
        ),
        types.FunctionDeclaration(
            name="get_current_datetime",
            description=(
                "Get the current local date, time, weekday, timezone, and UTC timestamp. "
                "Use this whenever the user asks about today's date, current time, or schedules relative to now."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {},
            },
        ),
        types.FunctionDeclaration(
            name="create_action_plan",
            description=(
                "Create a structured, prioritized set of planning steps for a goal. "
                "Use this when the user asks for a plan, roadmap, or structured task breakdown."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The goal or project to break into prioritized action steps.",
                    }
                },
                "required": ["goal"],
            },
        ),
    ]
    return [types.Tool(function_declarations=declarations)]


def summarize_tool_args(args: dict[str, Any] | None) -> str:
    """Create a concise, safe summary of tool input arguments for the activity log."""
    if not args:
        return "(no arguments)"

    parts: list[str] = []
    for key, value in args.items():
        key_str = str(key)
        if any(token in key_str.lower() for token in ("key", "secret", "token", "password")):
            parts.append(f"{key_str}=[REDACTED]")
            continue

        value_str = repr(value)
        if len(value_str) > 120:
            value_str = value_str[:117] + "..."
        parts.append(f"{key_str}={value_str}")
    return ", ".join(parts)


def summarize_tool_result(tool_name: str, payload: dict[str, Any]) -> str:
    """Create a concise, human-readable summary of a tool execution result."""
    if payload.get("status") == "error":
        return f"Error: {payload.get('error', 'Tool execution failed.')}"

    output = payload.get("output", {})
    if tool_name == "calculate" and isinstance(output, dict):
        return f"{output.get('expression')} = {output.get('result')}"
    if tool_name == "get_current_datetime" and isinstance(output, dict):
        return (
            f"{output.get('weekday')}, {output.get('local_date')} "
            f"{output.get('local_time')} ({output.get('timezone')})"
        )
    if tool_name == "create_action_plan" and isinstance(output, dict):
        steps = output.get("steps", [])
        return f"Created {len(steps)} prioritized planning steps for '{output.get('goal')}'."

    summary = str(output)
    return summary if len(summary) <= 200 else summary[:197] + "..."


def execute_tool(
    tool_name: str | None,
    raw_args: dict[str, Any] | None,
    timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    registry: dict[str, Callable[..., dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Execute an allowlisted Python tool safely with validation and timeout protection."""
    active_registry = registry if registry is not None else TOOL_REGISTRY
    normalized_name = (tool_name or "").strip()

    if not normalized_name or normalized_name not in active_registry:
        allowed = ", ".join(sorted(active_registry.keys()))
        return {
            "status": "error",
            "error": f"Unknown or disallowed tool '{normalized_name}'. Allowed tools: {allowed}.",
        }

    args: dict[str, Any] = dict(raw_args) if isinstance(raw_args, dict) else {}
    required = _REQUIRED_ARGS.get(normalized_name, ())
    allowed_arg_names = set(required)
    unexpected = sorted(set(args.keys()) - allowed_arg_names)
    if unexpected:
        return {
            "status": "error",
            "error": f"Unexpected argument(s) for '{normalized_name}': {', '.join(unexpected)}.",
        }

    missing = [name for name in required if name not in args or args[name] is None]
    if missing:
        return {
            "status": "error",
            "error": f"Missing required argument(s) for '{normalized_name}': {', '.join(missing)}.",
        }

    tool_fn = active_registry[normalized_name]
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(tool_fn, **args)
    try:
        output = future.result(timeout=timeout_seconds)
        return {
            "status": "success",
            "output": output,
        }
    except FuturesTimeoutError:
        future.cancel()
        return {
            "status": "error",
            "error": f"Tool '{normalized_name}' timed out after {timeout_seconds:.1f}s.",
        }
    except ValueError as exc:
        return {
            "status": "error",
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": f"Tool '{normalized_name}' failed: {type(exc).__name__}: {exc}",
        }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
