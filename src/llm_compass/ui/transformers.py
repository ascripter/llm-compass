"""Pure functions to translate between Streamlit UI data and the FastAPI schema."""

from typing import Any

_DEPLOYMENT_MAP = {
    "Any": "any",
    "Cloud API": "cloud",
    "Open Weights": "local",
}

_SPEED_MAP = {
    "Fast": "fast",
    "Medium+": "medium",
    "Any": None,
}

_REASONING_MAP = {
    "Standard+": "standard",
    "Native CoT": "native cot",
}

_TOOL_CALLING_MAP = {
    "Standard+": "standard",
    "Agentic": "agentic",
}


def sidebar_to_constraints(sidebar: dict[str, Any]) -> dict[str, Any]:
    """Convert the dict returned by render_sidebar() to the API Constraints format."""
    return {
        "min_context_window": sidebar.get("Min Context", 0),
        "modality_input": [m.lower() for m in sidebar.get("Inputs", ["text"])],
        "modality_output": [m.lower() for m in sidebar.get("Outputs", ["text"])],
        "deployment": _DEPLOYMENT_MAP.get(sidebar.get("Deployment", "Any"), "any"),
        "min_reasoning_type": _REASONING_MAP.get(sidebar.get("Reasoning", "Any")),
        "min_tool_calling": _TOOL_CALLING_MAP.get(sidebar.get("Tool Calling", "Any")),
        "min_speed_class": _SPEED_MAP.get(sidebar.get("Speed", "Any/Slow+"), None),
        "balanced_perf_weight": sidebar.get("Perf vs Cost", 50) / 100,
        "budget_perf_weight": sidebar.get("Budget Profile", 20) / 100,
    }


def response_to_display(api_response: dict[str, Any]) -> dict[str, Any]:
    """Flatten a QueryResponse dict into a shape the UI components can consume."""
    ui = api_response.get("ui_components") or {}
    traceability = api_response.get("traceability") or {}
    events = traceability.get("events", [])

    return {
        "session_id": api_response.get("session_id", ""),
        "status": api_response.get("status", "ok"),
        "clarification_question": api_response.get("clarification_question"),
        "trace_messages": [e.get("message", "") for e in events if isinstance(e, dict)],
        "debug_summary": api_response.get("debug_summary"),
        "summary_markdown": ui.get("summary_markdown"),
        "tier_tables": ui.get("tier_tables", []),
        "recommendation_cards": ui.get("recommendation_cards", []),
        "benchmarks_used": ui.get("benchmarks_used", []),
        "warnings": ui.get("warnings", []),
        "errors": api_response.get("errors", []),
    }
