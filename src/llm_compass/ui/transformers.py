"""Pure functions to translate between Streamlit UI data and the FastAPI schema."""

from typing import Any

_DEPLOYMENT_MAP = {
    "Any": "any",
    "Cloud API": "cloud",
    "Local / Open Weights": "local",
}

_SPEED_MAP = {
    "Fast only": "fast",
    "Balanced+": "medium",
    "Any/Slow+": None,
}


def sidebar_to_constraints(sidebar: dict[str, Any]) -> dict[str, Any]:
    """Convert the dict returned by render_sidebar() to the API Constraints format."""
    return {
        "min_context_window": sidebar.get("Min Context", 0),
        "modality_input": [m.lower() for m in sidebar.get("Inputs", ["text"])],
        "modality_output": [m.lower() for m in sidebar.get("Outputs", ["text"])],
        "deployment": _DEPLOYMENT_MAP.get(sidebar.get("Deployment", "Any"), "any"),
        "reasoning_type": "standard" if sidebar.get("Reasoning") else "none",
        "require_tool_calling": bool(sidebar.get("Tool Calling")),
        "min_speed_class": _SPEED_MAP.get(sidebar.get("Speed", "Any/Slow+"), None),
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
        "summary_markdown": ui.get("summary_markdown"),
        "comparison_table": ui.get("comparison_table"),
        "recommendation_cards": ui.get("recommendation_cards", []),
        "warnings": ui.get("warnings", []),
        "errors": api_response.get("errors", []),
    }
