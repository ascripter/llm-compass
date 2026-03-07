"""HTTP client for the LLM Compass FastAPI backend."""

import os

import httpx

_API_URL = os.environ.get("LLM_COMPASS_API_URL", "http://localhost:8000").rstrip("/")
_API_KEY = os.environ.get("LLM_COMPASS_API_KEY", "dev-api-key")
_HEADERS = {"X-API-Key": _API_KEY, "Content-Type": "application/json"}
_TIMEOUT = 120.0  # graph invocation can take a while


def post_query(user_query: str, constraints: dict) -> dict:
    """POST /api/v1/query — start a new query session."""
    payload = {"user_query": user_query, "constraints": constraints}
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(f"{_API_URL}/api/v1/query", json=payload, headers=_HEADERS)
        resp.raise_for_status()
        return resp.json()


def post_clarify(session_id: str, user_reply: str) -> dict:
    """POST /api/v1/query/{session_id}/clarify — send clarification."""
    payload = {"user_reply": user_reply}
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(
            f"{_API_URL}/api/v1/query/{session_id}/clarify",
            json=payload,
            headers=_HEADERS,
        )
        resp.raise_for_status()
        return resp.json()
