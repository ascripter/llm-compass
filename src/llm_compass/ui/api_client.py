"""HTTP client for the LLM Compass FastAPI backend."""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_API_URL = os.environ.get("LLM_COMPASS_API_URL", "http://localhost:8000").rstrip("/")
_API_KEY = os.environ.get("LLM_COMPASS_API_KEY", "dev-api-key")
_HEADERS = {"X-API-Key": _API_KEY, "Content-Type": "application/json"}
_TIMEOUT = 120.0  # graph invocation can take a while


def post_query(user_query: str, constraints: dict) -> dict:
    """POST /api/v1/query — start a new query session."""
    payload = {"user_query": user_query, "constraints": constraints}
    logger.debug("post_query REQUEST | user_query=%r | constraints=%s", user_query, constraints)
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(f"{_API_URL}/api/v1/query", json=payload, headers=_HEADERS)
        resp.raise_for_status()
        data = resp.json()
        logger.debug(
            "post_query RESPONSE | http_status=%d | session_id=%s | status=%r",
            resp.status_code,
            data.get("session_id"),
            data.get("status"),
        )
        return data


def post_clarify(session_id: str, user_reply: str) -> dict:
    """POST /api/v1/query/{session_id}/clarify — send clarification."""
    payload = {"user_reply": user_reply}
    logger.debug("post_clarify REQUEST | session_id=%s | user_reply=%r", session_id, user_reply)
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(
            f"{_API_URL}/api/v1/query/{session_id}/clarify",
            json=payload,
            headers=_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.debug(
            "post_clarify RESPONSE | http_status=%d | status=%r",
            resp.status_code,
            data.get("status"),
        )
        return data
