"""HTTP client for the LLM Compass FastAPI backend."""

import json
import logging
import os
from typing import Generator

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


def post_query_stream(user_query: str, constraints: dict) -> Generator[dict, None, None]:
    """POST /api/v1/query/stream — stream node-completion events as NDJSON."""
    payload = {"user_query": user_query, "constraints": constraints}
    logger.debug("post_query_stream REQUEST | user_query=%r", user_query)
    with httpx.Client(timeout=_TIMEOUT) as client:
        with client.stream(
            "POST",
            f"{_API_URL}/api/v1/query/stream",
            json=payload,
            headers=_HEADERS,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    yield json.loads(line)


def post_clarify_stream(session_id: str, user_reply: str, constraints: dict) -> Generator[dict, None, None]:
    """POST /api/v1/query/{session_id}/clarify/stream — stream clarification as NDJSON."""
    payload = {"user_reply": user_reply, "constraints": constraints}
    logger.debug("post_clarify_stream REQUEST | session_id=%s | user_reply=%r", session_id, user_reply)
    with httpx.Client(timeout=_TIMEOUT) as client:
        with client.stream(
            "POST",
            f"{_API_URL}/api/v1/query/{session_id}/clarify/stream",
            json=payload,
            headers=_HEADERS,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    yield json.loads(line)


def post_clarify(session_id: str, user_reply: str, constraints: dict) -> dict:
    """POST /api/v1/query/{session_id}/clarify — send clarification."""
    payload = {"user_reply": user_reply, "constraints": constraints}
    logger.debug(
        "post_clarify REQUEST | session_id=%s | user_reply=%r | constraints=%s",
        session_id,
        user_reply,
        constraints,
    )
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
