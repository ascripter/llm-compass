from fastapi.testclient import TestClient

from src.llm_compass.api.main import app
from src.llm_compass.api.routers import query as query_router


class FakeGraph:
    def invoke(self, state: dict) -> dict:
        return {
            **state,
            "clarification_needed": False,
            "clarification_question": None,
            "logs": ["validator complete", "synthesis complete"],
        }


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": "dev-api-key"}


def test_query_requires_api_key(monkeypatch):
    monkeypatch.setattr(query_router, "build_graph", lambda: FakeGraph())
    query_router._sessions.clear()

    client = TestClient(app)
    body = {"user_query": "I need a model for RAG on legal documents"}

    missing = client.post("/api/v1/query", json=body)
    assert missing.status_code == 401
    assert missing.json()["errors"][0]["code"] == "UNAUTHORIZED"

    invalid = client.post("/api/v1/query", json=body, headers={"X-API-Key": "bad-key"})
    assert invalid.status_code == 401
    assert invalid.json()["errors"][0]["code"] == "UNAUTHORIZED"


def test_query_returns_session_id_and_status(monkeypatch):
    monkeypatch.setattr(query_router, "build_graph", lambda: FakeGraph())
    query_router._sessions.clear()

    client = TestClient(app)
    response = client.post(
        "/api/v1/query",
        json={"user_query": "I need a model for RAG on legal documents"},
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["session_id"], str) and payload["session_id"]
    assert payload["status"] in {"ok", "needs_clarification", "error"}


def test_clarify_endpoint_returns_query_response(monkeypatch):
    monkeypatch.setattr(query_router, "build_graph", lambda: FakeGraph())
    query_router._sessions.clear()

    client = TestClient(app)
    first = client.post(
        "/api/v1/query",
        json={"user_query": "I need a model for RAG on legal documents"},
        headers=_auth_headers(),
    )
    session_id = first.json()["session_id"]

    clarify = client.post(
        f"/api/v1/query/{session_id}/clarify",
        json={"user_reply": "Input is text only, output can be text and image"},
        headers=_auth_headers(),
    )

    assert clarify.status_code == 200
    payload = clarify.json()
    assert payload["session_id"] == session_id
    assert payload["status"] in {"ok", "needs_clarification", "error"}

