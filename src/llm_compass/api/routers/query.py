import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_db, require_api_key
from ..schemas.common import ErrorDetail
from ..schemas.query import ClarifyRequest, QueryRequest, QueryResponse, TraceEvent
try:
    from src.llm_compass.agentic_core.graph import build_graph
except Exception:  # pragma: no cover - startup fallback when agent deps are unavailable
    def build_graph() -> Any:
        raise RuntimeError("LangGraph build_graph is unavailable")


router = APIRouter(prefix="/api/v1", tags=["Query"])

# In-memory session store (MVP)
_sessions: dict[str, dict[str, Any]] = {}


def _build_traceability(state: dict[str, Any]) -> dict[str, List[TraceEvent]]:
    events: list[TraceEvent] = []

    existing = state.get("traceability")
    if isinstance(existing, dict):
        raw_events = existing.get("events", [])
        if isinstance(raw_events, list):
            for event in raw_events:
                if isinstance(event, dict):
                    try:
                        events.append(TraceEvent(**event))
                    except Exception:
                        continue

    if not events:
        logs = state.get("logs", [])
        if isinstance(logs, list):
            for entry in logs:
                events.append(TraceEvent(stage="agent", message=str(entry), data={}))

    return {"events": events}


def _build_response(session_id: str, state: dict[str, Any]) -> QueryResponse:
    raw_errors = state.get("errors", [])
    errors: list[ErrorDetail] = []
    if isinstance(raw_errors, list):
        for item in raw_errors:
            if isinstance(item, dict):
                code = str(item.get("code", "ERROR"))
                message = str(item.get("message", "Unknown error"))
                errors.append(ErrorDetail(code=code, message=message))

    status: str = "needs_clarification" if state.get("clarification_needed") else "ok"
    if errors:
        status = "error"

    return QueryResponse(
        session_id=session_id,
        user_query=str(state.get("user_query", "")),
        applied_constraints=state.get("constraints", {}) if isinstance(state.get("constraints"), dict) else {},
        status=status,  # type: ignore[arg-type]
        clarification_question=state.get("clarification_question"),
        traceability=_build_traceability(state),
        ranked_data=state.get("ranked_data"),
        ui_components=state.get("ui_components"),
        errors=errors,
    )


@router.post("/query", response_model=QueryResponse)
async def create_query(
    req: QueryRequest,
    _: str = Depends(require_api_key),
    db: object | None = Depends(get_db),
) -> QueryResponse:
    del db

    session_id = req.session_id or str(uuid.uuid4())
    graph = build_graph()

    initial_state: dict[str, Any] = {
        "user_query": req.user_query,
        "constraints": req.constraints.model_dump(),
        "clarification_needed": False,
        "clarification_question": None,
        "predicted_io_ratio": {},
        "search_queries": [],
        "ranked_results": {},
        "final_response": None,
        "logs": [],
    }

    result = graph.invoke(initial_state)
    state = result if isinstance(result, dict) else initial_state

    _sessions[session_id] = state
    return _build_response(session_id, state)


@router.post("/query/{session_id}/clarify", response_model=QueryResponse)
async def clarify_query(
    session_id: str,
    req: ClarifyRequest,
    _: str = Depends(require_api_key),
    db: object | None = Depends(get_db),
) -> QueryResponse:
    del db

    prev_state = _sessions.get(session_id)
    if not prev_state:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Session not found"})

    user_query = str(prev_state.get("user_query", ""))
    if user_query:
        prev_state["user_query"] = f"{user_query}\n[Clarification]: {req.user_reply}"
    else:
        prev_state["user_query"] = req.user_reply

    prev_state["clarification_needed"] = False

    graph = build_graph()
    result = graph.invoke(prev_state)
    state = result if isinstance(result, dict) else prev_state

    _sessions[session_id] = state
    return _build_response(session_id, state)
