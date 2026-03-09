import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage

from ..deps import get_db, require_api_key
from ..schemas.common import ErrorDetail
from ..schemas.query import ClarifyRequest, QueryRequest, QueryResponse, TraceEvent, UIComponents
try:
    from llm_compass.agentic_core.graph import build_graph
except Exception:  # pragma: no cover - startup fallback when agent deps are unavailable
    def build_graph(settings=None, session=None) -> Any:
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


def _extract_clarification_question(state: dict[str, Any]) -> str | None:
    """Extract the last AI message content as the clarification question."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, dict):
            content = msg.get("content")
            msg_type = msg.get("type")
        else:
            content = getattr(msg, "content", None)
            msg_type = getattr(msg, "type", None)
        if msg_type == "ai" and content:
            return str(content)
    return None


def _build_intermediate_summary(state: dict[str, Any]) -> str:
    """Build a markdown summary from intermediate pipeline results (no synthesis node yet)."""
    parts: list[str] = []

    intent = state.get("intent_extraction")
    if intent is not None:
        if isinstance(intent, dict):
            reasoning = intent.get("reasoning", "")
            inputs = intent.get("intended_input_modalities", [])
            outputs = intent.get("intended_output_modalities", [])
        else:
            reasoning = getattr(intent, "reasoning", "")
            inputs = getattr(intent, "intended_input_modalities", [])
            outputs = getattr(intent, "intended_output_modalities", [])
        parts.append("## Intent Analysis")
        if reasoning:
            parts.append(reasoning)
        parts.append(f"**Input modalities:** {', '.join(inputs) if inputs else 'none detected'}")
        parts.append(f"**Output modalities:** {', '.join(outputs) if outputs else 'none detected'}")

    token_ratio = state.get("token_ratio_estimation")
    if token_ratio is not None:
        if isinstance(token_ratio, dict):
            in_ratios = token_ratio.get("normalized_input_ratios", {})
            out_ratios = token_ratio.get("normalized_output_ratios", {})
            tr_reasoning = token_ratio.get("reasoning", "")
        else:
            in_ratios = getattr(token_ratio, "normalized_input_ratios", {})
            out_ratios = getattr(token_ratio, "normalized_output_ratios", {})
            tr_reasoning = getattr(token_ratio, "reasoning", "")
        parts.append("\n## Token Ratio Estimation")
        if tr_reasoning:
            parts.append(tr_reasoning)
        if in_ratios:
            parts.append(f"**Normalized input ratios:** {in_ratios}")
        if out_ratios:
            parts.append(f"**Normalized output ratios:** {out_ratios}")

    search_queries = state.get("search_queries", [])
    if search_queries:
        parts.append("\n## Generated Search Queries")
        parts.extend(f"- {q}" for q in search_queries)

    if not parts:
        parts.append("Analysis complete. Ranking and recommendations are not yet available.")

    return "\n\n".join(parts)


def _build_response(session_id: str, state: dict[str, Any]) -> QueryResponse:
    raw_errors = state.get("errors", [])
    errors: list[ErrorDetail] = []
    if isinstance(raw_errors, list):
        for item in raw_errors:
            if isinstance(item, dict):
                code = str(item.get("code", "ERROR"))
                message = str(item.get("message", "Unknown error"))
                errors.append(ErrorDetail(code=code, message=message))

    # Derive status from intent_extraction.is_specific (not the missing clarification_needed key)
    intent = state.get("intent_extraction")
    if intent is None:
        is_specific = True
    elif isinstance(intent, dict):
        is_specific = bool(intent.get("is_specific", True))
    else:
        is_specific = bool(intent.is_specific)

    limit_exceeded = bool(state.get("clarification_limit_exceeded", False))
    if errors or limit_exceeded:
        status: str = "error"
    elif not is_specific:
        status = "needs_clarification"
    else:
        status = "ok"

    # Extract clarification question from the last AIMessage in conversation
    clarification_question: str | None = None
    if status == "needs_clarification":
        clarification_question = _extract_clarification_question(state)

    # Build fallback UIComponents from intermediate pipeline results when synthesis not yet done
    ui_components = state.get("ui_components")
    if ui_components is None and status == "ok":
        summary = _build_intermediate_summary(state)
        ui_components = UIComponents(summary_markdown=summary)

    return QueryResponse(
        session_id=session_id,
        user_query=str(state.get("user_query", "")),
        applied_constraints=state.get("constraints", {}) if isinstance(state.get("constraints"), dict) else {},
        status=status,  # type: ignore[arg-type]
        clarification_question=clarification_question,
        traceability=_build_traceability(state),
        ranked_data=state.get("ranked_data"),
        ui_components=ui_components,
        errors=errors,
    )


@router.post("/query", response_model=QueryResponse)
async def create_query(
    req: QueryRequest,
    _: str = Depends(require_api_key),
    db: object | None = Depends(get_db),
) -> QueryResponse:
    session_id = req.session_id or str(uuid.uuid4())
    graph = build_graph(session=db)

    initial_state: dict[str, Any] = {
        "user_query": req.user_query,
        "constraints": req.constraints.model_dump(),
        "messages": [HumanMessage(req.user_query)],
        "clarification_count": 0,
        "clarification_limit_exceeded": False,
        "search_queries": [],
        "ranked_results": {},
        "final_response": None,
        "logs": [],
    }

    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke(initial_state, config=config)
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
    prev_state = _sessions.get(session_id)
    if not prev_state:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Session not found"})

    prev_state["constraints"] = req.constraints.model_dump()

    msgs = list(prev_state.get("messages", []))
    msgs.append(HumanMessage(req.user_reply))
    prev_state["messages"] = msgs

    graph = build_graph(session=db)
    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke(prev_state, config=config)
    state = result if isinstance(result, dict) else prev_state

    _sessions[session_id] = state
    return _build_response(session_id, state)
