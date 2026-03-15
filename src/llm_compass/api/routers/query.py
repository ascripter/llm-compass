import json
import logging
import uuid
from typing import Any, AsyncIterator, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from llm_compass.agentic_core.graph import get_graph
from llm_compass.agentic_core.schemas.ranking import RankedLists
from llm_compass.agentic_core.schemas.synthesis import SynthesisOutput
from llm_compass.agentic_core.state import get_initial_state, AgentState
from ..deps import get_db, require_api_key
from ..schemas.common import ErrorDetail
from ..schemas.query import (
    ClarifyRequest,
    QueryRequest,
    QueryResponse,
    StreamEvent,
    TraceEvent,
    UIComponents,
)

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1", tags=["Query"])

# In-memory session store (MVP)
_sessions: dict[str, dict[str, Any] | AgentState] = {}


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
            # reasoning = intent.get("reasoning", "")
            inputs = intent.get("intended_input_modalities", [])
            outputs = intent.get("intended_output_modalities", [])
        else:
            # reasoning = getattr(intent, "reasoning", "")
            inputs = getattr(intent, "intended_input_modalities", [])
            outputs = getattr(intent, "intended_output_modalities", [])
        parts.append("## Intent Analysis (DEBUG OUTPUT)")
        # if reasoning:
        #     parts.append(reasoning)
        parts.append(f"**Input modalities:** {', '.join(inputs) if inputs else 'none detected'}")
        parts.append(
            f"**Output modalities:** {', '.join(outputs) if outputs else 'none detected'}"
        )

    token_ratio = state.get("token_ratio_estimation")
    if token_ratio is not None:
        if isinstance(token_ratio, dict):
            in_ratios = token_ratio.get("normalized_input_ratios", {})
            out_ratios = token_ratio.get("normalized_output_ratios", {})
            # tr_reasoning = token_ratio.get("reasoning", "")
        else:
            in_ratios = getattr(token_ratio, "normalized_input_ratios", {})
            out_ratios = getattr(token_ratio, "normalized_output_ratios", {})
            # tr_reasoning = getattr(token_ratio, "reasoning", "")
        parts.append("\n## Token Ratio Estimation")
        # if tr_reasoning:
        #     parts.append(tr_reasoning)
        if in_ratios:
            parts.append(f"**Normalized input ratios:** {in_ratios}")
        if out_ratios:
            parts.append(f"**Normalized output ratios:** {out_ratios}")

    search_queries = state.get("search_queries", [])
    if search_queries:
        parts.append("\n## Generated Search Queries")
        parts.extend(f"- {q}" for q in search_queries)

    weighted_benchmarks = state.get("weighted_benchmarks") or []
    if weighted_benchmarks:
        avg_sim = state.get("average_benchmark_similarity") or 0.0
        header = f"\n## Discovered Benchmarks\nAverage relevance: {avg_sim:.2f}  ·  {len(weighted_benchmarks)} benchmark(s) matched\n"
        rows = ["| Benchmark | Variant | Weight |", "|---|---|---|"]
        for b in weighted_benchmarks:
            name = b.get("name") or b.get("name_normalized") or b.get("id", "?")
            variant = b.get("variant") or "—"
            weight = b.get("weight", 0.0)
            rows.append(f"| {name} | {variant} | {weight:.3f} |")
        parts.append(header + "\n".join(rows))

    ranked_raw = state.get("ranked_results")
    if isinstance(ranked_raw, RankedLists):
        ranked_results = ranked_raw.model_dump()
    else:
        ranked_results = ranked_raw or {}
    categories = [
        ("top_performance", "Top Performance"),
        ("balanced", "Balanced"),
        ("budget", "Budget"),
    ]
    ranking_parts: list[str] = []
    for key, label in categories:
        models = ranked_results.get(key) or []
        if not models:
            continue
        ranking_parts.append(f"**{label}**")
        for i, m in enumerate(models[:3], 1):
            rm = m.get("rank_metrics") or {}
            blended = rm.get("blended_score", 0.0)
            name = m.get("name_normalized") or m.get("model_id", "?")
            provider = m.get("provider", "")
            provider_str = f" ({provider})" if provider else ""
            reason = m.get("reason_for_ranking", "")
            ranking_parts.append(f"{i}. **{name}**{provider_str} — blended score: {blended:.3f}")
            if reason:
                ranking_parts.append(f"   _{reason}_")
    if ranking_parts:
        parts.append("\n## Ranking Results\n\n" + "\n".join(ranking_parts))

    if not parts:
        parts.append("Analysis complete. Ranking and recommendations are not yet available.")

    return "\n\n".join(parts)


def _parse_ranked_results(raw: Any) -> RankedLists | None:
    if isinstance(raw, RankedLists):
        return raw
    if not raw or not isinstance(raw, dict):
        return None
    try:
        return RankedLists.model_validate(raw)
    except Exception:
        logger.warning("Could not parse ranked_results into RankedLists", exc_info=True)
        return None


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

    # Prefer SynthesisOutput from final_response; fall back to intermediate summary
    final = state.get("final_response")
    if isinstance(final, SynthesisOutput):
        ui_components = final
    elif status == "ok":
        summary = _build_intermediate_summary(state)
        ui_components = UIComponents(summary_markdown=summary)
    else:
        ui_components = None

    return QueryResponse(
        session_id=session_id,
        user_query=str(state.get("user_query", "")),
        applied_constraints=(
            state.get("constraints", {}) if isinstance(state.get("constraints"), dict) else {}
        ),
        status=status,  # type: ignore[arg-type]
        clarification_question=clarification_question,
        traceability=_build_traceability(state),
        ranked_data=_parse_ranked_results(state.get("ranked_results")),
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
    graph = get_graph()
    initial_state = get_initial_state()
    initial_state["user_query"] = req.user_query
    initial_state["constraints"] = req.constraints  # .model_dump()
    initial_state["messages"] = [HumanMessage(req.user_query)]

    config = {"configurable": {"thread_id": session_id, "session": db}}
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
        raise HTTPException(
            status_code=404, detail={"code": "NOT_FOUND", "message": "Session not found"}
        )

    prev_state["constraints"] = req.constraints.model_dump()

    msgs = list(prev_state.get("messages", []))
    msgs.append(HumanMessage(req.user_reply))
    prev_state["messages"] = msgs

    graph = get_graph()
    config = {"configurable": {"thread_id": session_id, "session": db}}
    result = graph.invoke(prev_state, config=config)
    state = result if isinstance(result, dict) else prev_state

    _sessions[session_id] = state
    return _build_response(session_id, state)


# ---------------------------------------------------------------------------
# Streaming endpoint (NDJSON)
# ---------------------------------------------------------------------------

# Mapping node names from agentic_core/graph.py to messages
_NODE_LABELS = {
    "validator": "Analyzing intent",
    "token_ratio": "Estimating token ratios",
    "refiner": "Generating search queries",
    "benchmark_discovery": "Discovering benchmarks",
    "ranking": "Ranking models",
    "synthesis": "Synthesizing response",
}

# Keys whose reducer is *append* rather than *overwrite*
_APPEND_KEYS = {"logs", "messages"}


async def _stream_graph(session_id: str, initial_state: dict, config: dict) -> AsyncIterator[str]:
    """Yield NDJSON lines: one per completed node, then a final ``complete`` event."""
    graph = get_graph()
    accumulated = dict(initial_state)

    try:
        async for chunk in graph.astream(initial_state, config=config, stream_mode="updates"):
            for node_name, update in chunk.items():
                # Merge update into accumulated state
                for key, value in update.items():
                    if key in _APPEND_KEYS:
                        existing = accumulated.get(key, [])
                        accumulated[key] = existing + (
                            value if isinstance(value, list) else [value]
                        )
                    else:
                        accumulated[key] = value

                event = StreamEvent(
                    event="node_complete",
                    node=node_name,
                    message=_NODE_LABELS.get(node_name, node_name),
                )
                yield json.dumps(event.model_dump()) + "\n"
    except Exception as exc:
        logger.exception("Error during graph streaming")
        err = StreamEvent(event="error", message=str(exc))
        yield json.dumps(err.model_dump()) + "\n"
        return

    _sessions[session_id] = accumulated
    response = _build_response(session_id, accumulated)

    complete = StreamEvent(event="complete", data=response.model_dump())
    yield json.dumps(complete.model_dump()) + "\n"


@router.post("/query/stream")
async def create_query_stream(
    req: QueryRequest,
    _: str = Depends(require_api_key),
    db: object | None = Depends(get_db),
) -> StreamingResponse:
    session_id = req.session_id or str(uuid.uuid4())
    initial_state = get_initial_state()
    initial_state["user_query"] = req.user_query
    initial_state["constraints"] = req.constraints
    initial_state["messages"] = [HumanMessage(req.user_query)]

    config = {"configurable": {"thread_id": session_id, "session": db}}

    return StreamingResponse(
        _stream_graph(session_id, initial_state, config),
        media_type="application/x-ndjson",
    )
