"""Main Application Entry Point — Streamlit frontend for the LLM Compass API."""

import logging
import streamlit as st

from llm_compass.config import get_settings
from llm_compass.ui.components import sidebar, chat, tables, traceability
from llm_compass.ui import api_client, transformers

logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="LLM Benchmark Analyst")


@st.cache_resource
def _init_logging():
    settings = get_settings()
    settings.setup_app_logging("frontend")
    logger.info("Streamlit frontend started")
    return True


def _init_session_state() -> None:
    defaults = {
        "messages": [],
        "trace_runs": [],      # list of {"label": str, "steps": list[dict], "messages": list[str]}
        "pending_query": None, # {"text": str, "mode": "query"|"clarify"} set before fast rerun
        "display": None,
        "session_id": None,
        "status": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _run_stream_loop(event_stream, tracker_ph) -> tuple[dict | None, list[dict]]:
    """Drive a node-event stream, updating tracker_ph live.

    Returns (raw_response_dict, final_steps).  raw is None on error.
    """
    steps = traceability.init_steps()
    with tracker_ph.container():
        traceability.render_live_tracker(steps, is_complete=False)

    raw = None
    for event in event_stream:
        etype = event.get("event")
        if etype == "node_complete":
            msg = event.get("message", "")
            matched = False
            for i, step in enumerate(steps):
                if step["status"] == "running" or step["name"] == msg:
                    step["status"] = "done"
                    if i + 1 < len(steps):
                        steps[i + 1]["status"] = "running"
                    matched = True
                    break
            if not matched:
                steps.append({"name": msg, "status": "done"})
            with tracker_ph.container():
                traceability.render_live_tracker(steps, is_complete=False)
        elif etype == "complete":
            raw = event.get("data")
            for step in steps:
                if step["status"] not in ("done", "failed"):
                    step["status"] = "done"
            with tracker_ph.container():
                traceability.render_live_tracker(steps, is_complete=True)
        elif etype == "error":
            for step in steps:
                if step["status"] == "running":
                    step["status"] = "failed"
                    break
            with tracker_ph.container():
                traceability.render_live_tracker(steps, is_complete=True)
            st.error(event.get("message", "Unknown error"))
            return None, steps

    return raw, steps


def _apply_clarification_failure(steps: list[dict], tracker_ph) -> None:
    """If the graph requested clarification, mark 'Analyzing intent' as failed."""
    for step in steps:
        if step["name"] == "Analyzing intent":
            step["status"] = "failed"
            break
    with tracker_ph.container():
        traceability.render_live_tracker(steps, is_complete=True)


def _run_query(user_query: str, sidebar_constraints: dict, tracker_ph) -> None:
    constraints = transformers.sidebar_to_constraints(sidebar_constraints)
    try:
        raw, steps = _run_stream_loop(
            api_client.post_query_stream(user_query, constraints),
            tracker_ph,
        )
    except Exception as exc:
        st.error(f"API error: {exc}")
        return

    if raw is None:
        st.error("Stream ended without a response")
        return

    if raw.get("status") == "needs_clarification":
        _apply_clarification_failure(steps, tracker_ph)

    display = transformers.response_to_display(raw)
    label = f"Query {len(st.session_state.trace_runs) + 1}"
    new_run = {"label": label, "steps": steps, "messages": display["trace_messages"]}
    st.session_state.trace_runs = [new_run]  # reset for new top-level query
    st.session_state.display = display
    st.session_state.session_id = display["session_id"]
    st.session_state.status = display["status"]

    if display["status"] == "error":
        msgs = "; ".join(e.get("message", "") for e in display["errors"])
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {msgs}"})
    elif display["status"] == "needs_clarification":
        q = display["clarification_question"] or "Could you clarify?"
        st.session_state.messages.append({"role": "assistant", "content": q})
    else:
        summary = display.get("summary_markdown") or "Query processed — see the Results panel."
        st.session_state.messages.append({"role": "assistant", "content": summary})


def _run_clarify(user_reply: str, sidebar_constraints: dict, tracker_ph) -> None:
    session_id = st.session_state.session_id
    if not session_id:
        st.error("No active session to clarify.")
        return
    constraints = transformers.sidebar_to_constraints(sidebar_constraints)
    try:
        raw, steps = _run_stream_loop(
            api_client.post_clarify_stream(session_id, user_reply, constraints),
            tracker_ph,
        )
    except Exception as exc:
        st.error(f"API error: {exc}")
        return

    if raw is None:
        st.error("Stream ended without a response")
        return

    if raw.get("status") == "needs_clarification":
        _apply_clarification_failure(steps, tracker_ph)

    display = transformers.response_to_display(raw)
    label = f"Clarification {len(st.session_state.trace_runs)}"
    new_run = {"label": label, "steps": steps, "messages": display["trace_messages"]}
    st.session_state.trace_runs = st.session_state.trace_runs + [new_run]  # append
    st.session_state.display = display
    st.session_state.status = display["status"]

    if display["status"] == "needs_clarification":
        q = display["clarification_question"] or "Could you clarify further?"
        st.session_state.messages.append({"role": "assistant", "content": q})
    else:
        summary = display.get("summary_markdown") or "Query processed — see the Results panel."
        st.session_state.messages.append({"role": "assistant", "content": summary})


def main() -> None:
    _init_logging()
    _init_session_state()

    sidebar_constraints = sidebar.render_sidebar()
    prompt = chat.render_chat(sidebar_constraints)

    # Step 1: new submission — save message and rerun immediately so it appears in chat
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        mode = "clarify" if st.session_state.status == "needs_clarification" else "query"
        st.session_state.pending_query = {"text": prompt, "mode": mode}
        st.rerun()

    # Step 2: process queued query (user message already visible in chat above)
    if st.session_state.get("pending_query"):
        pq = st.session_state.pop("pending_query")
        tracker_ph = st.empty()  # placeholder position: right after chat, above results
        tables.render_results(st.session_state.display)  # previous results visible during streaming
        traceability.render_accumulated_trace(st.session_state.trace_runs)  # previous trace visible
        if pq["mode"] == "clarify":
            _run_clarify(pq["text"], sidebar_constraints, tracker_ph)
        else:
            _run_query(pq["text"], sidebar_constraints, tracker_ph)
        st.rerun()

    # Idle state: results above trace
    tables.render_results(st.session_state.display)
    traceability.render_accumulated_trace(st.session_state.trace_runs)


main()
