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


def _run_stream_loop(
    event_stream, tracker_ph, prior_runs: list[dict] | None = None,
) -> tuple[dict | None, list[dict]]:
    """Drive a node-event stream, updating tracker_ph live.

    Returns (raw_response_dict, final_steps).  raw is None on error.
    """
    steps = traceability.init_steps()
    with tracker_ph.container():
        traceability.render_live_tracker(steps, is_complete=False, prior_runs=prior_runs)

    raw = None
    for event in event_stream:
        etype = event.get("event")
        if etype == "node_complete":
            msg = event.get("message", "")
            node_logs = event.get("logs") or []
            matched = False

            # Match by name so parallel nodes get the right logs
            for i, step in enumerate(steps):
                if step["name"] == msg:
                    step["status"] = "done"
                    step["logs"] = node_logs
                    matched = True

                    group = traceability.parallel_group_for(i)
                    if group is not None:
                        # Parallel step: advance only when ALL group members done
                        if all(steps[j]["status"] == "done" for j in group):
                            next_idx = max(group) + 1
                            if next_idx < len(steps):
                                steps[next_idx]["status"] = "running"
                    else:
                        # Sequential step: start next (or whole parallel group)
                        if i + 1 < len(steps):
                            next_group = traceability.parallel_group_for(i + 1)
                            if next_group is not None:
                                for j in next_group:
                                    steps[j]["status"] = "running"
                            else:
                                steps[i + 1]["status"] = "running"
                    break

            if not matched:
                steps.append({"name": msg, "status": "done", "logs": node_logs})
            with tracker_ph.container():
                traceability.render_live_tracker(steps, is_complete=False, prior_runs=prior_runs)
        elif etype == "complete":
            raw = event.get("data")
            status = (raw or {}).get("status")
            if status == "ok":
                for step in steps:
                    if step["status"] not in ("done", "failed"):
                        step["status"] = "done"
            else:
                # Pipeline ended early — reset steps that never ran
                for step in steps:
                    if step["status"] == "running":
                        step["status"] = "pending"
            with tracker_ph.container():
                traceability.render_live_tracker(steps, is_complete=True, prior_runs=prior_runs)
        elif etype == "error":
            first_running_found = False
            for step in steps:
                if step["status"] == "running":
                    if not first_running_found:
                        step["status"] = "failed"
                        first_running_found = True
                    else:
                        step["status"] = "pending"
            with tracker_ph.container():
                traceability.render_live_tracker(steps, is_complete=True, prior_runs=prior_runs)
            st.error(event.get("message", "Unknown error"))
            return None, steps

    return raw, steps


def _apply_clarification_failure(
    steps: list[dict], tracker_ph, prior_runs: list[dict] | None = None,
) -> None:
    """If the graph requested clarification, mark 'Analyzing intent' as failed."""
    for step in steps:
        if step["name"] == "Analyzing intent":
            step["status"] = "failed"
            break
    with tracker_ph.container():
        traceability.render_live_tracker(steps, is_complete=True, prior_runs=prior_runs)


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
    prior_runs = st.session_state.trace_runs
    try:
        raw, steps = _run_stream_loop(
            api_client.post_clarify_stream(session_id, user_reply, constraints),
            tracker_ph,
            prior_runs=prior_runs,
        )
    except Exception as exc:
        st.error(f"API error: {exc}")
        return

    if raw is None:
        st.error("Stream ended without a response")
        return

    if raw.get("status") == "needs_clarification":
        _apply_clarification_failure(steps, tracker_ph, prior_runs=prior_runs)

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
    settings = get_settings()
    prompt = chat.render_chat(sidebar_constraints, debug=settings.debug_output)

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
        if st.session_state.display is not None:
            tables.render_results(st.session_state.display)  # previous results visible during streaming
        if pq["mode"] == "clarify":
            _run_clarify(pq["text"], sidebar_constraints, tracker_ph)
        else:
            _run_query(pq["text"], sidebar_constraints, tracker_ph)
        st.rerun()

    # Idle state: only show results + trace if a query has completed
    if st.session_state.display is not None:
        tables.render_results(st.session_state.display)
        traceability.render_accumulated_trace(st.session_state.trace_runs)


main()
