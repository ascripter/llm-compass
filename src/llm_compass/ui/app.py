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
        "trace_messages": [],
        "display": None,
        "session_id": None,
        "status": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _run_query(user_query: str, sidebar_constraints: dict) -> None:
    constraints = transformers.sidebar_to_constraints(sidebar_constraints)
    try:
        raw = api_client.post_query(user_query, constraints)
    except Exception as exc:
        st.error(f"API error: {exc}")
        return

    display = transformers.response_to_display(raw)
    st.session_state.display = display
    st.session_state.session_id = display["session_id"]
    st.session_state.status = display["status"]
    st.session_state.trace_messages = display["trace_messages"]

    if display["status"] == "error":
        msgs = "; ".join(e.get("message", "") for e in display["errors"])
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {msgs}"})
    elif display["status"] == "needs_clarification":
        q = display["clarification_question"] or "Could you clarify?"
        st.session_state.messages.append({"role": "assistant", "content": q})
    else:
        summary = display.get("summary_markdown") or "Results ready."
        st.session_state.messages.append({"role": "assistant", "content": summary})


def _run_clarify(user_reply: str) -> None:
    session_id = st.session_state.session_id
    if not session_id:
        st.error("No active session to clarify.")
        return
    try:
        raw = api_client.post_clarify(session_id, user_reply)
    except Exception as exc:
        st.error(f"API error: {exc}")
        return

    display = transformers.response_to_display(raw)
    st.session_state.display = display
    st.session_state.status = display["status"]
    st.session_state.trace_messages = display["trace_messages"]

    if display["status"] == "needs_clarification":
        q = display["clarification_question"] or "Could you clarify further?"
        st.session_state.messages.append({"role": "assistant", "content": q})
    else:
        summary = display.get("summary_markdown") or "Results ready."
        st.session_state.messages.append({"role": "assistant", "content": summary})


def main() -> None:
    _init_logging()
    _init_session_state()

    sidebar_constraints = sidebar.render_sidebar()

    col_chat, col_trace = st.columns([2, 1])

    with col_chat:
        prompt = chat.render_chat(sidebar_constraints)

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            if st.session_state.status == "needs_clarification":
                _run_clarify(prompt)
            else:
                _run_query(prompt, sidebar_constraints)

            st.rerun()

        tables.render_results(st.session_state.display)

    with col_trace:
        traceability.render_traceability()


main()
