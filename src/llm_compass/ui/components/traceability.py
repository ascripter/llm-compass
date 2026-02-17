"""
Req 3.3.A: Traceability View.
Displays the agent's 'thought process'.
"""

import streamlit as st


def render_traceability_logs(logs: list[str]):
    """
    Renders a collapsible sidebar or expander with logs.
    e.g. "> Searching benchmarks...", "> Found 3..."
    """
    with st.expander("Agent Trace", expanded=True):
        for log in logs:
            st.text(log)
