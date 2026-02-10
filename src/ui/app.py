"""
Main Application Entry Point.
Req 3.1: "Mission Control" layout.
"""

import streamlit as st
from src.ui.components import sidebar, chat, tables, traceability
from src.agentic_core.graph import build_graph

st.set_page_config(layout="wide", page_title="LLM Benchmark Analyst")


def main():
    # 1. Render Sidebar & Get Constraints
    constraints = sidebar.render_sidebar()

    # 2. Render Main Layout
    col_chat, col_trace = st.columns([2, 1])

    # 3. Chat Logic
    user_input = st.chat_input("Describe your task...")
    if user_input:
        graph = build_graph()
        # Execute graph with user_input + constraints
        # Update session state with results

    # 4. Render Results
    if "results" in st.session_state:
        tables.render_comparison_table(st.session_state.results)
