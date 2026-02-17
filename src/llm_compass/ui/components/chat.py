"""
Chat interface components.
Req 3.3.B: Renders the structured response ("Chat Bubble").
"""

import streamlit as st
import json


def render_chat_history():
    """
    Renders the chat history from session state.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "json_response":
                # Render the structured agent response
                render_structured_response(msg["content"])
            else:
                st.markdown(msg["content"])


def render_structured_response(response_json: dict):
    """
    Parses the Agent's JSON output and renders the 'Executive Summary'
    and other non-tabular UI components.

    Req 3.3.B.1: Executive Summary Markdown.
    Req 3.3.B.3: Recommendation Cards.
    """
    # 1. Executive Summary
    summary = response_json.get("ui_components", {}).get("summary_markdown", "")
    st.markdown(summary)

    # 2. Recommendation Cards (Visual Highlighting)
    cards = response_json.get("ui_components", {}).get("recommendation_cards", [])
    if cards:
        cols = st.columns(len(cards))
        for idx, card in enumerate(cards):
            with cols[idx]:
                st.info(
                    f"**{card['category']}**\n\n"
                    f"🏆 {card['model_name']}\n\n"
                    f"_{card['reason']}_"
                )

    # Tables are handled separately by tables.py, usually called after this
