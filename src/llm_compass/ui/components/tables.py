import pandas as pd
import streamlit as st

from llm_compass.config import get_settings


def render_results(display: dict | None) -> None:
    if display is None:
        return

    settings = get_settings()

    # Debug report (intermediate pipeline summary)
    debug_summary = display.get("debug_summary")
    if debug_summary and settings.debug_output:
        with st.expander("Debug Report", expanded=True):
            st.markdown(debug_summary)

    # Warnings
    for w in display.get("warnings", []):
        st.warning(w.get("message", ""))

    # Only show Results heading and content when synthesis has produced output
    table = display.get("comparison_table")
    cards = display.get("recommendation_cards", [])

    if not table and not cards:
        return

    st.subheader("Results")

    if table:
        st.markdown(f"**{table.get('title', 'Results')}**")
        df = pd.DataFrame(table.get("rows", []), columns=table.get("columns", []))
        st.dataframe(df, use_container_width=True)

    if cards:
        st.subheader("Recommendations")
        cols = st.columns(max(len(cards), 1))
        for col, card in zip(cols, cards):
            with col:
                st.markdown(
                    f'<div class="metric-card"><h4>{card.get("category", "")}</h4>'
                    f'<p><strong>{card.get("model_name", "")}</strong></p>'
                    f'<p>{card.get("reason", "")}</p></div>',
                    unsafe_allow_html=True,
                )
