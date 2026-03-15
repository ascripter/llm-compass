from datetime import datetime, timezone

import streamlit as st


def _build_report_markdown(display: dict) -> str:
    """Assemble a full markdown report from the display dict."""
    parts: list[str] = []

    summary = display.get("summary_markdown")
    if summary:
        parts.append(summary)

    table = display.get("comparison_table")
    if table:
        cols = table.get("columns", [])
        rows = table.get("rows", [])
        if cols and rows:
            parts.append(f"## {table.get('title', 'Comparison')}\n")
            parts.append("| " + " | ".join(cols) + " |")
            parts.append("| " + " | ".join("---" for _ in cols) + " |")
            for row in rows:
                parts.append("| " + " | ".join(str(v) for v in row) + " |")
            parts.append("")

    cards = display.get("recommendation_cards", [])
    if cards:
        parts.append("## Recommendations\n")
        for card in cards:
            category = card.get("category", "")
            model = card.get("model_name", "")
            reason = card.get("reason", "")
            parts.append(f"**{category}**: {model}  ")
            parts.append(f"{reason}\n")

    return "\n".join(parts)


_END_STATE_CSS = """
<style>
/* Red 'Start Over' button */
div[data-testid="stButton"] button[kind="secondary"][data-testid="start-over-btn"],
div[data-testid="stButton"]:has(button[kind="secondary"]) button.start-over {
    background-color: #d9534f;
    color: white;
    border: none;
}
/* Target by key – Streamlit adds key as id on the container */
div[data-testid="stButton"].start-over button,
#start-over button {
    background-color: #d9534f;
    color: white;
    border: none;
}
/* Green download button */
div[data-testid="stDownloadButton"] button {
    background-color: #5cb85c;
    color: white;
    border: none;
}
</style>
"""


def render_chat(constraints, *, debug: bool = False, status: str | None = None,
                display: dict | None = None):
    if debug:
        with st.expander("Active Constraints", expanded=True):
            st.write(constraints)

    st.subheader("💬 LLM Compass")

    if not st.session_state.messages:
        st.info("What task do you want to accomplish?")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    is_finished = status == "ok" and display and display.get("summary_markdown")
    is_error = status == "error"

    if is_finished or is_error:
        st.markdown(_END_STATE_CSS, unsafe_allow_html=True)

        if is_finished:
            report = _build_report_markdown(display)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "Save Markdown Report",
                data=report,
                file_name=f"LLM_compass_{ts}.md",
                mime="text/markdown",
            )

        start_over = st.button("Start Over", key="start-over")
        return None, start_over

    prompt = st.chat_input("Describe your task (e.g., 'RAG on legal documents')")
    return prompt, False
