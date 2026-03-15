from datetime import datetime, timezone

import streamlit as st


def _build_report_markdown(display: dict) -> str:
    """Assemble a full markdown report from the display dict."""
    parts: list[str] = []

    summary = display.get("summary_markdown")
    if summary:
        parts.append(summary)

    for tier in display.get("tier_tables", []):
        tier_data = tier if isinstance(tier, dict) else tier
        tier_name = tier_data.get("tier_name", "Results") if isinstance(tier_data, dict) else getattr(tier_data, "tier_name", "Results")
        columns_raw = tier_data.get("columns", []) if isinstance(tier_data, dict) else getattr(tier_data, "columns", [])
        rows_raw = tier_data.get("rows", []) if isinstance(tier_data, dict) else getattr(tier_data, "rows", [])

        cols = ["Model", "Provider", "Speed", "Score"] + list(columns_raw)
        parts.append(f"## {tier_name}\n")
        parts.append("| " + " | ".join(cols) + " |")
        parts.append("| " + " | ".join("---" for _ in cols) + " |")
        for row in rows_raw:
            if isinstance(row, dict):
                cells = [row.get("model_name", ""), row.get("provider", ""),
                         row.get("speed", ""), f"{row.get('score', 0):.3f}"]
                for bs in row.get("benchmark_scores", []):
                    val = bs.get("value") if isinstance(bs, dict) else getattr(bs, "value", None)
                    is_est = bs.get("is_estimated", False) if isinstance(bs, dict) else getattr(bs, "is_estimated", False)
                    if val is None:
                        cells.append("--")
                    elif is_est:
                        cells.append(f"*{val:.2f}*")
                    else:
                        cells.append(f"{val:.2f}")
            else:
                cells = [row.model_name, row.provider, row.speed, f"{row.score:.3f}"]
                for bs in row.benchmark_scores:
                    if bs.value is None:
                        cells.append("--")
                    elif bs.is_estimated:
                        cells.append(f"*{bs.value:.2f}*")
                    else:
                        cells.append(f"{bs.value:.2f}")
            parts.append("| " + " | ".join(cells) + " |")
        parts.append("")

    cards = display.get("recommendation_cards", [])
    if cards:
        parts.append("## Recommendations\n")
        for card in cards:
            category = card.get("category", "") if isinstance(card, dict) else card.category
            model = card.get("model_name", "") if isinstance(card, dict) else card.model_name
            reason = card.get("reason", "") if isinstance(card, dict) else card.reason
            parts.append(f"**{category}**: {model}  ")
            parts.append(f"{reason}\n")

    benchmarks = display.get("benchmarks_used", [])
    if benchmarks:
        parts.append("## Benchmarks Used\n")
        parts.append("| Benchmark | Weight | Description |")
        parts.append("|---|---|---|")
        for b in benchmarks:
            name = b.get("benchmark_name", "") if isinstance(b, dict) else b.benchmark_name
            weight = b.get("weight", 0) if isinstance(b, dict) else b.weight
            desc = b.get("description", "") if isinstance(b, dict) else b.description
            parts.append(f"| {name} | {weight:.2f} | {desc} |")

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


def render_chat(constraints, *, debug: bool = False):
    """Render chat messages and input. Returns the user prompt or None."""
    if debug:
        with st.expander("Active Constraints", expanded=True):
            st.write(constraints)

    st.subheader("💬 LLM Compass")

    if not st.session_state.messages:
        st.info("What task do you want to accomplish?")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Hide chat input when query is finished or errored
    status = st.session_state.get("status")
    if status in ("ok", "error"):
        return None

    prompt = st.chat_input("Describe your task (e.g., 'RAG on legal documents')")
    return prompt


def render_end_buttons(display: dict | None, status: str | None) -> bool:
    """Render Save + Start Over buttons at the bottom. Returns True if Start Over clicked."""
    is_finished = status == "ok" and display and display.get("summary_markdown")
    is_error = status == "error"

    if not (is_finished or is_error):
        return False

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

    return st.button("Start Over", key="start-over")
