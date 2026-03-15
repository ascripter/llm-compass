import html

import pandas as pd
import streamlit as st

from llm_compass.config import get_settings


_TIER_TABLE_CSS = """
<style>
.tier-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1rem;
    font-size: 0.9rem;
}
.tier-table th, .tier-table td {
    padding: 0.45rem 0.7rem;
    text-align: left;
    border-bottom: 1px solid rgba(128,128,128,0.25);
}
.tier-table th {
    font-weight: 600;
    background-color: rgba(128,128,128,0.1);
}
.tier-table tr:hover {
    background-color: rgba(128,128,128,0.07);
}
.tier-table .est-cell {
    font-style: italic;
    opacity: 0.6;
}
</style>
"""


def _esc(text: str) -> str:
    """HTML-escape a string for safe embedding in table cells."""
    return html.escape(str(text))


def _render_tier_table(tier: dict) -> None:
    """Render a single tier as an HTML table with estimated-score styling."""
    tier_name = tier.get("tier_name", "Results")
    bench_columns = tier.get("columns", [])
    rows = tier.get("rows", [])

    st.markdown(f"### {_esc(tier_name)}")

    if not rows:
        st.caption("No models available for this tier.")
        return

    all_columns = ["Model", "Provider", "Speed", "Score"] + list(bench_columns)

    parts = ['<table class="tier-table"><thead><tr>']
    for col in all_columns:
        parts.append(f"<th>{_esc(col)}</th>")
    parts.append("</tr></thead><tbody>")

    for row in rows:
        # Support both dict and object access
        if isinstance(row, dict):
            model_name = row.get("model_name", "")
            provider = row.get("provider", "")
            speed = row.get("speed", "")
            score = row.get("score", 0)
            bench_scores = row.get("benchmark_scores", [])
        else:
            model_name = row.model_name
            provider = row.provider
            speed = row.speed
            score = row.score
            bench_scores = row.benchmark_scores

        parts.append("<tr>")
        parts.append(f"<td>{_esc(model_name)}</td>")
        parts.append(f"<td>{_esc(provider)}</td>")
        parts.append(f"<td>{_esc(speed)}</td>")
        parts.append(f"<td>{score:.3f}</td>")

        for bs in bench_scores:
            if isinstance(bs, dict):
                val = bs.get("value")
                is_est = bs.get("is_estimated", False)
                est_src = bs.get("estimation_source")
            else:
                val = bs.value
                is_est = bs.is_estimated
                est_src = bs.estimation_source

            if val is None:
                parts.append("<td>--</td>")
            elif is_est:
                tooltip = f"This score was estimated from {_esc(est_src)}" if est_src else "This score was estimated"
                parts.append(
                    f'<td class="est-cell" title="{tooltip}">'
                    f"<em>{val:.2f}</em></td>"
                )
            else:
                parts.append(f"<td>{val:.2f}</td>")

        parts.append("</tr>")

    parts.append("</tbody></table>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def _render_benchmarks_used(benchmarks: list[dict]) -> None:
    """Render the 'Benchmarks Used' reference table."""
    if not benchmarks:
        return

    st.markdown("### Benchmarks Used")
    rows = []
    for b in benchmarks:
        if isinstance(b, dict):
            name = b.get("benchmark_name", "")
            weight = b.get("weight", 0)
            desc = b.get("description", "")
        else:
            name = b.benchmark_name
            weight = b.weight
            desc = b.description
        rows.append([name, f"{weight:.2f}", desc])

    df = pd.DataFrame(rows, columns=["Benchmark", "Weight", "Description"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_results(display: dict | None) -> None:
    if display is None:
        return

    settings = get_settings()

    # Debug report (intermediate pipeline summary)
    debug_summary = display.get("debug_summary")
    if debug_summary and settings.debug_output:
        with st.expander("Debug Report", expanded=True):
            st.markdown(debug_summary)

    tier_tables = display.get("tier_tables", [])
    cards = display.get("recommendation_cards", [])
    benchmarks_used = display.get("benchmarks_used", [])

    if not tier_tables and not cards:
        return

    # Results heading at same level as "## Your Task"
    st.markdown("## Results")

    # Warnings directly under heading
    for w in display.get("warnings", []):
        if isinstance(w, dict):
            st.warning(w.get("message", ""))
        else:
            st.warning(w.message)

    # Score normalization info panel
    if tier_tables:
        st.caption(
            "Scores are normalized 0\u20131. "
            "Top Performance = pure performance index; "
            "Balanced = 50% performance + 50% cost efficiency; "
            "Budget Picks = 20% performance + 80% cost efficiency."
        )

    # Inject tier table CSS once
    st.markdown(_TIER_TABLE_CSS, unsafe_allow_html=True)

    # 3 tier tables
    for tier in tier_tables:
        _render_tier_table(tier if isinstance(tier, dict) else tier.model_dump())

    # Recommendation cards
    if cards:
        st.markdown("### Recommendations")
        cols = st.columns(max(len(cards), 1))
        for col, card in zip(cols, cards):
            with col:
                if isinstance(card, dict):
                    category = card.get("category", "")
                    model_name = card.get("model_name", "")
                    reason = card.get("reason", "")
                else:
                    category = card.category
                    model_name = card.model_name
                    reason = card.reason
                st.markdown(
                    f'<div class="metric-card"><h4>{_esc(category)}</h4>'
                    f'<p><strong>{_esc(model_name)}</strong></p>'
                    f'<p>{_esc(reason)}</p></div>',
                    unsafe_allow_html=True,
                )

    # Benchmarks Used section
    _render_benchmarks_used(benchmarks_used)
