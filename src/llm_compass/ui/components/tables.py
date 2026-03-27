import html

import streamlit as st

from llm_compass.config import get_settings


_TABLE_CSS = """
<style>
/* ── Shared base: tier tables and benchmark table ── */
.tier-table, .benchmark-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    font-weight: 400;
}
.tier-table th, .tier-table td,
.benchmark-table th, .benchmark-table td {
    padding: 0.45rem 0.7rem;
    text-align: left;
    border-bottom: 1px solid rgba(128,128,128,0.25);
}
.tier-table th, .benchmark-table th {
    font-weight: 600;
    background-color: rgba(128,128,128,0.1);
}
.tier-table tr:hover, .benchmark-table tr:hover {
    background-color: rgba(128,128,128,0.07);
}
/* ── Tier-table specific ── */
.tier-table .est-cell {
    font-style: italic;
    opacity: 0.6;
}
.tier-table .score-col[title] {
    cursor: help;
    text-decoration: underline dotted rgba(128,128,128,0.6);
}
/* ── Shared column styles ── */
.score-col {
    font-weight: 600;
    color: #04AF37;
}
.desc-cell {
    white-space: normal;
    word-wrap: break-word;
    min-width: 200px;
}
/* ── Benchmark rows not shown in tier tables ── */
.greyed-row td {
    color: rgba(128,128,128,1.0);
}
</style>
"""

_TIER_CAPTIONS = {
    "Top Performance": "Pure benchmark performance, cost ignored.",
    "Balanced": "Weighted 50% performance, 50% cost (normalized)",
    "Budget Picks": "Weighted 20% performance, 80% cost (normalized)",
}


def _esc(text: str) -> str:
    """HTML-escape a string for safe embedding in table cells."""
    return html.escape(str(text))


def _render_tier_table(tier: dict) -> None:
    """Render a single tier as an HTML table with estimated-score styling."""
    tier_name = tier.get("tier_name", "Results")
    bench_columns = tier.get("columns", [])
    rows = tier.get("rows", [])

    st.markdown(f"### {_esc(tier_name)}")
    caption = tier.get("caption") or _TIER_CAPTIONS.get(tier_name)
    if caption:
        st.caption(caption)

    if not rows:
        st.caption("No models available for this tier.")
        return

    all_columns = ["Model", "Provider", "Speed (tps)", "Score"] + list(bench_columns)

    parts = ['<table class="tier-table"><thead><tr>']
    for col in all_columns:
        if col == "Speed (tps)":
            parts.append(
                f'<th title="Tokens/second approximated for fast providers'
                f' (if data available)">{_esc(col)}</th>'
            )
        elif col == "Score":
            parts.append(f'<th class="score-col">{_esc(col)}</th>')
        else:
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
            _sci = row.get("score_ci")  # dict or None
            ci_low = _sci.get("low") if isinstance(_sci, dict) else None
            ci_high = _sci.get("high") if isinstance(_sci, dict) else None
        else:
            model_name = row.model_name
            provider = row.provider
            speed = row.speed
            score = row.score
            bench_scores = row.benchmark_scores
            _sci = row.score_ci  # ScoreCI or None
            ci_low = _sci.low if _sci is not None else None
            ci_high = _sci.high if _sci is not None else None

        # Build Score cell — add tooltip when CI is a genuine interval (low < high)
        if ci_low is not None and ci_high is not None and ci_low < ci_high:
            ci_title = f' title="~ {ci_low:.3f} – {ci_high:.3f}\n(due to missing score data)"'
        else:
            ci_title = ""

        parts.append("<tr>")
        parts.append(f"<td>{_esc(model_name)}</td>")
        parts.append(f"<td>{_esc(provider)}</td>")
        parts.append(f"<td>{_esc(speed)}</td>")
        parts.append(f'<td class="score-col"{ci_title}>{score:.3f}</td>')

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
                tooltip = (
                    f"This score was estimated from {_esc(est_src)}"
                    if est_src
                    else "This score was estimated"
                )
                parts.append(
                    f'<td class="est-cell" title="{tooltip}">' f"<em>{val:.2f}</em></td>"
                )
            else:
                parts.append(f"<td>{val:.2f}</td>")

        parts.append("</tr>")

    parts.append("</tbody></table>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def _render_benchmarks_used(
    benchmarks: list[dict],
    tier_column_names: set[str] | None = None,
) -> None:
    """Render the 'Benchmarks Used' reference table using the shared HTML/CSS style."""
    if not benchmarks:
        return

    st.markdown("### Benchmarks Used")
    st.caption("Grey: Used in score weighting, but not shown in table")

    parts = ['<table class="benchmark-table"><thead><tr>']
    for col in ["Benchmark", "Weight", "Description"]:
        parts.append(f"<th>{_esc(col)}</th>")
    parts.append("</tr></thead><tbody>")

    for b in benchmarks:
        if isinstance(b, dict):
            name = b.get("benchmark_name", "")
            weight = b.get("weight", 0)
            desc = b.get("description", "")
        else:
            name = b.benchmark_name
            weight = b.weight
            desc = b.description

        is_shown = tier_column_names is None or name in tier_column_names
        row_class = "" if is_shown else ' class="greyed-row"'

        parts.append(f"<tr{row_class}>")
        parts.append(f"<td>{_esc(name)}</td>")
        parts.append(f"<td>{weight:.2f}</td>")
        parts.append(f'<td class="desc-cell">{_esc(desc)}</td>')
        parts.append("</tr>")

    parts.append("</tbody></table>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


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
        st.caption("Scores are normalized 0\u20131. Benchmark descriptions see below.")

    # Inject shared CSS once
    st.markdown(_TABLE_CSS, unsafe_allow_html=True)

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
                    blended_score = card.get("blended_score")
                else:
                    category = card.category
                    model_name = card.model_name
                    reason = card.reason
                    blended_score = getattr(card, "blended_score", None)
                if blended_score is not None:
                    body = f'<p style="color:grey">blended score {blended_score:.3f}</p>'
                else:
                    body = f"<p>{_esc(reason)}</p>"
                st.markdown(
                    f'<div class="metric-card"><h4>{_esc(category)}</h4>'
                    f"<p><strong>{_esc(model_name)}</strong></p>"
                    f"{body}</div>",
                    unsafe_allow_html=True,
                )

    # Collect benchmark names shown as columns in any tier table
    tier_column_names: set[str] = set()
    for tier in tier_tables:
        t = tier if isinstance(tier, dict) else tier.model_dump()
        tier_column_names.update(t.get("columns", []))

    # Benchmarks Used section
    _render_benchmarks_used(benchmarks_used, tier_column_names)
