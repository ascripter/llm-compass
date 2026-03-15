"""Traceability UI components — pipeline step tracker and accumulated trace."""

import streamlit as st

# Ordered pipeline steps matching backend _NODE_LABELS in query.py
PIPELINE_STEPS = [
    "Analyzing intent",
    "Estimating token ratios",
    "Generating search queries",
    "Discovering benchmarks",
    "Judging benchmark relevance",
    "Ranking models",
    "Synthesizing response",
]

_ICONS = {
    "done": "✓",
    "running": "⟳",
    "pending": "○",
    "failed": "✗",
}

_SPINNER_CSS = """\
<style>
@keyframes spin { to { transform: rotate(360deg); } }
.step-spinner { display: inline-block; animation: spin 1s linear infinite; }
</style>
"""


def init_steps() -> list[dict]:
    """Return a fresh steps list: first step running, rest pending."""
    steps = [{"name": s, "status": "pending"} for s in PIPELINE_STEPS]
    steps[0]["status"] = "running"
    return steps


def _dot_row(steps: list[dict]) -> str:
    symbols = {
        "done": "●",
        "running": "⟳",
        "pending": "○",
        "failed": "✗",
    }
    return "──".join(symbols.get(s["status"], "○") for s in steps)


def _step_lines(steps: list[dict]) -> None:
    """Write step lines into the current container."""
    for step in steps:
        icon = _ICONS.get(step["status"], "○")
        st.write(f"{icon}  {step['name']}")


def render_live_tracker(steps: list[dict], is_complete: bool) -> None:
    """Render inside tracker_ph during and just after streaming.

    Shows the dot-row status header and a single open 'Execution trace' expander
    with step lines added as nodes complete.
    """
    if not steps:
        return

    current = next((s["name"] for s in steps if s["status"] == "running"), None)
    done_count = sum(1 for s in steps if s["status"] == "done")
    failed_count = sum(1 for s in steps if s["status"] == "failed")

    if is_complete:
        if failed_count:
            label = f"✗ {failed_count} step(s) failed · {done_count} done"
        else:
            label = f"✓ {done_count} of {len(steps)} steps complete"
    else:
        label = f"{current}..." if current else "Running..."

    st.markdown(f"`{_dot_row(steps)}`  {label}")

    with st.expander("Execution trace", expanded=not is_complete):
        for step in steps:
            if step["status"] != "pending":
                if step["status"] == "running":
                    st.markdown(
                        f'{_SPINNER_CSS}<span class="step-spinner">⟳</span>  {step["name"]}',
                        unsafe_allow_html=True,
                    )
                else:
                    icon = _ICONS.get(step["status"], "○")
                    st.write(f"{icon}  {step['name']}")


def render_accumulated_trace(trace_runs: list[dict]) -> None:
    """Render the full accumulated trace at the bottom of the page.

    Shows all runs (initial query + clarifications) in a single collapsed expander.
    Each run is labelled and includes step lines plus verbose log messages.
    """
    if not trace_runs:
        return

    with st.expander("Execution trace", expanded=False):
        for i, run in enumerate(trace_runs):
            if i > 0:
                st.divider()
            _step_lines(run["steps"])
            for msg in run.get("messages", []):
                st.code(msg)
