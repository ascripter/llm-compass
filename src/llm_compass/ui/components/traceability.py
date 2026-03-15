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

# Sets of step indices that execute in parallel (LangGraph fan-out/fan-in)
PARALLEL_GROUPS: list[set[int]] = [{1, 2}]


def parallel_group_for(idx: int) -> set[int] | None:
    """Return the parallel group containing step index *idx*, or ``None``."""
    for group in PARALLEL_GROUPS:
        if idx in group:
            return group
    return None


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
    steps = [{"name": s, "status": "pending", "logs": []} for s in PIPELINE_STEPS]
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


def _format_step_line(step: dict) -> str:
    """Format a single step as icon + name + optional inline logs."""
    icon = _ICONS.get(step["status"], "○")
    logs = step.get("logs", [])
    if logs:
        return f"{icon}  {step['name']}: {' | '.join(logs)}"
    return f"{icon}  {step['name']}"


def _step_lines(steps: list[dict]) -> None:
    """Write step lines into the current container."""
    for step in steps:
        if step["status"] == "pending":
            continue
        st.write(_format_step_line(step))


def render_live_tracker(
    steps: list[dict], is_complete: bool, prior_runs: list[dict] | None = None,
) -> None:
    """Render inside tracker_ph during and just after streaming.

    Shows the dot-row status header and a single open 'Execution trace' expander
    with step lines added as nodes complete.  When *prior_runs* is provided (e.g.
    from a previous query before a clarification), those runs are rendered first
    inside the same expander so only one "Execution trace" section ever appears.
    """
    if not steps:
        return

    running_names = [s["name"] for s in steps if s["status"] == "running"]
    done_count = sum(1 for s in steps if s["status"] == "done")
    failed_count = sum(1 for s in steps if s["status"] == "failed")

    if is_complete:
        if failed_count:
            label = f"✗ {failed_count} step(s) failed · {done_count} done"
        else:
            label = f"✓ {done_count} of {len(steps)} steps complete"
    else:
        label = (" | ".join(running_names) + "...") if running_names else "Running..."

    st.markdown(f"`{_dot_row(steps)}`  {label}")

    with st.expander("Execution trace", expanded=True):
        if prior_runs:
            for run in prior_runs:
                _step_lines(run["steps"])
            st.divider()
        rendered_groups: set[frozenset[int]] = set()
        for i, step in enumerate(steps):
            if step["status"] == "pending":
                continue

            group = parallel_group_for(i)
            if group is not None:
                group_key = frozenset(group)
                if group_key in rendered_groups:
                    continue  # already rendered this group's steps
                rendered_groups.add(group_key)

                group_steps = [steps[j] for j in sorted(group)]
                all_running = all(s["status"] == "running" for s in group_steps)

                if all_running:
                    # Combined spinner line for parallel nodes
                    combined = " | ".join(s["name"] for s in group_steps)
                    st.markdown(
                        f'{_SPINNER_CSS}<span class="step-spinner">⟳</span>  {combined}',
                        unsafe_allow_html=True,
                    )
                else:
                    # Mixed state: render each group member individually
                    for gs in group_steps:
                        if gs["status"] == "pending":
                            continue
                        if gs["status"] == "running":
                            st.markdown(
                                f'{_SPINNER_CSS}<span class="step-spinner">⟳</span>  {gs["name"]}',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.write(_format_step_line(gs))
            else:
                # Sequential step — unchanged behaviour
                if step["status"] == "running":
                    st.markdown(
                        f'{_SPINNER_CSS}<span class="step-spinner">⟳</span>  {step["name"]}',
                        unsafe_allow_html=True,
                    )
                else:
                    st.write(_format_step_line(step))


def render_accumulated_trace(trace_runs: list[dict]) -> None:
    """Render the full accumulated trace at the bottom of the page.

    Shows all runs (initial query + clarifications) in a single collapsed expander.
    Each run is labelled and includes step lines with inline log messages.
    """
    if not trace_runs:
        return

    with st.expander("Execution trace", expanded=True):
        for i, run in enumerate(trace_runs):
            if i > 0:
                st.divider()
            _step_lines(run["steps"])
