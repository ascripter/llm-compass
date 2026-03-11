"""Node 5: Synthesis (LLM + Deterministic).

Builds the final ``SynthesisOutput`` from ranked results and pipeline state.
The LLM generates natural-language parts (task summary, executive summary,
recommendation reasons, offset calibration note); deterministic helpers build
the comparison table, recommendation cards, citations, and warnings.
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from llm_compass.config import Settings
from ..schemas.ranking import RankedLists, RankedModel
from ..schemas.synthesis import (
    Citation,
    ComparisonTable,
    RecommendationCard,
    SynthesisLLMOutput,
    SynthesisOutput,
    Warning,
)
from ..state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def _build_comparison_table(ranked: RankedLists) -> ComparisonTable:
    """Fixed base columns + top-N benchmark names by weight as extra columns.

    Deduplicates models across all three ranking lists; sorts by blended_score desc.
    """
    # Collect unique models (deduplicate by model_id)
    seen: set[int] = set()
    all_models: list[RankedModel] = []
    for model_list in [ranked.top_performance, ranked.balanced, ranked.budget]:
        for m in model_list:
            if m.model_id not in seen:
                seen.add(m.model_id)
                all_models.append(m)

    all_models.sort(key=lambda m: m.rank_metrics.blended_score, reverse=True)

    # Determine top-N benchmark columns by weight across all models
    bench_weight: Dict[str, float] = {}
    for m in all_models:
        for br in m.benchmark_results:
            key = br.benchmark_name
            if br.benchmark_variant:
                key = f"{key} ({br.benchmark_variant})"
            if key not in bench_weight or br.weight_used > bench_weight[key]:
                bench_weight[key] = br.weight_used

    top_benchmarks = sorted(bench_weight, key=bench_weight.get, reverse=True)[:6]  # type: ignore[arg-type]

    columns = ["Model", "Provider", "Blended Score", "Cost Index", "Speed (tps)", "Est?"]
    columns.extend(top_benchmarks)

    rows: List[List[Any]] = []
    for m in all_models:
        has_estimated = any(br.is_estimated for br in m.benchmark_results)
        row: List[Any] = [
            m.name_normalized,
            m.provider,
            round(m.rank_metrics.blended_score, 3),
            round(m.rank_metrics.blended_cost_index, 3),
            m.speed_tps,
            "Yes" if has_estimated else "No",
        ]
        # Add benchmark score columns
        bench_scores: Dict[str, float] = {}
        for br in m.benchmark_results:
            key = br.benchmark_name
            if br.benchmark_variant:
                key = f"{key} ({br.benchmark_variant})"
            bench_scores[key] = br.score

        for bname in top_benchmarks:
            score = bench_scores.get(bname)
            row.append(round(score, 2) if score is not None else None)

        rows.append(row)

    return ComparisonTable(
        title="Model Comparison",
        columns=columns,
        rows=rows,
    )


def _extract_citations(ranked: RankedLists) -> list[Citation]:
    """Collect unique (benchmark_name, source_url) from non-estimated results."""
    seen_urls: set[str] = set()
    citations: list[Citation] = []
    idx = 0

    for model_list in [ranked.top_performance, ranked.balanced, ranked.budget]:
        for m in model_list:
            for br in m.benchmark_results:
                if br.is_estimated or not br.source_url:
                    continue
                if br.source_url in seen_urls:
                    continue
                seen_urls.add(br.source_url)
                idx += 1
                citations.append(
                    Citation(
                        id=f"cite-{idx}",
                        label=br.benchmark_name,
                        url=br.source_url,
                    )
                )

    return citations


def _generate_warnings(state: dict[str, Any], ranked: RankedLists) -> list[Warning]:
    """Produce data-quality warnings based on pipeline state and ranked results."""
    warnings: list[Warning] = []

    avg_sim = state.get("average_benchmark_similarity", 0.0)
    if isinstance(avg_sim, (int, float)) and avg_sim < 0.6:
        warnings.append(
            Warning(
                code="LOW_RELEVANCE",
                message=f"Average benchmark relevance is low ({avg_sim:.2f}). "
                "Results may not closely match the intended use case.",
            )
        )

    # Check top-3 models across all lists for cost and estimation issues
    for label, model_list in [
        ("top_performance", ranked.top_performance),
        ("balanced", ranked.balanced),
        ("budget", ranked.budget),
    ]:
        for m in model_list[:3]:
            if m.cost_null_fraction is not None and m.cost_null_fraction > 0.3:
                warnings.append(
                    Warning(
                        code="PARTIAL_COST_DATA",
                        message=f"{m.name_normalized} ({label}) has {m.cost_null_fraction:.0%} "
                        "missing cost data; cost ranking may be unreliable.",
                    )
                )
            if any(br.is_estimated for br in m.benchmark_results):
                warnings.append(
                    Warning(
                        code="ESTIMATED_SCORES",
                        message=f"{m.name_normalized} ({label}) uses estimated benchmark scores.",
                    )
                )

        if 0 < len(model_list) < 3:
            warnings.append(
                Warning(
                    code="FEW_CANDIDATES",
                    message=f"Only {len(model_list)} model(s) in the {label} list "
                    "(fewer than the expected 3).",
                )
            )

    return warnings


def _pick_recommendation_cards(
    ranked: RankedLists,
    llm_reasons: Dict[str, str] | None = None,
) -> list[RecommendationCard]:
    """Top-1 from each list, collapsing duplicates.

    If *llm_reasons* is provided, use LLM-generated reasons keyed by
    ``top_performance``, ``balanced``, ``budget``; fall back to the
    ranking node's ``reason_for_ranking`` otherwise.
    """
    seen: set[int] = set()
    cards: list[RecommendationCard] = []

    for key, label, model_list in [
        ("top_performance", "Top Performance", ranked.top_performance),
        ("balanced", "Balanced", ranked.balanced),
        ("budget", "Budget", ranked.budget),
    ]:
        if not model_list:
            continue
        m = model_list[0]
        if m.model_id in seen:
            continue
        seen.add(m.model_id)
        reason = (
            (llm_reasons or {}).get(key)
            or m.reason_for_ranking
        )
        cards.append(
            RecommendationCard(
                category=label,
                model_name=m.name_normalized,
                reason=reason,
            )
        )

    return cards


def _build_fallback_summary(state: dict[str, Any], ranked: RankedLists) -> str:
    """Build a markdown summary from pipeline results (no LLM call)."""
    parts: list[str] = []

    # Intent
    intent = state.get("intent_extraction")
    if intent is not None:
        if isinstance(intent, dict):
            reasoning = intent.get("reasoning", "")
        else:
            reasoning = getattr(intent, "reasoning", "")
        if reasoning:
            parts.append(f"## Your Task\n\n{reasoning}")

    # Recommendations per category
    categories = [
        ("top_performance", "Top Performance"),
        ("balanced", "Balanced"),
        ("budget", "Budget"),
    ]
    ranking_parts: list[str] = []
    for key, label in categories:
        models = getattr(ranked, key, [])
        if not models:
            continue
        ranking_parts.append(f"**{label}**")
        for i, m in enumerate(models[:3], 1):
            blended = m.rank_metrics.blended_score
            provider_str = f" ({m.provider})" if m.provider else ""
            ranking_parts.append(
                f"{i}. **{m.name_normalized}**{provider_str} — blended score: {blended:.3f}"
            )
            if m.reason_for_ranking:
                ranking_parts.append(f"   _{m.reason_for_ranking}_")

    if ranking_parts:
        parts.append("## Recommendations\n\n" + "\n".join(ranking_parts))

    if not parts:
        parts.append("Analysis complete. Ranking and recommendations are not yet available.")

    return "\n\n".join(parts)


def _assemble_summary_markdown(llm_out: SynthesisLLMOutput) -> str:
    """Assemble ``summary_markdown`` from LLM output per PRD Node 5 spec."""
    parts = [
        f"## Your Task\n\n{llm_out.task_summary}",
        f"## Recommendations\n\n{llm_out.executive_summary}",
    ]
    if llm_out.offset_calibration_note:
        parts.append(f"> **Note:** {llm_out.offset_calibration_note}")
    return "\n\n".join(parts)


def _has_estimated_scores(ranked: RankedLists) -> bool:
    """Return True if any model across all lists has estimated benchmark scores."""
    for model_list in [ranked.top_performance, ranked.balanced, ranked.budget]:
        for m in model_list:
            if any(br.is_estimated for br in m.benchmark_results):
                return True
    return False


def _build_ranking_context(ranked: RankedLists) -> str:
    """Build a concise text summary of ranked results for the LLM prompt."""
    lines: list[str] = []
    for key, label in [
        ("top_performance", "Top Performance"),
        ("balanced", "Balanced"),
        ("budget", "Budget"),
    ]:
        model_list = getattr(ranked, key, [])
        if not model_list:
            lines.append(f"\n### {label}: (no models)")
            continue
        lines.append(f"\n### {label}")
        for i, m in enumerate(model_list[:3], 1):
            rm = m.rank_metrics
            provider_str = f" ({m.provider})" if m.provider else ""
            lines.append(
                f"{i}. {m.name_normalized}{provider_str}"
                f" | perf={rm.performance_index:.3f}"
                f" | cost_idx={rm.blended_cost_index:.3f}"
                f" | blended={rm.blended_score:.3f}"
            )
            for br in m.benchmark_results:
                est_tag = " [ESTIMATED]" if br.is_estimated else ""
                lines.append(
                    f"   - {br.benchmark_name}"
                    f"{f' ({br.benchmark_variant})' if br.benchmark_variant else ''}"
                    f": {br.score:.2f} {br.metric_unit} (weight={br.weight_used:.2f}){est_tag}"
                )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """You are The Benchmark Analyst, producing the final summary \
for a developer who asked which LLM best fits their task.

You will receive:
- The user's original query and validated intent.
- Ranked model lists (top_performance, balanced, budget) with benchmark scores.

Your job is to produce ONLY the natural-language parts of the response. \
The comparison table, citations, and warnings are built deterministically — \
do NOT reproduce them.

Follow the output schema strictly."""


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------


def synthesis_node(state: AgentState, settings: Settings) -> dict:
    """Node 5 — Synthesis (LLM + Deterministic).

    Calls an LLM to generate natural-language summaries, then combines them
    with deterministically-built structured components.
    """
    ranked_raw = state.get("ranked_results")
    if ranked_raw is None:
        ranked = RankedLists()
    elif isinstance(ranked_raw, dict):
        ranked = RankedLists(**ranked_raw)
    else:
        ranked = ranked_raw

    has_models = bool(ranked.top_performance or ranked.balanced or ranked.budget)
    has_estimated = _has_estimated_scores(ranked)

    logger.debug(
        "synthesis_node ENTRY"
        " | user_query=%r"
        " | top_performance=%d | balanced=%d | budget=%d"
        " | has_estimated=%s",
        state.get("user_query", "")[:80],
        len(ranked.top_performance),
        len(ranked.balanced),
        len(ranked.budget),
        has_estimated,
    )

    # --- Deterministic components (always built) ---
    comparison_table = _build_comparison_table(ranked)
    citations = _extract_citations(ranked)
    warnings = _generate_warnings(state, ranked)

    logger.debug(
        "synthesis_node deterministic | table_rows=%d | citations=%d | warnings=%d",
        len(comparison_table.rows),
        len(citations),
        len(warnings),
    )

    # --- LLM call ---
    llm_output: SynthesisLLMOutput | None = None
    logs: list[str] = []

    if not has_models:
        logger.warning("synthesis_node | no ranked models — skipping LLM call, using fallback")
    else:
        try:
            llm_output = _invoke_synthesis_llm(state, ranked, settings)
            logger.debug(
                "synthesis_node LLM | task_summary_len=%d | exec_summary_len=%d"
                " | reasons_keys=%s | calibration_note=%s",
                len(llm_output.task_summary),
                len(llm_output.executive_summary),
                list(llm_output.recommendation_reasons.keys()),
                llm_output.offset_calibration_note is not None,
            )
            logs.append("Synthesis: LLM generated summary.")
        except Exception:
            logger.exception(
                "synthesis_node | LLM call failed; falling back to deterministic summary"
            )
            logs.append("Synthesis: LLM call failed, using deterministic fallback.")

    # --- Recommendation cards (use LLM reasons when available) ---
    recommendation_cards = _pick_recommendation_cards(
        ranked,
        llm_reasons=llm_output.recommendation_reasons if llm_output else None,
    )

    # --- Assemble summary markdown ---
    if llm_output is not None:
        summary = _assemble_summary_markdown(llm_output)
    else:
        summary = _build_fallback_summary(state, ranked)

    output = SynthesisOutput(
        llm_output=llm_output,
        summary_markdown=summary,
        comparison_table=comparison_table,
        recommendation_cards=recommendation_cards,
        citations=citations,
        warnings=warnings,
    )

    logger.info(
        "synthesis_node EXIT | llm_used=%s | cards=%d | citations=%d | warnings=%d",
        llm_output is not None,
        len(recommendation_cards),
        len(citations),
        len(warnings),
    )

    logs.append("Synthesis: generated final response.")
    return {
        "final_response": output,
        "logs": logs,
    }


def _invoke_synthesis_llm(
    state: dict[str, Any],
    ranked: RankedLists,
    settings: Settings,
) -> SynthesisLLMOutput:
    """Call the LLM to produce the natural-language synthesis components."""
    llm = settings.make_llm("openai/gpt-4o-mini", temperature=0.3)
    structured_llm = llm.with_structured_output(SynthesisLLMOutput)

    # Build context for the LLM
    user_query = state.get("user_query", "")
    intent = state.get("intent_extraction")
    intent_reasoning = ""
    if intent is not None:
        if isinstance(intent, dict):
            intent_reasoning = intent.get("reasoning", "")
        else:
            intent_reasoning = getattr(intent, "reasoning", "")

    has_estimated = _has_estimated_scores(ranked)
    ranking_context = _build_ranking_context(ranked)

    user_msg_parts = [
        f"User query: {user_query}",
        f"Validated intent: {intent_reasoning}",
        f"\nRanked results:{ranking_context}",
    ]
    if has_estimated:
        user_msg_parts.append(
            "\nSome scores are ESTIMATED (marked [ESTIMATED] above). "
            "You MUST provide an offset_calibration_note explaining which "
            "models/benchmarks were estimated."
        )
    else:
        user_msg_parts.append(
            "\nNo estimated scores. Set offset_calibration_note to null."
        )

    messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content="\n".join(user_msg_parts)),
    ]

    logger.debug(
        "_invoke_synthesis_llm | user_query=%r | intent_reasoning_len=%d"
        " | has_estimated=%s | ranking_context_len=%d",
        user_query[:80],
        len(intent_reasoning),
        has_estimated,
        len(ranking_context),
    )

    return structured_llm.invoke(messages)  # type: ignore[return-value]
