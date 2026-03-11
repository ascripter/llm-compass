"""Node 5: Synthesis — deterministic stub (no LLM call yet).

Builds the final ``SynthesisOutput`` from ranked results and pipeline state.
The LLM-generated natural-language parts (SynthesisLLMOutput) will be added
in a future iteration; for now we produce a fallback markdown summary and
populate all deterministic components.
"""

import logging
from typing import Any, Dict, List

from langgraph.types import RunnableConfig

from ..schemas.ranking import RankedLists, RankedModel
from ..schemas.synthesis import (
    Citation,
    ComparisonTable,
    RecommendationCard,
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


def _pick_recommendation_cards(ranked: RankedLists) -> list[RecommendationCard]:
    """Top-1 from each list, collapsing duplicates."""
    seen: set[int] = set()
    cards: list[RecommendationCard] = []

    for label, model_list in [
        ("Top Performance", ranked.top_performance),
        ("Balanced", ranked.balanced),
        ("Budget", ranked.budget),
    ]:
        if not model_list:
            continue
        m = model_list[0]
        if m.model_id in seen:
            continue
        seen.add(m.model_id)
        cards.append(
            RecommendationCard(
                category=label,
                model_name=m.name_normalized,
                reason=m.reason_for_ranking,
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


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------


def synthesis_node(state: AgentState, config: RunnableConfig) -> dict:
    """Node 5 — deterministic stub.

    Builds final ``SynthesisOutput`` from ranked results and pipeline state.
    """
    ranked = state.get("ranked_results")
    if ranked is None:
        ranked = RankedLists()

    comparison_table = _build_comparison_table(ranked)
    citations = _extract_citations(ranked)
    warnings = _generate_warnings(state, ranked)
    recommendation_cards = _pick_recommendation_cards(ranked)
    summary = _build_fallback_summary(state, ranked)

    output = SynthesisOutput(
        summary_markdown=summary,
        comparison_table=comparison_table,
        recommendation_cards=recommendation_cards,
        citations=citations,
        warnings=warnings,
    )

    logger.info(
        "Synthesis complete | cards=%d | citations=%d | warnings=%d",
        len(recommendation_cards),
        len(citations),
        len(warnings),
    )

    return {
        "final_response": output,
        "logs": ["Synthesis: generated final response."],
    }
