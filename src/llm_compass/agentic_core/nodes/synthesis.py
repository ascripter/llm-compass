"""Node 5: Synthesis (LLM + Deterministic).

Builds the final ``SynthesisOutput`` from ranked results and pipeline state.
The LLM generates natural-language parts (task summary, recommendation reasons,
offset calibration note); deterministic helpers build the tier tables,
recommendation cards, benchmarks used, citations, and warnings.
"""

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from llm_compass.config import Settings
from ..schemas.benchmark_judgment import BenchmarkJudgments
from ..schemas.ranking import RankedLists, RankedModel
from ..schemas.synthesis import (
    BenchmarkUsed,
    Citation,
    RecommendationCard,
    RecommendationReasons,
    SynthesisLLMOutput,
    SynthesisOutput,
    TierBenchmarkScore,
    TierTable,
    TierTableRow,
    Warning,
)
from ..state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def _select_benchmark_columns(
    benchmark_judgments: BenchmarkJudgments | None,
    weighted_benchmarks: list[dict],
) -> list[dict]:
    """Select which benchmarks become table columns based on judgment weights.

    Returns list of dicts: ``{benchmark_id, display_name, weight}``.
    Logic: pick ALL benchmarks in the highest relevance tier that has entries.
    If only 1 benchmark is in that tier, also add ALL from the next lower tier.
    """
    # Build id -> display info lookup
    bm_lookup: Dict[int, dict] = {}
    for b in weighted_benchmarks:
        bid = b.get("id")
        if bid is None:
            continue
        name = b.get("name_normalized") or str(bid)
        variant = b.get("variant")
        display = f"{name} ({variant})" if variant else name
        bm_lookup[int(bid)] = {"display_name": display, "name": name, "variant": variant}

    if benchmark_judgments is None or not benchmark_judgments.judgments:
        # Fallback: top-6 from weighted_benchmarks by discovery weight
        selected = []
        for b in weighted_benchmarks[:6]:
            bid = b.get("id")
            if bid is None:
                continue
            info = bm_lookup.get(int(bid), {})
            selected.append(
                {
                    "benchmark_id": int(bid),
                    "display_name": info.get("display_name", str(bid)),
                    "weight": b.get("weight", 0.0),
                }
            )
        return selected

    # Group judgments by weight tier (descending)
    weight_tiers: Dict[float, list[dict]] = {}
    for j in benchmark_judgments.judgments:
        w = j.relevance_weight
        if w <= 0.0:
            continue
        info = bm_lookup.get(j.benchmark_id, {})
        entry = {
            "benchmark_id": j.benchmark_id,
            "display_name": info.get("display_name", str(j.benchmark_id)),
            "weight": w,
        }
        weight_tiers.setdefault(w, []).append(entry)

    if not weight_tiers:
        return []

    # Sort tiers descending by weight
    sorted_weights = sorted(weight_tiers.keys(), reverse=True)
    selected = list(weight_tiers[sorted_weights[0]])

    # If only 1 benchmark in the top tier, add all from the next tier
    if len(selected) == 1 and len(sorted_weights) > 1:
        selected.extend(weight_tiers[sorted_weights[1]])

    return selected


def _format_speed(model: RankedModel) -> str:
    """Format speed_class with optional tps in brackets."""
    sc = model.speed_class or "unknown"
    if model.speed_tps is not None:
        return f"{sc} ({model.speed_tps})"
    return sc


def _find_estimation_source(
    model: RankedModel, target_name: str, target_variant: str | None
) -> str | None:
    """Find the source benchmark variant that an estimated score was derived from."""
    for br in model.benchmark_results:
        if br.benchmark_name == target_name and not br.is_estimated:
            # This is a non-estimated result for the same benchmark name but different variant
            if br.benchmark_variant != target_variant:
                if br.benchmark_variant:
                    return f"{br.benchmark_name} ({br.benchmark_variant})"
                return br.benchmark_name
    return None


def _build_tier_tables(
    ranked: RankedLists,
    selected_columns: list[dict],
    balanced_perf_weight: float = 0.5,
    budget_perf_weight: float = 0.2,
) -> list[TierTable]:
    """Build 3 TierTable objects (Top Performance, Balanced, Budget Picks)."""
    column_ids = [c["benchmark_id"] for c in selected_columns]
    column_names = [c["display_name"] for c in selected_columns]

    captions = {
        "Top Performance": "Pure benchmark performance, cost ignored.",
        "Balanced": (
            f"Weighted {balanced_perf_weight:.0%} performance, "
            f"{1 - balanced_perf_weight:.0%} cost (normalized)"
        ),
        "Budget Picks": (
            f"Weighted {budget_perf_weight:.0%} performance, "
            f"{1 - budget_perf_weight:.0%} cost (normalized)"
        ),
    }

    tables: list[TierTable] = []
    for tier_key, tier_name, model_list in [
        ("top_performance", "Top Performance", ranked.top_performance),
        ("balanced", "Balanced", ranked.balanced),
        ("budget", "Budget Picks", ranked.budget),
    ]:
        top5 = sorted(model_list[:5], key=lambda m: m.rank_metrics.blended_score, reverse=True)

        rows: list[TierTableRow] = []
        for m in top5:
            # Build benchmark score lookup: benchmark_id -> BenchmarkResult
            br_by_id: Dict[int, Any] = {}
            for br in m.benchmark_results:
                br_by_id[br.benchmark_id] = br

            bench_scores: list[TierBenchmarkScore] = []
            for bid in column_ids:
                br = br_by_id.get(bid)
                if br is None:
                    bench_scores.append(TierBenchmarkScore(value=None))
                else:
                    est_source = None
                    if br.is_estimated:
                        est_source = _find_estimation_source(
                            m, br.benchmark_name, br.benchmark_variant
                        )
                    bench_scores.append(
                        TierBenchmarkScore(
                            value=round(br.score, 2),
                            is_estimated=br.is_estimated,
                            estimation_source=est_source,
                        )
                    )

            rows.append(
                TierTableRow(
                    model_name=m.name_normalized,
                    provider=m.provider,
                    speed=_format_speed(m),
                    score=round(m.rank_metrics.blended_score, 3),
                    benchmark_scores=bench_scores,
                )
            )

        tables.append(
            TierTable(
                tier_name=tier_name,
                caption=captions.get(tier_name, ""),
                columns=column_names,
                rows=rows,
            )
        )

    return tables


def _build_benchmarks_used(
    benchmark_judgments: BenchmarkJudgments | None,
    weighted_benchmarks: list[dict],
) -> list[BenchmarkUsed]:
    """Build 'Benchmarks Used' reference table from judgment + discovery data."""
    if benchmark_judgments is None or not benchmark_judgments.judgments:
        return []

    # Build id -> info lookup from weighted_benchmarks
    bm_lookup: Dict[int, dict] = {}
    for b in weighted_benchmarks:
        bid = b.get("id")
        if bid is None:
            continue
        bm_lookup[int(bid)] = {
            "name_normalized": b.get("name_normalized", ""),
            "variant": b.get("variant"),
            "description": b.get("description", ""),
        }

    entries: list[BenchmarkUsed] = []
    for j in benchmark_judgments.judgments:
        if j.relevance_weight <= 0.0:
            continue
        info = bm_lookup.get(j.benchmark_id, {})
        name = info.get("name_normalized", str(j.benchmark_id))
        variant = info.get("variant")
        display_name = f"{name} ({variant})" if variant else name

        entries.append(
            BenchmarkUsed(
                benchmark_name=display_name,
                weight=j.relevance_weight,
                description=info.get("description", ""),
            )
        )

    entries.sort(key=lambda e: e.weight, reverse=True)
    return entries


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

    best_weight = state.get("best_benchmark_weight", 0.0)
    if isinstance(best_weight, (int, float)) and best_weight < 0.6:
        warnings.append(
            Warning(
                code="LOW_RELEVANCE",
                message=f"Best benchmark relevance is low ({best_weight:.2f}). "
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
    llm_reasons: RecommendationReasons | None = None,
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
            cards.append(
                RecommendationCard(
                    category=label,
                    model_name=m.name_normalized,
                    reason="",
                    blended_score=m.rank_metrics.blended_score,
                )
            )
            continue
        seen.add(m.model_id)
        reason = (
            getattr(llm_reasons, key, None) if llm_reasons else None
        ) or m.reason_for_ranking
        cards.append(
            RecommendationCard(
                category=label,
                model_name=m.name_normalized,
                reason=reason,
            )
        )

    return cards


def _build_fallback_summary(state: dict[str, Any]) -> str:
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

    if not parts:
        parts.append("Analysis complete. Ranking and recommendations are not yet available.")

    return "\n\n".join(parts)


def _assemble_summary_markdown(llm_out: SynthesisLLMOutput) -> str:
    """Assemble ``summary_markdown`` from LLM output per PRD Node 5 spec."""
    parts = [
        f"## Your Task\n\n{llm_out.task_summary}",
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


def _parse_benchmark_judgments(state: dict[str, Any]) -> BenchmarkJudgments | None:
    """Parse benchmark_judgements from state (may be dict or object)."""
    raw = state.get("benchmark_judgements")
    if raw is None:
        return None
    if isinstance(raw, BenchmarkJudgments):
        return raw
    if isinstance(raw, dict):
        try:
            return BenchmarkJudgments.model_validate(raw)
        except Exception:
            logger.warning("Could not parse benchmark_judgements", exc_info=True)
            return None
    return None


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

    # --- Parse benchmark judgment data from state ---
    benchmark_judgments = _parse_benchmark_judgments(state)
    weighted_benchmarks = state.get("weighted_benchmarks", [])

    # --- Extract ranking weights from constraints ---
    constraints_val = state.get("constraints") or {}
    if hasattr(constraints_val, "model_dump"):
        constraints_val = constraints_val.model_dump()
    balanced_w = constraints_val.get("balanced_perf_weight", 0.5)
    budget_w = constraints_val.get("budget_perf_weight", 0.2)

    # --- Deterministic components (always built) ---
    selected_columns = _select_benchmark_columns(benchmark_judgments, weighted_benchmarks)
    tier_tables = _build_tier_tables(ranked, selected_columns, balanced_w, budget_w)
    benchmarks_used = _build_benchmarks_used(benchmark_judgments, weighted_benchmarks)
    citations = _extract_citations(ranked)
    warnings = _generate_warnings(state, ranked)

    total_rows = sum(len(t.rows) for t in tier_tables)
    logger.debug(
        "synthesis_node deterministic | tier_table_rows=%d | benchmarks_used=%d"
        " | citations=%d | warnings=%d",
        total_rows,
        len(benchmarks_used),
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
                "synthesis_node LLM | task_summary_len=%d"
                " | reasons_keys=%s | calibration_note=%s",
                len(llm_output.task_summary),
                [
                    k
                    for k in ("top_performance", "balanced", "budget")
                    if getattr(llm_output.recommendation_reasons, k, None)
                ],
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
        summary = _build_fallback_summary(state)

    output = SynthesisOutput(
        llm_output=llm_output,
        summary_markdown=summary,
        tier_tables=tier_tables,
        recommendation_cards=recommendation_cards,
        benchmarks_used=benchmarks_used,
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
        user_msg_parts.append("\nNo estimated scores. Set offset_calibration_note to null.")

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
