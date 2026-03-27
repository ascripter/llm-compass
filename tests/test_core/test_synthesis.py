"""Unit tests for Req 2.3 Node 5: Synthesis."""

from unittest.mock import MagicMock

import pytest

from llm_compass.agentic_core.nodes.synthesis import (
    _assemble_summary_markdown,
    _build_benchmarks_used,
    _build_fallback_summary,
    _build_ranking_context,
    _build_tier_tables,
    _extract_citations,
    _generate_warnings,
    _has_estimated_scores,
    _pick_recommendation_cards,
    _select_benchmark_columns,
    synthesis_node,
)
from llm_compass.agentic_core.schemas.benchmark_judgment import (
    BenchmarkJudgment,
    BenchmarkJudgments,
)
from llm_compass.agentic_core.schemas.ranking import (
    BenchmarkResult,
    PerformanceCI,
    RankMetrics,
    RankedLists,
    RankedModel,
)
from llm_compass.agentic_core.schemas.synthesis import (
    Citation,
    RecommendationCard,
    RecommendationReasons,
    SynthesisLLMOutput,
    SynthesisOutput,
)
from llm_compass.config import Settings


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_benchmark_result(
    benchmark_id: int = 1,
    benchmark_name: str = "HumanEval",
    benchmark_variant: str | None = None,
    score: float = 80.0,
    metric_unit: str = "%",
    weight_used: float = 1.0,
    is_estimated: bool = False,
    source_url: str | None = "http://example.com",
    estimation_note: str | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_id=benchmark_id,
        benchmark_name=benchmark_name,
        benchmark_variant=benchmark_variant,
        score=score,
        metric_unit=metric_unit,
        weight_used=weight_used,
        is_estimated=is_estimated,
        source_url=source_url,
        estimation_note=estimation_note,
    )


def _make_ranked_model(
    model_id: int = 1,
    name: str = "model-a",
    provider: str = "ProvA",
    performance_index: float = 0.8,
    blended_cost_index: float = 0.5,
    blended_score: float = 0.8,
    speed_tps: float | None = 100.0,
    cost_null_fraction: float | None = 0.0,
    reason: str = "High benchmark score",
    benchmark_results: list[BenchmarkResult] | None = None,
) -> RankedModel:
    return RankedModel(
        model_id=model_id,
        name_normalized=name,
        provider=provider,
        speed_class="fast",
        speed_tps=speed_tps,
        cost_null_fraction=cost_null_fraction,
        rank_metrics=RankMetrics(
            performance_index=PerformanceCI(
                low=performance_index, mid=performance_index, high=performance_index
            ),
            blended_cost_index=blended_cost_index,
            blended_score=blended_score,
        ),
        benchmark_results=benchmark_results or [_make_benchmark_result()],
        reason_for_ranking=reason,
    )


def _make_ranked_lists(
    top_performance: list[RankedModel] | None = None,
    balanced: list[RankedModel] | None = None,
    budget: list[RankedModel] | None = None,
) -> RankedLists:
    return RankedLists(
        top_performance=top_performance or [],
        balanced=balanced or [],
        budget=budget or [],
    )


def _make_settings(llm_response: SynthesisLLMOutput) -> MagicMock:
    """Returns a mock Settings whose make_llm returns an LLM yielding *llm_response*."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = llm_response
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_settings = MagicMock(spec=Settings)
    mock_settings.make_llm.return_value = mock_llm
    return mock_settings


def _make_failing_settings() -> MagicMock:
    """Returns a mock Settings whose LLM call raises an exception."""
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("LLM unavailable")
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_settings = MagicMock(spec=Settings)
    mock_settings.make_llm.return_value = mock_llm
    return mock_settings


def _default_llm_output(**kwargs) -> SynthesisLLMOutput:
    defaults = dict(
        task_summary="Summarise legal documents using an LLM.",
        recommendation_reasons={
            "top_performance": "Best HumanEval score.",
            "balanced": "Good balance of cost and quality.",
            "budget": "Cheapest option with acceptable quality.",
        },
        offset_calibration_note=None,
    )
    defaults.update(kwargs)
    return SynthesisLLMOutput(**defaults)


def _make_state(**kwargs) -> dict:
    defaults: dict = {
        "user_query": "Which model is best for RAG on legal docs?",
        "messages": [],
        "ranked_results": None,
        "best_benchmark_weight": 0.8,
        "intent_extraction": {},
        "benchmark_judgements": None,
        "weighted_benchmarks": [],
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# TestSelectBenchmarkColumns
# ---------------------------------------------------------------------------


class TestSelectBenchmarkColumns:
    def test_no_judgments_falls_back_to_weighted_benchmarks(self):
        wb = [{"id": 1, "name_normalized": "MMLU", "variant": "5-shot", "weight": 0.9}]
        cols = _select_benchmark_columns(None, wb)
        assert len(cols) == 1
        assert cols[0]["display_name"] == "MMLU (5-shot)"

    def test_selects_highest_weight_tier(self):
        # 2 perfect_match benchmarks → only those selected (no next-tier expansion)
        judgments = BenchmarkJudgments(judgments=[
            BenchmarkJudgment(benchmark_id=1, relevance_class="perfect_match", short_rationale="direct"),
            BenchmarkJudgment(benchmark_id=2, relevance_class="perfect_match", short_rationale="also direct"),
            BenchmarkJudgment(benchmark_id=3, relevance_class="strong_match", short_rationale="related"),
        ])
        wb = [
            {"id": 1, "name_normalized": "A", "variant": None},
            {"id": 2, "name_normalized": "B", "variant": None},
            {"id": 3, "name_normalized": "C", "variant": None},
        ]
        cols = _select_benchmark_columns(judgments, wb)
        assert len(cols) == 2
        ids = {c["benchmark_id"] for c in cols}
        assert ids == {1, 2}

    def test_single_top_tier_adds_next_tier(self):
        judgments = BenchmarkJudgments(judgments=[
            BenchmarkJudgment(benchmark_id=1, relevance_class="perfect_match", short_rationale="direct"),
            BenchmarkJudgment(benchmark_id=2, relevance_class="strong_match", short_rationale="a"),
            BenchmarkJudgment(benchmark_id=3, relevance_class="strong_match", short_rationale="b"),
        ])
        wb = [
            {"id": 1, "name_normalized": "A", "variant": None},
            {"id": 2, "name_normalized": "B", "variant": None},
            {"id": 3, "name_normalized": "C", "variant": None},
        ]
        cols = _select_benchmark_columns(judgments, wb)
        ids = {c["benchmark_id"] for c in cols}
        assert ids == {1, 2, 3}

    def test_no_match_excluded(self):
        judgments = BenchmarkJudgments(judgments=[
            BenchmarkJudgment(benchmark_id=1, relevance_class="no_match", short_rationale="not relevant"),
        ])
        wb = [{"id": 1, "name_normalized": "A", "variant": None}]
        cols = _select_benchmark_columns(judgments, wb)
        assert cols == []


# ---------------------------------------------------------------------------
# TestBuildTierTables
# ---------------------------------------------------------------------------


class TestBuildTierTables:
    def test_three_tables_created(self):
        m = _make_ranked_model()
        ranked = _make_ranked_lists(top_performance=[m], balanced=[m], budget=[m])
        tables = _build_tier_tables(ranked, [])
        assert len(tables) == 3
        names = [t.tier_name for t in tables]
        assert names == ["Top Performance", "Balanced", "Budget Picks"]

    def test_top5_limit(self):
        models = [_make_ranked_model(model_id=i, name=f"m{i}") for i in range(8)]
        ranked = _make_ranked_lists(top_performance=models)
        tables = _build_tier_tables(ranked, [])
        assert len(tables[0].rows) == 5

    def test_rows_sorted_by_score_desc(self):
        low = _make_ranked_model(model_id=1, name="low", blended_score=0.3)
        high = _make_ranked_model(model_id=2, name="high", blended_score=0.9)
        ranked = _make_ranked_lists(top_performance=[low, high])
        tables = _build_tier_tables(ranked, [])
        assert tables[0].rows[0].model_name == "high"

    def test_speed_format_with_tps(self):
        m = _make_ranked_model(speed_tps=150)
        ranked = _make_ranked_lists(top_performance=[m])
        tables = _build_tier_tables(ranked, [])
        assert tables[0].rows[0].speed == "fast (150)"

    def test_speed_format_without_tps(self):
        m = _make_ranked_model(speed_tps=None)
        ranked = _make_ranked_lists(top_performance=[m])
        tables = _build_tier_tables(ranked, [])
        assert tables[0].rows[0].speed == "fast"

    def test_benchmark_score_populated(self):
        br = _make_benchmark_result(benchmark_id=1, score=85.5)
        m = _make_ranked_model(benchmark_results=[br])
        ranked = _make_ranked_lists(top_performance=[m])
        cols = [{"benchmark_id": 1, "display_name": "HumanEval", "weight": 1.0}]
        tables = _build_tier_tables(ranked, cols)
        assert tables[0].rows[0].benchmark_scores[0].value == 85.5

    def test_missing_benchmark_is_none(self):
        br = _make_benchmark_result(benchmark_id=1)
        m = _make_ranked_model(benchmark_results=[br])
        ranked = _make_ranked_lists(top_performance=[m])
        cols = [{"benchmark_id": 99, "display_name": "Other", "weight": 1.0}]
        tables = _build_tier_tables(ranked, cols)
        assert tables[0].rows[0].benchmark_scores[0].value is None

    def test_estimated_flag_set(self):
        br = _make_benchmark_result(benchmark_id=1, is_estimated=True, source_url=None, estimation_note="est")
        m = _make_ranked_model(benchmark_results=[br])
        ranked = _make_ranked_lists(top_performance=[m])
        cols = [{"benchmark_id": 1, "display_name": "HumanEval", "weight": 1.0}]
        tables = _build_tier_tables(ranked, cols)
        assert tables[0].rows[0].benchmark_scores[0].is_estimated is True


# ---------------------------------------------------------------------------
# TestBuildBenchmarksUsed
# ---------------------------------------------------------------------------


class TestBuildBenchmarksUsed:
    def test_empty_when_no_judgments(self):
        assert _build_benchmarks_used(None, []) == []

    def test_excludes_no_match(self):
        judgments = BenchmarkJudgments(judgments=[
            BenchmarkJudgment(benchmark_id=1, relevance_class="no_match", short_rationale="no"),
        ])
        wb = [{"id": 1, "name_normalized": "A", "variant": None, "description": "desc"}]
        result = _build_benchmarks_used(judgments, wb)
        assert result == []

    def test_includes_relevant_benchmarks(self):
        judgments = BenchmarkJudgments(judgments=[
            BenchmarkJudgment(benchmark_id=1, relevance_class="strong_match", short_rationale="yes"),
        ])
        wb = [{"id": 1, "name_normalized": "MMLU", "variant": "5-shot", "description": "General knowledge"}]
        result = _build_benchmarks_used(judgments, wb)
        assert len(result) == 1
        assert result[0].benchmark_name == "MMLU (5-shot)"
        assert result[0].weight == 0.75
        assert result[0].description == "General knowledge"

    def test_sorted_by_weight_desc(self):
        judgments = BenchmarkJudgments(judgments=[
            BenchmarkJudgment(benchmark_id=1, relevance_class="partial_match", short_rationale="a"),
            BenchmarkJudgment(benchmark_id=2, relevance_class="perfect_match", short_rationale="b"),
        ])
        wb = [
            {"id": 1, "name_normalized": "A", "variant": None, "description": "a"},
            {"id": 2, "name_normalized": "B", "variant": None, "description": "b"},
        ]
        result = _build_benchmarks_used(judgments, wb)
        assert result[0].weight > result[1].weight


# ---------------------------------------------------------------------------
# TestExtractCitations
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_empty_ranked_lists_returns_empty(self):
        assert _extract_citations(_make_ranked_lists()) == []

    def test_non_estimated_with_url_included(self):
        br = _make_benchmark_result(is_estimated=False, source_url="http://paper.com")
        model = _make_ranked_model(benchmark_results=[br])
        citations = _extract_citations(_make_ranked_lists(top_performance=[model]))
        assert len(citations) == 1
        assert citations[0].url == "http://paper.com"
        assert citations[0].label == "HumanEval"

    def test_estimated_score_excluded(self):
        br = _make_benchmark_result(
            is_estimated=True, source_url="http://paper.com", estimation_note="est"
        )
        model = _make_ranked_model(benchmark_results=[br])
        citations = _extract_citations(_make_ranked_lists(top_performance=[model]))
        assert citations == []

    def test_missing_url_excluded(self):
        # Non-estimated results without source_url are now forbidden by the schema validator.
        # The closest valid scenario: estimated result with no source_url is also excluded.
        br = _make_benchmark_result(is_estimated=True, source_url=None, estimation_note="est")
        model = _make_ranked_model(benchmark_results=[br])
        citations = _extract_citations(_make_ranked_lists(top_performance=[model]))
        assert citations == []

    def test_deduplication_by_url(self):
        br = _make_benchmark_result(source_url="http://shared.com")
        m1 = _make_ranked_model(model_id=1, benchmark_results=[br])
        m2 = _make_ranked_model(model_id=2, benchmark_results=[br])
        citations = _extract_citations(_make_ranked_lists(top_performance=[m1], balanced=[m2]))
        urls = [c.url for c in citations]
        assert urls.count("http://shared.com") == 1

    def test_ids_are_sequential(self):
        br1 = _make_benchmark_result(benchmark_name="A", source_url="http://a.com")
        br2 = _make_benchmark_result(benchmark_name="B", source_url="http://b.com")
        model = _make_ranked_model(benchmark_results=[br1, br2])
        citations = _extract_citations(_make_ranked_lists(top_performance=[model]))
        assert {c.id for c in citations} == {"cite-1", "cite-2"}

    def test_citations_collected_across_all_three_lists(self):
        br1 = _make_benchmark_result(benchmark_name="A", source_url="http://a.com")
        br2 = _make_benchmark_result(benchmark_name="B", source_url="http://b.com")
        br3 = _make_benchmark_result(benchmark_name="C", source_url="http://c.com")
        m1 = _make_ranked_model(model_id=1, benchmark_results=[br1])
        m2 = _make_ranked_model(model_id=2, benchmark_results=[br2])
        m3 = _make_ranked_model(model_id=3, benchmark_results=[br3])
        citations = _extract_citations(
            _make_ranked_lists(top_performance=[m1], balanced=[m2], budget=[m3])
        )
        assert len(citations) == 3


# ---------------------------------------------------------------------------
# TestGenerateWarnings
# ---------------------------------------------------------------------------


class TestGenerateWarnings:
    def test_no_warnings_for_clean_data(self):
        # FEW_CANDIDATES triggers when unique models < 5, so provide 5 unique models.
        models = [_make_ranked_model(model_id=i, cost_null_fraction=0.0) for i in range(1, 6)]
        ranked = _make_ranked_lists(
            top_performance=models,
            balanced=models,
            budget=models,
        )
        state = _make_state(best_benchmark_weight=0.9)
        warnings = _generate_warnings(state, ranked)
        assert warnings == []

    def test_low_relevance_warning_when_avg_sim_below_threshold(self):
        state = _make_state(best_benchmark_weight=0.4)
        warnings = _generate_warnings(state, _make_ranked_lists())
        codes = [w.code for w in warnings]
        assert "LOW_RELEVANCE" in codes

    def test_no_low_relevance_warning_at_boundary(self):
        state = _make_state(best_benchmark_weight=0.6)
        warnings = _generate_warnings(state, _make_ranked_lists())
        codes = [w.code for w in warnings]
        assert "LOW_RELEVANCE" not in codes

    def test_partial_cost_data_warning(self):
        model = _make_ranked_model(cost_null_fraction=0.5)
        ranked = _make_ranked_lists(top_performance=[model])
        warnings = _generate_warnings(_make_state(), ranked)
        codes = [w.code for w in warnings]
        assert "PARTIAL_COST_DATA" in codes

    def test_no_partial_cost_warning_at_boundary(self):
        model = _make_ranked_model(cost_null_fraction=0.3)
        ranked = _make_ranked_lists(top_performance=[model])
        warnings = _generate_warnings(_make_state(), ranked)
        codes = [w.code for w in warnings]
        assert "PARTIAL_COST_DATA" not in codes

    def test_estimated_scores_warning(self):
        br = _make_benchmark_result(is_estimated=True, source_url=None, estimation_note="est")
        model = _make_ranked_model(benchmark_results=[br])
        ranked = _make_ranked_lists(top_performance=[model])
        warnings = _generate_warnings(_make_state(), ranked)
        codes = [w.code for w in warnings]
        assert "ESTIMATED_SCORES" in codes

    def test_few_candidates_warning_when_one_model(self):
        ranked = _make_ranked_lists(top_performance=[_make_ranked_model()])
        warnings = _generate_warnings(_make_state(), ranked)
        codes = [w.code for w in warnings]
        assert "FEW_CANDIDATES" in codes

    def test_no_few_candidates_warning_for_empty_list(self):
        # Empty list → condition is `0 < len < 3` — should NOT trigger
        ranked = _make_ranked_lists(top_performance=[])
        warnings = _generate_warnings(_make_state(), ranked)
        codes = [w.code for w in warnings]
        assert "FEW_CANDIDATES" not in codes

    def test_few_candidates_not_triggered_for_five_models(self):
        # Threshold is < 5 unique models across all lists; exactly 5 should not trigger.
        models = [_make_ranked_model(model_id=i) for i in range(5)]
        ranked = _make_ranked_lists(top_performance=models)
        warnings = _generate_warnings(_make_state(), ranked)
        codes = [w.code for w in warnings]
        assert "FEW_CANDIDATES" not in codes

    def test_warnings_from_multiple_lists(self):
        # Models are deduplicated by model_id across all lists, so the same model
        # appearing in all 3 lists generates only 1 PARTIAL_COST_DATA warning.
        model = _make_ranked_model(model_id=1, cost_null_fraction=0.8)
        ranked = _make_ranked_lists(
            top_performance=[model],
            balanced=[model],
            budget=[model],
        )
        warnings = _generate_warnings(_make_state(), ranked)
        partial_warnings = [w for w in warnings if w.code == "PARTIAL_COST_DATA"]
        assert len(partial_warnings) == 1


# ---------------------------------------------------------------------------
# TestPickRecommendationCards
# ---------------------------------------------------------------------------


class TestPickRecommendationCards:
    def test_one_card_per_category(self):
        tp = _make_ranked_model(model_id=1, name="perf-winner")
        bal = _make_ranked_model(model_id=2, name="bal-winner")
        bud = _make_ranked_model(model_id=3, name="budget-winner")
        cards = _pick_recommendation_cards(
            _make_ranked_lists(top_performance=[tp], balanced=[bal], budget=[bud])
        )
        assert len(cards) == 3
        categories = {c.category for c in cards}
        assert categories == {"Top Performance", "Balanced", "Budget"}

    def test_duplicate_model_across_lists_collapsed(self):
        m = _make_ranked_model(model_id=1, name="only-model")
        ranked = _make_ranked_lists(top_performance=[m], balanced=[m], budget=[m])
        cards = _pick_recommendation_cards(ranked)
        assert len(cards) == 1

    def test_empty_list_skipped(self):
        m = _make_ranked_model(model_id=1)
        cards = _pick_recommendation_cards(
            _make_ranked_lists(top_performance=[m], balanced=[], budget=[])
        )
        assert len(cards) == 1
        assert cards[0].category == "Top Performance"

    def test_uses_llm_reasons_when_provided(self):
        m = _make_ranked_model(model_id=1, reason="generic reason")
        llm_reasons = RecommendationReasons(top_performance="LLM-generated reason")
        cards = _pick_recommendation_cards(
            _make_ranked_lists(top_performance=[m]),
            llm_reasons=llm_reasons,
        )
        assert cards[0].reason == "LLM-generated reason"

    def test_falls_back_to_rank_reason_when_llm_reasons_absent(self):
        m = _make_ranked_model(model_id=1, reason="rank reason")
        cards = _pick_recommendation_cards(_make_ranked_lists(top_performance=[m]))
        assert cards[0].reason == "rank reason"

    def test_falls_back_when_llm_reason_key_missing(self):
        m = _make_ranked_model(model_id=1, reason="rank reason")
        # llm_reasons has no "top_performance" value
        cards = _pick_recommendation_cards(
            _make_ranked_lists(top_performance=[m]),
            llm_reasons=RecommendationReasons(balanced="bal reason"),
        )
        assert cards[0].reason == "rank reason"

    def test_uses_top_1_model_only(self):
        winner = _make_ranked_model(model_id=1, name="winner", blended_score=0.9)
        runner_up = _make_ranked_model(model_id=2, name="runner-up", blended_score=0.5)
        cards = _pick_recommendation_cards(
            _make_ranked_lists(top_performance=[winner, runner_up])
        )
        assert len(cards) == 1
        assert cards[0].model_name == "winner"


# ---------------------------------------------------------------------------
# TestBuildFallbackSummary
# ---------------------------------------------------------------------------


class TestBuildFallbackSummary:
    def test_includes_intent_reasoning_when_present(self):
        state = _make_state(intent_extraction={"reasoning": "User needs legal RAG."})
        summary = _build_fallback_summary(state)
        assert "User needs legal RAG." in summary

    def test_includes_intent_reasoning_from_object(self):
        from types import SimpleNamespace

        intent = SimpleNamespace(reasoning="Object-style reasoning.")
        state = _make_state(intent_extraction=intent)
        summary = _build_fallback_summary(state)
        assert "Object-style reasoning." in summary

    def test_no_intent_no_task_section(self):
        state = _make_state(intent_extraction=None)
        summary = _build_fallback_summary(state)
        assert "## Your Task" not in summary

    def test_fallback_message_when_no_data(self):
        state = _make_state(intent_extraction=None)
        summary = _build_fallback_summary(state)
        assert "Analysis complete" in summary


# ---------------------------------------------------------------------------
# TestAssembleSummaryMarkdown
# ---------------------------------------------------------------------------


class TestAssembleSummaryMarkdown:
    def test_contains_task_section(self):
        llm_out = _default_llm_output(task_summary="A legal RAG task.")
        md = _assemble_summary_markdown(llm_out)
        assert "## Your Task" in md
        assert "A legal RAG task." in md

    def test_no_recommendations_section(self):
        llm_out = _default_llm_output()
        md = _assemble_summary_markdown(llm_out)
        assert "## Recommendations" not in md

    def test_calibration_note_included_when_present(self):
        llm_out = _default_llm_output(offset_calibration_note="Score for X was inferred.")
        md = _assemble_summary_markdown(llm_out)
        assert "Score for X was inferred." in md
        assert "> **Note:**" in md

    def test_calibration_note_absent_when_none(self):
        llm_out = _default_llm_output(offset_calibration_note=None)
        md = _assemble_summary_markdown(llm_out)
        assert "> **Note:**" not in md


# ---------------------------------------------------------------------------
# TestHasEstimatedScores
# ---------------------------------------------------------------------------


class TestHasEstimatedScores:
    def test_empty_lists_returns_false(self):
        assert _has_estimated_scores(_make_ranked_lists()) is False

    def test_all_non_estimated_returns_false(self):
        model = _make_ranked_model(benchmark_results=[_make_benchmark_result(is_estimated=False)])
        assert _has_estimated_scores(_make_ranked_lists(top_performance=[model])) is False

    def test_one_estimated_returns_true(self):
        br = _make_benchmark_result(is_estimated=True, source_url=None, estimation_note="note")
        model = _make_ranked_model(benchmark_results=[br])
        assert _has_estimated_scores(_make_ranked_lists(top_performance=[model])) is True

    def test_estimated_in_budget_list_detected(self):
        br = _make_benchmark_result(is_estimated=True, source_url=None, estimation_note="note")
        model = _make_ranked_model(benchmark_results=[br])
        assert _has_estimated_scores(_make_ranked_lists(budget=[model])) is True


# ---------------------------------------------------------------------------
# TestBuildRankingContext
# ---------------------------------------------------------------------------


class TestBuildRankingContext:
    def test_empty_lists_mentions_no_models(self):
        ctx = _build_ranking_context(_make_ranked_lists())
        assert "(no models)" in ctx

    def test_contains_model_name(self):
        m = _make_ranked_model(name="alpha-llm")
        ctx = _build_ranking_context(_make_ranked_lists(top_performance=[m]))
        assert "alpha-llm" in ctx

    def test_contains_benchmark_name(self):
        br = _make_benchmark_result(benchmark_name="MMLU")
        m = _make_ranked_model(benchmark_results=[br])
        ctx = _build_ranking_context(_make_ranked_lists(top_performance=[m]))
        assert "MMLU" in ctx

    def test_estimated_tag_present_for_estimated_scores(self):
        br = _make_benchmark_result(is_estimated=True, source_url=None, estimation_note="est")
        m = _make_ranked_model(benchmark_results=[br])
        ctx = _build_ranking_context(_make_ranked_lists(top_performance=[m]))
        assert "[ESTIMATED]" in ctx

    def test_non_estimated_has_no_estimated_tag(self):
        br = _make_benchmark_result(is_estimated=False)
        m = _make_ranked_model(benchmark_results=[br])
        ctx = _build_ranking_context(_make_ranked_lists(top_performance=[m]))
        assert "[ESTIMATED]" not in ctx

    def test_shows_at_most_3_models_per_list(self):
        models = [_make_ranked_model(model_id=i, name=f"m{i}") for i in range(5)]
        ctx = _build_ranking_context(_make_ranked_lists(top_performance=models))
        # models m0-m2 appear in context; m3, m4 are beyond the top-3 cutoff
        assert "m0" in ctx
        assert "m2" in ctx
        assert "m3" not in ctx


# ---------------------------------------------------------------------------
# TestSynthesisNode
# ---------------------------------------------------------------------------


class TestSynthesisNode:
    def _ranked_with_two_models(self) -> RankedLists:
        m1 = _make_ranked_model(model_id=1, name="perf-winner", blended_score=0.9)
        m2 = _make_ranked_model(model_id=2, name="budget-winner", blended_score=0.4)
        return _make_ranked_lists(
            top_performance=[m1],
            balanced=[m1],
            budget=[m2],
        )

    def test_returns_final_response_and_logs(self):
        ranked = self._ranked_with_two_models()
        state = _make_state(ranked_results=ranked)
        result = synthesis_node(state, settings=_make_settings(_default_llm_output()))
        assert "final_response" in result
        assert isinstance(result["final_response"], SynthesisOutput)
        assert "logs" in result
        assert len(result["logs"]) > 0

    def test_llm_output_stored_in_final_response(self):
        ranked = self._ranked_with_two_models()
        state = _make_state(ranked_results=ranked)
        result = synthesis_node(state, settings=_make_settings(_default_llm_output()))
        assert result["final_response"].llm_output is not None

    def test_summary_markdown_uses_llm_output(self):
        llm_out = _default_llm_output(task_summary="LLM-generated task summary.")
        ranked = self._ranked_with_two_models()
        state = _make_state(ranked_results=ranked)
        result = synthesis_node(state, settings=_make_settings(llm_out))
        assert "LLM-generated task summary." in result["final_response"].summary_markdown

    def test_recommendation_cards_use_llm_reasons(self):
        llm_out = _default_llm_output(
            recommendation_reasons={"top_performance": "LLM top reason."}
        )
        m = _make_ranked_model(model_id=1, reason="generic reason")
        ranked = _make_ranked_lists(top_performance=[m])
        state = _make_state(ranked_results=ranked)
        result = synthesis_node(state, settings=_make_settings(llm_out))
        cards: list[RecommendationCard] = result["final_response"].recommendation_cards
        top_card = next(c for c in cards if c.category == "Top Performance")
        assert top_card.reason == "LLM top reason."

    def test_llm_failure_uses_deterministic_fallback(self):
        ranked = self._ranked_with_two_models()
        state = _make_state(
            ranked_results=ranked,
            intent_extraction={"reasoning": "Legal RAG task."},
        )
        result = synthesis_node(state, settings=_make_failing_settings())
        output: SynthesisOutput = result["final_response"]
        assert output.llm_output is None
        # Deterministic fallback includes intent reasoning
        assert "Legal RAG task." in output.summary_markdown
        assert any("fallback" in log.lower() for log in result["logs"])

    def test_llm_failure_log_message_recorded(self):
        ranked = self._ranked_with_two_models()
        state = _make_state(ranked_results=ranked)
        result = synthesis_node(state, settings=_make_failing_settings())
        assert any("LLM call failed" in log for log in result["logs"])

    def test_no_models_skips_llm_call_and_uses_fallback(self):
        state = _make_state(ranked_results=_make_ranked_lists())
        mock_settings = _make_settings(_default_llm_output())
        result = synthesis_node(state, settings=mock_settings)
        # LLM should NOT have been called
        mock_settings.make_llm.assert_not_called()
        assert result["final_response"].llm_output is None

    def test_none_ranked_results_handled(self):
        state = _make_state(ranked_results=None)
        result = synthesis_node(state, settings=_make_settings(_default_llm_output()))
        assert isinstance(result["final_response"], SynthesisOutput)

    def test_dict_ranked_results_hydrated_from_checkpoint(self):
        """ranked_results stored as dict (from LangGraph checkpoint) must be coerced to RankedLists."""
        m = _make_ranked_model()
        ranked = _make_ranked_lists(top_performance=[m])
        state = _make_state(ranked_results=ranked.model_dump())
        result = synthesis_node(state, settings=_make_settings(_default_llm_output()))
        assert isinstance(result["final_response"], SynthesisOutput)

    def test_tier_tables_populated(self):
        ranked = self._ranked_with_two_models()
        state = _make_state(ranked_results=ranked)
        result = synthesis_node(state, settings=_make_settings(_default_llm_output()))
        tier_tables = result["final_response"].tier_tables
        assert len(tier_tables) == 3
        # top_performance has 1 model, balanced has 1, budget has 1
        assert len(tier_tables[0].rows) == 1

    def test_citations_populated(self):
        br = _make_benchmark_result(source_url="http://cited.com")
        m = _make_ranked_model(benchmark_results=[br])
        ranked = _make_ranked_lists(top_performance=[m])
        state = _make_state(ranked_results=ranked)
        result = synthesis_node(state, settings=_make_settings(_default_llm_output()))
        citations: list[Citation] = result["final_response"].citations
        assert any(c.url == "http://cited.com" for c in citations)

    def test_warnings_populated_for_low_similarity(self):
        ranked = self._ranked_with_two_models()
        state = _make_state(ranked_results=ranked, best_benchmark_weight=0.3)
        result = synthesis_node(state, settings=_make_settings(_default_llm_output()))
        codes = [w.code for w in result["final_response"].warnings]
        assert "LOW_RELEVANCE" in codes

    def test_log_contains_synthesis_complete(self):
        ranked = self._ranked_with_two_models()
        state = _make_state(ranked_results=ranked)
        result = synthesis_node(state, settings=_make_settings(_default_llm_output()))
        assert any("Generated final response" in log for log in result["logs"])

    def test_calibration_note_in_summary_when_estimated(self):
        br = _make_benchmark_result(
            is_estimated=True, source_url=None, estimation_note="via bridge"
        )
        m = _make_ranked_model(benchmark_results=[br])
        ranked = _make_ranked_lists(top_performance=[m])
        state = _make_state(ranked_results=ranked)
        llm_out = _default_llm_output(offset_calibration_note="Model X was estimated via bridge.")
        result = synthesis_node(state, settings=_make_settings(llm_out))
        assert "Model X was estimated via bridge." in result["final_response"].summary_markdown
