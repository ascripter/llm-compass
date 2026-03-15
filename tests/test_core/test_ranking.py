"""Unit and integration tests for Req 2.3 Node 4: Scoring and Ranking."""

from datetime import date, datetime
from types import SimpleNamespace
from typing import cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from llm_compass.agentic_core.nodes.ranking import (
    _precompute_bridge_calibration,
    _calculate_blended_cost,
    _normalize_scores_to_0_1,
    execute_ranking,
    retrieve_and_rank_models,
)
from llm_compass.agentic_core.schemas.ranking import RankedLists
from llm_compass.agentic_core.state import AgentState
from llm_compass.data.models import Base, BenchmarkDictionary, BenchmarkScore, LLMMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_model(session: Session, name: str = "test-model", **kwargs) -> LLMMetadata:
    defaults = dict(
        name_normalized=name,
        model_type="instruct",
        provider="TestCo",
        release_date=date(2024, 1, 1),
        modality_input=["text"],
        modality_output=["text"],
        context_window=8192,
        speed_class="fast",
        speed_tps=100,
        is_open_weights=False,
        reasoning_type="none",
        tool_calling="none",
        is_outdated=False,
        cost_input_text_1m=1.0,
        cost_output_text_1m=2.0,
    )
    defaults.update(kwargs)
    model = LLMMetadata(**defaults)
    session.add(model)
    session.flush()
    return model


def make_benchmark(session: Session, name: str = "bench", variant: str = "", **kwargs) -> BenchmarkDictionary:
    benchmark = BenchmarkDictionary(
        name_normalized=name,
        variant=variant,
        description="Test benchmark",
        categories=["reasoning"],
        **kwargs,
    )
    session.add(benchmark)
    session.flush()
    return benchmark


def make_score(
    session: Session,
    model: LLMMetadata,
    benchmark: BenchmarkDictionary,
    score_value: float = 80.0,
    **kwargs,
) -> BenchmarkScore:
    score = BenchmarkScore(
        model_id=model.id,
        benchmark_id=benchmark.id,
        score_value=score_value,
        metric_unit="%",
        source_name="test",
        source_url="http://example.com",
        date_published=datetime(2024, 1, 1),
        original_model_name=model.name_normalized,
        original_benchmark_name=benchmark.name_normalized,
        **kwargs,
    )
    session.add(score)
    session.flush()
    return score


def _text_only_token_ratio() -> dict:
    return {
        "normalized_input_ratios": {"text": 0.5, "image": 0.0, "audio": 0.0, "video": 0.0},
        "normalized_output_ratios": {"text": 0.5, "image": 0.0, "audio": 0.0, "video": 0.0},
    }


# ---------------------------------------------------------------------------
# TestCalculateBlendedCost
# ---------------------------------------------------------------------------

class TestCalculateBlendedCost:
    def _model(self, **kwargs):
        defaults = dict(
            cost_input_text_1m=1.0,
            cost_output_text_1m=2.0,
            cost_input_image_1024=None,
            cost_output_image_1024=None,
            cost_input_audio_1h=None,
            cost_output_audio_1h=None,
            cost_input_video_1s=None,
            cost_output_video_1s=None,
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_text_only(self):
        model = self._model()
        ratio = {
            "normalized_input_ratios": {"text": 0.5},
            "normalized_output_ratios": {"text": 0.5},
        }
        cost, null_frac = _calculate_blended_cost(model, ratio)  # type: ignore[arg-type]
        # non_null_weight = 1.0, numerator = 0.5*1.0 + 0.5*2.0 = 1.5
        assert cost == pytest.approx(1.5)
        assert null_frac == pytest.approx(0.0)

    def test_all_modalities(self):
        model = self._model(
            cost_input_text_1m=1.0,
            cost_output_text_1m=2.0,
            cost_input_image_1024=3.0,
            cost_output_image_1024=4.0,
            cost_input_audio_1h=5.0,
            cost_output_audio_1h=6.0,
            cost_input_video_1s=7.0,
            cost_output_video_1s=8.0,
        )
        ratio = {
            "normalized_input_ratios": {"text": 0.125, "image": 0.125, "audio": 0.125, "video": 0.125},
            "normalized_output_ratios": {"text": 0.125, "image": 0.125, "audio": 0.125, "video": 0.125},
        }
        # numerator = 0.125*(1+3+5+7+2+4+6+8) = 0.125*36 = 4.5
        # non_null_weight = 1.0, so cost = 4.5
        cost, null_frac = _calculate_blended_cost(model, ratio)  # type: ignore[arg-type]
        assert cost == pytest.approx(4.5)
        assert null_frac == pytest.approx(0.0)

    def test_all_null_costs(self):
        model = self._model(cost_input_text_1m=None, cost_output_text_1m=None)
        ratio = {
            "normalized_input_ratios": {"text": 1.0},
            "normalized_output_ratios": {"text": 1.0},
        }
        cost, null_frac = _calculate_blended_cost(model, ratio)  # type: ignore[arg-type]
        assert cost == 0.0
        assert null_frac == pytest.approx(1.0)

    def test_empty_ratios(self):
        model = self._model()
        cost, null_frac = _calculate_blended_cost(model, {})  # type: ignore[arg-type]
        assert cost == 0.0
        assert null_frac == 0.0

    def test_partial_null_costs(self):
        """One cost present, one null → rescaled cost and 50% null fraction."""
        model = self._model(cost_input_text_1m=4.0, cost_output_text_1m=None)
        ratio = {
            "normalized_input_ratios": {"text": 0.5},
            "normalized_output_ratios": {"text": 0.5},
        }
        cost, null_frac = _calculate_blended_cost(model, ratio)  # type: ignore[arg-type]
        # numerator = 0.5*4.0 = 2.0, non_null_weight = 0.5
        # cost = 2.0/0.5 = 4.0 (rescaled over known data only)
        assert cost == pytest.approx(4.0)
        assert null_frac == pytest.approx(0.5)

    def test_mixed_modalities_with_some_null(self):
        """Text costs present, image costs null → null fraction reflects image weight."""
        model = self._model(
            cost_input_text_1m=2.0,
            cost_output_text_1m=4.0,
            cost_input_image_1024=None,
            cost_output_image_1024=None,
        )
        ratio = {
            "normalized_input_ratios": {"text": 0.3, "image": 0.2},
            "normalized_output_ratios": {"text": 0.3, "image": 0.2},
        }
        cost, null_frac = _calculate_blended_cost(model, ratio)  # type: ignore[arg-type]
        # non_null_weight = 0.3+0.3 = 0.6, null_weight = 0.2+0.2 = 0.4
        # numerator = 0.3*2.0 + 0.3*4.0 = 1.8
        # cost = 1.8/0.6 = 3.0
        assert cost == pytest.approx(3.0)
        assert null_frac == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# TestNormalizeScores
# ---------------------------------------------------------------------------

class TestNormalizeScores:
    def test_normal_range(self):
        result = _normalize_scores_to_0_1([0.0, 50.0, 100.0])
        assert result == pytest.approx([0.0, 0.5, 1.0])

    def test_all_equal(self):
        result = _normalize_scores_to_0_1([42.0, 42.0, 42.0])
        assert result == [0.5, 0.5, 0.5]

    def test_empty_list(self):
        assert _normalize_scores_to_0_1([]) == []

    def test_single_value(self):
        # Single value → all-equal edge case
        assert _normalize_scores_to_0_1([7.0]) == [0.5]

    def test_preserves_order(self):
        scores = [30.0, 10.0, 20.0]
        result = _normalize_scores_to_0_1(scores)
        assert result[1] < result[2] < result[0]


# ---------------------------------------------------------------------------
# TestRetrieveAndRankModels
# ---------------------------------------------------------------------------

class TestRetrieveAndRankModels:
    def test_empty_db_returns_empty_lists(self, db_session):
        result = retrieve_and_rank_models(
            benchmark_weights=[],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        assert result.top_performance == []
        assert result.balanced == []
        assert result.budget == []

    def test_no_models_pass_constraint_returns_empty(self, db_session):
        make_model(db_session, name="small-model", context_window=1000)
        bench = make_benchmark(db_session)
        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={"min_context_window": 100_000},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        assert result.top_performance == []

    def test_no_benchmark_data_returns_empty(self, db_session):
        make_model(db_session, name="model-no-scores")
        bench = make_benchmark(db_session)
        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        assert result.top_performance == []

    def test_three_lists_returned(self, db_session):
        bench = make_benchmark(db_session)
        for i, name in enumerate(["model-a", "model-b", "model-c"]):
            m = make_model(db_session, name=name, cost_input_text_1m=float(i + 1))
            make_score(db_session, m, bench, score_value=float(60 + i * 10))

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        assert len(result.top_performance) == 3
        assert len(result.balanced) == 3
        assert len(result.budget) == 3

    def test_top_performance_ordered_by_score(self, db_session):
        bench = make_benchmark(db_session)
        low = make_model(db_session, name="low-perf")
        high = make_model(db_session, name="high-perf")
        make_score(db_session, low, bench, score_value=50.0)
        make_score(db_session, high, bench, score_value=90.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        assert result.top_performance[0].name_normalized == "high-perf"

    def test_budget_favours_low_cost(self, db_session):
        bench = make_benchmark(db_session)
        # cheap model has slightly lower perf but much lower cost
        cheap = make_model(db_session, name="cheap", cost_input_text_1m=0.1, cost_output_text_1m=0.1)
        expensive = make_model(db_session, name="expensive", cost_input_text_1m=100.0, cost_output_text_1m=100.0)
        make_score(db_session, cheap, bench, score_value=75.0)
        make_score(db_session, expensive, bench, score_value=80.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        assert result.budget[0].name_normalized == "cheap"

    def test_rank_metrics_attached_to_all_entries(self, db_session):
        bench = make_benchmark(db_session)
        m = make_model(db_session)
        make_score(db_session, m, bench, score_value=70.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        for entries in [result.top_performance, result.balanced, result.budget]:
            for entry in entries:
                assert entry.rank_metrics is not None

    def test_internal_fields_removed(self, db_session):
        bench = make_benchmark(db_session)
        m = make_model(db_session)
        make_score(db_session, m, bench, score_value=70.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        for entries in [result.top_performance, result.balanced, result.budget]:
            for entry in entries:
                # Internal fields should not be top-level attributes on RankedModel
                assert not hasattr(entry, "raw_performance_score")
                assert not hasattr(entry, "blended_cost_1m_usd")
                # performance_index and blended_cost_index live inside rank_metrics
                assert entry.rank_metrics is not None
                assert entry.rank_metrics.performance_index is not None
                assert entry.rank_metrics.blended_cost_index is not None
                # cost_null_fraction should survive into output
                assert entry.cost_null_fraction is not None

    def test_cost_null_fraction_in_output(self, db_session):
        """Models with null cost fields get a non-zero cost_null_fraction."""
        bench = make_benchmark(db_session)
        # Model with only input text cost, output text cost is None
        m = make_model(
            db_session, name="partial-cost",
            cost_input_text_1m=5.0, cost_output_text_1m=None,
        )
        make_score(db_session, m, bench, score_value=70.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        for entries in [result.top_performance, result.balanced, result.budget]:
            entry = entries[0]
            assert entry.cost_null_fraction == pytest.approx(0.5)

    def test_cost_null_fraction_zero_when_all_present(self, db_session):
        """Models with all cost fields present get cost_null_fraction == 0."""
        bench = make_benchmark(db_session)
        m = make_model(
            db_session, name="full-cost",
            cost_input_text_1m=1.0, cost_output_text_1m=2.0,
        )
        make_score(db_session, m, bench, score_value=70.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        for entries in [result.top_performance, result.balanced, result.budget]:
            entry = entries[0]
            assert entry.cost_null_fraction == pytest.approx(0.0)

    def test_metadata_fields_present(self, db_session):
        bench = make_benchmark(db_session, name="mmlu")
        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 0.8, "name": "mmlu"}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        assert "applied_io_ratio" in result.metadata
        assert "benchmarks_used" in result.metadata
        assert "benchmark_weights" in result.metadata

    def test_constraint_context_window(self, db_session):
        bench = make_benchmark(db_session)
        small = make_model(db_session, name="small-ctx", context_window=4096)
        large = make_model(db_session, name="large-ctx", context_window=128000)
        make_score(db_session, small, bench, score_value=80.0)
        make_score(db_session, large, bench, score_value=80.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={"min_context_window": 100_000},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        names = [e.name_normalized for e in result.top_performance]
        assert "large-ctx" in names
        assert "small-ctx" not in names

    def test_constraint_deployment_local(self, db_session):
        bench = make_benchmark(db_session)
        cloud = make_model(db_session, name="cloud-model", is_open_weights=False)
        local = make_model(db_session, name="local-model", is_open_weights=True)
        make_score(db_session, cloud, bench, score_value=80.0)
        make_score(db_session, local, bench, score_value=80.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={"deployment": "local"},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        names = [e.name_normalized for e in result.top_performance]
        assert "local-model" in names
        assert "cloud-model" not in names

    def test_constraint_deployment_cloud(self, db_session):
        bench = make_benchmark(db_session)
        cloud = make_model(db_session, name="cloud-model", is_open_weights=False)
        local = make_model(db_session, name="local-model", is_open_weights=True)
        make_score(db_session, cloud, bench, score_value=80.0)
        make_score(db_session, local, bench, score_value=80.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={"deployment": "cloud"},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        names = [e.name_normalized for e in result.top_performance]
        assert "cloud-model" in names
        assert "local-model" not in names

    def test_constraint_reasoning_model(self, db_session):
        bench = make_benchmark(db_session)
        plain = make_model(db_session, name="plain-model", reasoning_type="none")
        reasoner = make_model(db_session, name="reasoner-model", reasoning_type="native cot")
        make_score(db_session, plain, bench, score_value=80.0)
        make_score(db_session, reasoner, bench, score_value=80.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={"reasoning_model": True},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        names = [e.name_normalized for e in result.top_performance]
        assert "reasoner-model" in names
        assert "plain-model" not in names

    def test_constraint_tool_calling(self, db_session):
        bench = make_benchmark(db_session)
        no_tools = make_model(db_session, name="no-tools", tool_calling="none")
        has_tools = make_model(db_session, name="has-tools", tool_calling="standard")
        make_score(db_session, no_tools, bench, score_value=80.0)
        make_score(db_session, has_tools, bench, score_value=80.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={"tool_calling": True},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        names = [e.name_normalized for e in result.top_performance]
        assert "has-tools" in names
        assert "no-tools" not in names

    def test_outdated_models_excluded(self, db_session):
        bench = make_benchmark(db_session)
        current = make_model(db_session, name="current-model", is_outdated=False)
        outdated = make_model(db_session, name="old-model", is_outdated=True)
        make_score(db_session, current, bench, score_value=80.0)
        make_score(db_session, outdated, bench, score_value=95.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        names = [e.name_normalized for e in result.top_performance]
        assert "old-model" not in names
        assert "current-model" in names

    def test_returns_ranked_lists_instance(self, db_session):
        bench = make_benchmark(db_session)
        m = make_model(db_session)
        make_score(db_session, m, bench, score_value=80.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        assert isinstance(result, RankedLists)

    def test_provider_present(self, db_session):
        bench = make_benchmark(db_session)
        m = make_model(db_session, name="provider-test", provider="TestCo")
        make_score(db_session, m, bench, score_value=80.0)

        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        assert result.top_performance[0].provider == "TestCo"


# ---------------------------------------------------------------------------
# TestBridgeModelCalibration
# ---------------------------------------------------------------------------

class TestBridgeModelCalibration:
    def test_no_bridge_models_returns_none(self, db_session):
        bench = make_benchmark(db_session, name="mmlu", variant="v1")
        m = make_model(db_session, name="target-model")
        make_score(db_session, m, bench, score_value=70.0)

        result, _ = _precompute_bridge_calibration(db_session, [bench.id])
        assert result[bench.id] is None

    def test_bridge_model_estimate(self, db_session):
        # Two benchmark variants for the same benchmark name
        bench_v1 = make_benchmark(db_session, name="mmlu", variant="v1")
        bench_v2 = make_benchmark(db_session, name="mmlu", variant="v2")

        # Two bridge models that have scores for BOTH variants
        # offset per bridge = v1_score - v2_score
        # bridge A: 90 - 80 = 10
        # bridge B: 70 - 60 = 10
        # median_offset = 10
        bridge_a = make_model(db_session, name="bridge-a")
        bridge_b = make_model(db_session, name="bridge-b")
        make_score(db_session, bridge_a, bench_v1, score_value=90.0)
        make_score(db_session, bridge_a, bench_v2, score_value=80.0)
        make_score(db_session, bridge_b, bench_v1, score_value=70.0)
        make_score(db_session, bridge_b, bench_v2, score_value=60.0)

        # Target model only has v2 score
        target = make_model(db_session, name="target")
        make_score(db_session, target, bench_v2, score_value=75.0)

        # Pre-compute bridge calibration for bench_v1 (target variant = v1)
        result, _ = _precompute_bridge_calibration(db_session, [bench_v1.id])

        calib = result[bench_v1.id]
        assert calib is not None
        median_offset, note = calib

        # offset = scores[other_variant] - scores[target_variant] = v2 - v1
        # bridge A: 80 - 90 = -10, bridge B: 60 - 70 = -10, median = -10
        # estimated v1 = other_score - median_offset = 75 - (-10) = 85
        other_score = 75.0  # target's v2 score
        assert other_score - median_offset == pytest.approx(85.0)
        assert note is not None
        assert "bridge" in note.lower()


# ---------------------------------------------------------------------------
# TestExecuteRanking
# ---------------------------------------------------------------------------

class TestExecuteRanking:
    def test_populates_ranked_results(self, db_session):
        bench = make_benchmark(db_session)
        m = make_model(db_session)
        make_score(db_session, m, bench, score_value=80.0)

        state = cast(AgentState, {
            "weighted_benchmarks": [{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            "constraints": {},
            "token_ratio_estimation": _text_only_token_ratio(),
        })
        config = {"configurable": {"session": db_session}}
        result = execute_ranking(state, config)

        assert "ranked_results" in result
        assert isinstance(result["ranked_results"], RankedLists)
        assert result["ranked_results"].top_performance is not None

    def test_empty_weighted_benchmarks_returns_empty_lists(self, db_session):
        state = cast(AgentState, {
            "weighted_benchmarks": [],
            "constraints": {},
            "token_ratio_estimation": _text_only_token_ratio(),
        })
        config = {"configurable": {"session": db_session}}
        result = execute_ranking(state, config)

        assert result["ranked_results"].top_performance == []
        assert result["ranked_results"].balanced == []
        assert result["ranked_results"].budget == []
