"""Unit and integration tests for Req 2.3 Node 4: Scoring and Ranking."""

from datetime import date, datetime
from types import SimpleNamespace
from typing import cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from llm_compass.agentic_core.nodes.ranking import (
    _apply_bridge_model_calibration,
    _calculate_blended_cost,
    _normalize_scores_to_0_1,
    execute_ranking,
    retrieve_and_rank_models,
)
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
        result = _calculate_blended_cost(model, ratio)  # type: ignore[arg-type]
        assert result == pytest.approx(0.5 * 1.0 + 0.5 * 2.0)

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
            "normalized_input_ratios": {"text": 0.1, "image": 0.1, "audio": 0.1, "video": 0.1},
            "normalized_output_ratios": {"text": 0.1, "image": 0.1, "audio": 0.1, "video": 0.1},
        }
        expected = 0.1 * (1 + 3 + 5 + 7) + 0.1 * (2 + 4 + 6 + 8)
        assert _calculate_blended_cost(model, ratio) == pytest.approx(expected)

    def test_none_costs_treated_as_zero(self):
        model = self._model(cost_input_text_1m=None, cost_output_text_1m=None)
        ratio = {
            "normalized_input_ratios": {"text": 1.0},
            "normalized_output_ratios": {"text": 1.0},
        }
        assert _calculate_blended_cost(model, ratio) == 0.0  # type: ignore[arg-type]

    def test_empty_ratios(self):
        model = self._model()
        assert _calculate_blended_cost(model, {}) == 0.0  # type: ignore[arg-type]


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
        assert result["top_performance"] == []
        assert result["balanced"] == []
        assert result["budget"] == []

    def test_no_models_pass_constraint_returns_empty(self, db_session):
        make_model(db_session, name="small-model", context_window=1000)
        bench = make_benchmark(db_session)
        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={"min_context_window": 100_000},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        assert result["top_performance"] == []

    def test_no_benchmark_data_returns_empty(self, db_session):
        make_model(db_session, name="model-no-scores")
        bench = make_benchmark(db_session)
        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 1.0, "name": bench.name_normalized}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        assert result["top_performance"] == []

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
        assert len(result["top_performance"]) == 3
        assert len(result["balanced"]) == 3
        assert len(result["budget"]) == 3

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
        assert result["top_performance"][0]["name_normalized"] == "high-perf"

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
        assert result["budget"][0]["name_normalized"] == "cheap"

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
        for list_name in ["top_performance", "balanced", "budget"]:
            for entry in result[list_name]:
                assert "rank_metrics" in entry
                assert "reason_for_ranking" in entry

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
        for list_name in ["top_performance", "balanced", "budget"]:
            for entry in result[list_name]:
                assert "raw_performance_score" not in entry
                assert "blended_cost_1m_usd" not in entry
                assert "performance_index" not in entry
                assert "blended_cost_index" not in entry

    def test_metadata_fields_present(self, db_session):
        bench = make_benchmark(db_session, name="mmlu")
        result = retrieve_and_rank_models(
            benchmark_weights=[{"id": bench.id, "weight": 0.8, "name": "mmlu"}],
            constraints={},
            token_ratio_estimation=_text_only_token_ratio(),
            session=db_session,
        )
        assert "applied_io_ratio" in result["metadata"]
        assert "benchmarks_used" in result["metadata"]
        assert "benchmark_weights" in result["metadata"]

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
        names = [e["name_normalized"] for e in result["top_performance"]]
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
        names = [e["name_normalized"] for e in result["top_performance"]]
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
        names = [e["name_normalized"] for e in result["top_performance"]]
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
        names = [e["name_normalized"] for e in result["top_performance"]]
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
        names = [e["name_normalized"] for e in result["top_performance"]]
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
        names = [e["name_normalized"] for e in result["top_performance"]]
        assert "old-model" not in names
        assert "current-model" in names


# ---------------------------------------------------------------------------
# TestBridgeModelCalibration
# ---------------------------------------------------------------------------

class TestBridgeModelCalibration:
    def test_no_bridge_models_returns_none(self, db_session):
        bench = make_benchmark(db_session, name="mmlu", variant="v1")
        m = make_model(db_session, name="target-model")
        make_score(db_session, m, bench, score_value=70.0)

        score, note = _apply_bridge_model_calibration(
            session=db_session,
            model_id=m.id,
            benchmark_id=bench.id,
            variant="v1",
        )
        assert score is None
        assert note is None

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

        # Estimate v1 score for target (it has v2 but not v1)
        estimated, note = _apply_bridge_model_calibration(
            session=db_session,
            model_id=target.id,
            benchmark_id=bench_v1.id,
            variant="v1",
        )

        # Code computes: other_score - median_offset = 75 - 10 = 65
        assert estimated == pytest.approx(65.0)
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
        result = execute_ranking(state, session=db_session)

        assert "ranked_results" in result
        assert "top_performance" in result["ranked_results"]

    def test_empty_weighted_benchmarks_returns_empty_lists(self, db_session):
        state = cast(AgentState, {
            "weighted_benchmarks": [],
            "constraints": {},
            "token_ratio_estimation": _text_only_token_ratio(),
        })
        result = execute_ranking(state, session=db_session)

        assert result["ranked_results"]["top_performance"] == []
        assert result["ranked_results"]["balanced"] == []
        assert result["ranked_results"]["budget"] == []
