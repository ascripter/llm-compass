"""Unit tests for Req 2.3 Node 3(b): Benchmark Judgment."""

from typing import cast
from unittest.mock import MagicMock

import pytest

from llm_compass.agentic_core.nodes.benchmark_judgment import benchmark_judgment_node
from llm_compass.agentic_core.schemas.benchmark_judgment import (
    BenchmarkJudgment,
    BenchmarkJudgments,
)
from llm_compass.agentic_core.schemas.validate_intent import IntentExtraction
from llm_compass.agentic_core.state import AgentState
from llm_compass.config import Settings


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_bm(
    id: int = 1,
    name_normalized: str = "mmlu",
    variant: str = "MMLU-Pro",
    description: str = "Massive Multitask Language Understanding benchmark.",
    categories: list | None = None,
    weight: float = 0.85,
) -> dict:
    return {
        "id": id,
        "name_normalized": name_normalized,
        "variant": variant,
        "description": description,
        "categories": categories or ["reasoning", "knowledge"],
        "weight": weight,
    }


def _make_judgment(
    benchmark_id: int = 1,
    relevance_class: str = "strong_match",
    short_rationale: str = "Directly measures the needed reasoning capability.",
) -> BenchmarkJudgment:
    return BenchmarkJudgment(
        benchmark_id=benchmark_id,
        relevance_class=relevance_class,
        short_rationale=short_rationale,
    )


def _make_intent(
    input_modalities: list | None = None,
    output_modalities: list | None = None,
) -> IntentExtraction:
    return IntentExtraction(
        is_specific=True,
        intended_input_modalities=input_modalities or ["text"],
        intended_output_modalities=output_modalities or ["text"],
        clarification_needed=[],
    )


def _make_settings(response: BenchmarkJudgments) -> MagicMock:
    """Settings mock whose LLM returns *response* from structured_output.invoke()."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = response
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_settings = MagicMock(spec=Settings)
    mock_settings.make_llm.return_value = mock_llm
    return mock_settings


def _make_state(**kwargs) -> AgentState:
    defaults: dict = {
        "user_query": "Which LLM is best for answering complex knowledge questions?",
        "messages": [],
        "weighted_benchmarks": [_make_bm()],
        "intent_extraction": None,
        "benchmark_judgements": None,
    }
    defaults.update(kwargs)
    return cast(AgentState, defaults)


# LangGraph passes _config but the judgment node ignores it
_DUMMY_CONFIG: dict = {}


# ---------------------------------------------------------------------------
# Guard: empty / missing benchmarks
# ---------------------------------------------------------------------------


class TestEmptyBenchmarks:
    def test_returns_empty_judgments_and_skips_llm_when_no_benchmarks(self):
        settings = MagicMock(spec=Settings)
        result = benchmark_judgment_node(
            _make_state(weighted_benchmarks=[]), _DUMMY_CONFIG, settings=settings
        )

        assert result["benchmark_judgements"].judgments == []
        settings.make_llm.assert_not_called()

    def test_returns_empty_judgments_when_key_absent_from_state(self):
        state = cast(AgentState, {"user_query": "test", "messages": []})
        settings = MagicMock(spec=Settings)

        result = benchmark_judgment_node(state, _DUMMY_CONFIG, settings=settings)

        assert result["benchmark_judgements"].judgments == []
        settings.make_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Normal judgment output
# ---------------------------------------------------------------------------


class TestJudgmentOutput:
    def test_returns_judgment_object_in_state(self):
        response = BenchmarkJudgments(judgments=[_make_judgment(benchmark_id=1)])
        result = benchmark_judgment_node(
            _make_state(), _DUMMY_CONFIG, settings=_make_settings(response)
        )

        assert "benchmark_judgements" in result
        assert isinstance(result["benchmark_judgements"], BenchmarkJudgments)

    def test_preserves_all_judgment_fields(self):
        response = BenchmarkJudgments(
            judgments=[
                _make_judgment(1, "perfect_match", "Exact fit for the task."),
                _make_judgment(2, "no_match", "Unrelated to the task."),
            ]
        )
        benchmarks = [_make_bm(id=1), _make_bm(id=2)]
        result = benchmark_judgment_node(
            _make_state(weighted_benchmarks=benchmarks),
            _DUMMY_CONFIG,
            settings=_make_settings(response),
        )

        js = result["benchmark_judgements"].judgments
        assert js[0].benchmark_id == 1
        assert js[0].relevance_class == "perfect_match"
        assert js[0].short_rationale == "Exact fit for the task."
        assert js[1].benchmark_id == 2
        assert js[1].relevance_class == "no_match"

    def test_relevance_weights_match_all_classes(self):
        """CLASS_TO_WEIGHT mapping is applied correctly for every class."""
        classes = ["perfect_match", "strong_match", "partial_match", "weak_match", "no_match"]
        expected = [1.00, 0.75, 0.50, 0.25, 0.00]

        benchmarks = [_make_bm(id=i + 1) for i in range(len(classes))]
        response = BenchmarkJudgments(
            judgments=[
                _make_judgment(benchmark_id=i + 1, relevance_class=cls)
                for i, cls in enumerate(classes)
            ]
        )
        result = benchmark_judgment_node(
            _make_state(weighted_benchmarks=benchmarks),
            _DUMMY_CONFIG,
            settings=_make_settings(response),
        )

        for j, exp in zip(result["benchmark_judgements"].judgments, expected):
            assert j.relevance_weight == pytest.approx(exp)

    def test_includes_log_entry(self):
        response = BenchmarkJudgments(
            judgments=[
                _make_judgment(1, "strong_match"),
                _make_judgment(2, "no_match"),
            ]
        )
        result = benchmark_judgment_node(
            _make_state(weighted_benchmarks=[_make_bm(id=1), _make_bm(id=2)]),
            _DUMMY_CONFIG,
            settings=_make_settings(response),
        )

        assert "logs" in result
        assert len(result["logs"]) == 1
        log = result["logs"][0]
        assert "2" in log   # total judged
        assert "1" in log   # 1 relevant (strong_match counts, no_match doesn't)


# ---------------------------------------------------------------------------
# Intent handling
# ---------------------------------------------------------------------------


class TestIntentHandling:
    def _get_human_msg(self, mock_settings: MagicMock) -> str:
        structured_llm = mock_settings.make_llm.return_value.with_structured_output.return_value
        structured_llm.invoke.assert_called_once()
        messages = structured_llm.invoke.call_args[0][0]
        return messages[1].content  # SystemMessage is [0], HumanMessage is [1]

    def test_no_intent_omits_modality_lines(self):
        response = BenchmarkJudgments(judgments=[_make_judgment()])
        mock_settings = _make_settings(response)

        benchmark_judgment_node(
            _make_state(intent_extraction=None), _DUMMY_CONFIG, settings=mock_settings
        )

        content = self._get_human_msg(mock_settings)
        assert "Input modalities" not in content
        assert "Output modalities" not in content

    def test_intent_modalities_appear_in_human_message(self):
        intent = _make_intent(input_modalities=["image", "text"], output_modalities=["text"])
        response = BenchmarkJudgments(judgments=[_make_judgment()])
        mock_settings = _make_settings(response)

        benchmark_judgment_node(
            _make_state(intent_extraction=intent), _DUMMY_CONFIG, settings=mock_settings
        )

        content = self._get_human_msg(mock_settings)
        assert "Input modalities" in content
        assert "Output modalities" in content
        assert "image" in content

    def test_intent_as_dict_is_reconstructed(self):
        """Intent arriving as a plain dict (LangGraph checkpoint form) must be handled."""
        intent_dict = {
            "is_specific": True,
            "intended_input_modalities": ["text"],
            "intended_output_modalities": ["text"],
            "clarification_needed": [],
        }
        response = BenchmarkJudgments(judgments=[_make_judgment()])
        mock_settings = _make_settings(response)

        # Must not raise
        benchmark_judgment_node(
            _make_state(intent_extraction=intent_dict), _DUMMY_CONFIG, settings=mock_settings
        )

        content = self._get_human_msg(mock_settings)
        assert "Input modalities" in content


# ---------------------------------------------------------------------------
# Human message content
# ---------------------------------------------------------------------------


class TestHumanMessageContent:
    def _invoke_and_get_human(self, state: AgentState, mock_settings: MagicMock) -> str:
        benchmark_judgment_node(state, _DUMMY_CONFIG, settings=mock_settings)
        structured_llm = mock_settings.make_llm.return_value.with_structured_output.return_value
        return structured_llm.invoke.call_args[0][0][1].content

    def test_benchmark_id_name_variant_in_message(self):
        bm = _make_bm(id=42, name_normalized="hellaswag", variant="HellaSwag-v2")
        response = BenchmarkJudgments(judgments=[_make_judgment(benchmark_id=42)])
        mock_settings = _make_settings(response)

        content = self._invoke_and_get_human(_make_state(weighted_benchmarks=[bm]), mock_settings)

        assert "ID=42" in content
        assert "hellaswag" in content
        assert "HellaSwag-v2" in content

    def test_description_in_message(self):
        bm = _make_bm(description="Measures commonsense reasoning ability at scale.")
        response = BenchmarkJudgments(judgments=[_make_judgment()])
        mock_settings = _make_settings(response)

        content = self._invoke_and_get_human(_make_state(weighted_benchmarks=[bm]), mock_settings)

        assert "Measures commonsense reasoning ability at scale." in content

    def test_user_query_in_message(self):
        response = BenchmarkJudgments(judgments=[_make_judgment()])
        mock_settings = _make_settings(response)

        content = self._invoke_and_get_human(
            _make_state(user_query="Summarize long legal contracts"), mock_settings
        )

        assert "Summarize long legal contracts" in content

    def test_similarity_score_absent_from_message(self):
        """Embedding similarity score must NOT appear — prevents anchoring bias."""
        bm = _make_bm(weight=0.9876)
        response = BenchmarkJudgments(judgments=[_make_judgment()])
        mock_settings = _make_settings(response)

        content = self._invoke_and_get_human(_make_state(weighted_benchmarks=[bm]), mock_settings)

        assert "0.9876" not in content
        assert "Similarity" not in content
        assert "similarity" not in content

    def test_no_trailing_paren_after_name_without_variant(self):
        """Regression: stray ) was appended when variant was empty."""
        bm = _make_bm(name_normalized="arc", variant="")
        response = BenchmarkJudgments(judgments=[_make_judgment()])
        mock_settings = _make_settings(response)

        content = self._invoke_and_get_human(_make_state(weighted_benchmarks=[bm]), mock_settings)

        # "arc)" would be the bug; correct is just "arc"
        assert "arc)" not in content

    def test_multiline_description_newlines_stripped(self):
        """Newlines inside description must be collapsed to spaces."""
        bm = _make_bm(description="Line one.\nLine two.\nLine three.")
        response = BenchmarkJudgments(judgments=[_make_judgment()])
        mock_settings = _make_settings(response)

        content = self._invoke_and_get_human(_make_state(weighted_benchmarks=[bm]), mock_settings)

        # No raw newlines within a single description field
        assert "Line one.\nLine two." not in content
        assert "Line one. Line two. Line three." in content
