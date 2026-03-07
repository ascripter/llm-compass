"""Unit tests for Req 2.3 Node 2 (b) (Token Ratio Estimation)."""

from typing import cast
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from llm_compass.agentic_core.nodes.token_ratio_estimation import token_ratio_estimation_node
from llm_compass.agentic_core.schemas.token_ratio_estimation import ModalityUnits, TokenRatioEstimation
from llm_compass.agentic_core.state import AgentState
from llm_compass.common.schemas import Constraints
from llm_compass.config import Settings


def _make_state(messages: list | None = None) -> AgentState:
    msgs = messages or [HumanMessage(content="I need a model for long legal-document summarization")]
    return cast(
        AgentState,
        {
            "user_query": "I need a model for long legal-document summarization",
            "messages": msgs,
            "constraints": Constraints(min_context_window=0),
        },
    )


def _make_token_response() -> TokenRatioEstimation:
    return TokenRatioEstimation(
        reasoning="Input is large documents and output is a short summary.",
        input_units=ModalityUnits(text_word_count=6000, image_count=0, audio_minutes=0, video_minutes=0),
        output_units=ModalityUnits(text_word_count=400, image_count=0, audio_minutes=0, video_minutes=0),
    )


def _make_settings(token_response: TokenRatioEstimation) -> MagicMock:
    """Returns a mock Settings whose make_llm returns an LLM that yields *token_response*."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = token_response
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_settings = MagicMock(spec=Settings)
    mock_settings.make_llm.return_value = mock_llm
    return mock_settings


def test_token_ratio_estimation_returns_estimation():
    token_response = _make_token_response()
    result = token_ratio_estimation_node(_make_state(), settings=_make_settings(token_response))

    assert result["token_ratio_estimation"] is token_response


def test_token_ratio_estimation_computes_normalized_ratios():
    token_response = _make_token_response()
    result = token_ratio_estimation_node(_make_state(), settings=_make_settings(token_response))

    estimation: TokenRatioEstimation = result["token_ratio_estimation"]
    total = (
        sum(estimation.normalized_input_ratios.values())
        + sum(estimation.normalized_output_ratios.values())
    )
    assert abs(total - 1.0) < 1e-4


def test_token_ratio_estimation_adds_logs():
    token_response = _make_token_response()
    result = token_ratio_estimation_node(_make_state(), settings=_make_settings(token_response))

    assert "logs" in result
    assert any("Token Ratio Estimation" in entry for entry in result["logs"])


def test_token_ratio_estimation_logs_contain_ratios():
    token_response = _make_token_response()
    result = token_ratio_estimation_node(_make_state(), settings=_make_settings(token_response))

    assert any("input=" in entry and "output=" in entry for entry in result["logs"])


def test_token_ratio_estimation_uses_multi_turn_prompt():
    # Multi-turn history → system prompt should reference "consecutive clarification chat"
    messages = [
        HumanMessage(content="Summarize legal documents"),
        AIMessage(content="How long are the documents?"),
        HumanMessage(content="About 50 pages"),
    ]
    token_response = _make_token_response()
    mock_settings = _make_settings(token_response)
    token_ratio_estimation_node(_make_state(messages=messages), settings=mock_settings)

    invoke_args = mock_settings.make_llm.return_value.with_structured_output.return_value.invoke.call_args
    system_msg = invoke_args[0][0][0]
    assert "consecutive clarification chat" in system_msg.content


def test_token_ratio_estimation_zero_units_defaults():
    # All units zero → guard rail sets 0.5/0.5 text split
    estimation = TokenRatioEstimation(
        reasoning="All zeros",
        input_units=ModalityUnits(),
        output_units=ModalityUnits(),
    )

    assert estimation.normalized_input_ratios["text"] == 0.5
    assert estimation.normalized_output_ratios["text"] == 0.5
    assert estimation.normalized_input_ratios["image"] == 0.0
    assert estimation.normalized_output_ratios["image"] == 0.0


def test_token_ratio_estimation_mixed_modalities():
    # text=100 words (130 tok) + image=2 (2000 tok) input; text=50 words (65 tok) output
    # total = 2195 tokens
    estimation = TokenRatioEstimation(
        reasoning="Images with text description",
        input_units=ModalityUnits(text_word_count=100, image_count=2),
        output_units=ModalityUnits(text_word_count=50),
    )

    total = (
        sum(estimation.normalized_input_ratios.values())
        + sum(estimation.normalized_output_ratios.values())
    )
    assert abs(total - 1.0) < 1e-4
    assert estimation.normalized_input_ratios["image"] > estimation.normalized_input_ratios["text"]
