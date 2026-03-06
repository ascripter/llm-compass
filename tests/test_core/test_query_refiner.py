"""Unit tests for Req 2.3 Node 2 (Query Refiner)."""

from typing import cast
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from llm_compass.agentic_core.nodes.refine_query import query_refiner_node
from llm_compass.agentic_core.schemas.refine_query import QueryExpansion
from llm_compass.agentic_core.schemas.validate_intent import ModalityUnits, TokenRatioEstimation
from llm_compass.agentic_core.state import AgentState
from llm_compass.common.schemas import Constraints


def _make_state(constraints: Constraints | dict | None = None) -> AgentState:
    return cast(
        AgentState,
        {
            "user_query": "I need a model for long legal-document summarization",
            "messages": [
                HumanMessage(content="I need a model for long legal-document summarization")
            ],
            "constraints": constraints or Constraints(min_context_window=0),
        },
    )


def _make_token_response() -> TokenRatioEstimation:
    return TokenRatioEstimation(
        reasoning="Input is large documents and output is a short summary.",
        input_units=ModalityUnits(text_word_count=6000, image_count=0, audio_minutes=0, video_minutes=0),
        output_units=ModalityUnits(text_word_count=400, image_count=0, audio_minutes=0, video_minutes=0),
    )


def _patch_llm(token_response: TokenRatioEstimation, query_response: QueryExpansion):
    token_structured = MagicMock()
    token_structured.invoke.return_value = token_response

    query_structured = MagicMock()
    query_structured.invoke.return_value = query_response

    mock_llm = MagicMock()
    mock_llm.with_structured_output.side_effect = [token_structured, query_structured]

    return patch(
        "llm_compass.agentic_core.nodes.refine_query.ChatOpenAI",
        return_value=mock_llm,
    )


def test_query_refiner_returns_token_ratio_and_queries():
    token_response = _make_token_response()
    query_response = QueryExpansion(
        reasoning="Derived from summarization intent.",
        search_queries=[
            "long context summarization benchmark",
            "legal document QA benchmark",
            "document understanding llm benchmark",
        ],
    )

    with _patch_llm(token_response, query_response):
        result = query_refiner_node(_make_state())

    assert result["token_ratio_estimation"] is token_response
    assert len(result["search_queries"]) == 3
    assert all(isinstance(item, str) and item for item in result["search_queries"])


def test_query_refiner_adds_fallback_queries_when_llm_returns_too_few():
    token_response = _make_token_response()
    query_response = QueryExpansion(
        reasoning="Only one query returned by model.",
        search_queries=["legal summarization benchmark"],
    )

    with _patch_llm(token_response, query_response):
        result = query_refiner_node(_make_state())

    assert len(result["search_queries"]) >= 3
    assert any("benchmark" in query.lower() for query in result["search_queries"])


def test_query_refiner_supports_dict_constraints_from_checkpoint():
    token_response = _make_token_response()
    query_response = QueryExpansion(
        reasoning="Derived from query and constraints.",
        search_queries=[
            "long context summarization benchmark",
            "legal text reasoning benchmark",
            "document retrieval qa benchmark",
        ],
    )

    constraints = Constraints(min_context_window=0, modality_input=["text"], modality_output=["text"])
    with _patch_llm(token_response, query_response):
        result = query_refiner_node(_make_state(constraints=constraints.model_dump()))

    assert len(result["search_queries"]) == 3
    assert result["token_ratio_estimation"] is token_response


def test_query_refiner_adds_logs():
    token_response = _make_token_response()
    query_response = QueryExpansion(
        reasoning="Derived from summarization intent.",
        search_queries=[
            "long context summarization benchmark",
            "legal document QA benchmark",
            "document understanding llm benchmark",
        ],
    )

    with _patch_llm(token_response, query_response):
        result = query_refiner_node(_make_state())

    assert "logs" in result
    assert any("token ratios estimated" in entry for entry in result["logs"])
