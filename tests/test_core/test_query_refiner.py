"""Unit tests for Req 2.3 Node 2 (a) (Query Refiner)."""

from typing import cast
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from llm_compass.agentic_core.nodes.refine_query import query_refiner_node
from llm_compass.agentic_core.schemas.refine_query import QueryExpansion
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


def _patch_llm(query_response: QueryExpansion):
    query_structured = MagicMock()
    query_structured.invoke.return_value = query_response

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = query_structured

    return patch(
        "llm_compass.agentic_core.nodes.refine_query.ChatOpenAI",
        return_value=mock_llm,
    )


def test_query_refiner_returns_search_queries():
    query_response = QueryExpansion(
        reasoning="Derived from summarization intent.",
        search_queries=[
            "long context summarization benchmark",
            "legal document QA benchmark",
            "document understanding llm benchmark",
        ],
    )

    with _patch_llm(query_response):
        result = query_refiner_node(_make_state())

    assert len(result["search_queries"]) == 3
    assert all(isinstance(item, str) and item for item in result["search_queries"])
    assert "token_ratio_estimation" not in result


def test_query_refiner_adds_fallback_queries_when_llm_returns_too_few():
    query_response = QueryExpansion(
        reasoning="Only one query returned by model.",
        search_queries=["legal summarization benchmark"],
    )

    with _patch_llm(query_response):
        result = query_refiner_node(_make_state())

    assert len(result["search_queries"]) >= 3
    assert any("benchmark" in query.lower() for query in result["search_queries"])


def test_query_refiner_supports_dict_constraints_from_checkpoint():
    query_response = QueryExpansion(
        reasoning="Derived from query and constraints.",
        search_queries=[
            "long context summarization benchmark",
            "legal text reasoning benchmark",
            "document retrieval qa benchmark",
        ],
    )

    constraints = Constraints(min_context_window=0, modality_input=["text"], modality_output=["text"])
    with _patch_llm(query_response):
        result = query_refiner_node(_make_state(constraints=constraints.model_dump()))

    assert len(result["search_queries"]) == 3


def test_query_refiner_adds_logs():
    query_response = QueryExpansion(
        reasoning="Derived from summarization intent.",
        search_queries=[
            "long context summarization benchmark",
            "legal document QA benchmark",
            "document understanding llm benchmark",
        ],
    )

    with _patch_llm(query_response):
        result = query_refiner_node(_make_state())

    assert "logs" in result
    assert any("Query Refiner" in entry for entry in result["logs"])


def test_query_refiner_deduplicates_and_falls_back():
    # normalize_queries deduplicates to 1 unique query; _ensure_query_count pads to 3
    query_response = QueryExpansion(
        reasoning="Repeated queries.",
        search_queries=["Legal summarization", "legal summarization", "LEGAL SUMMARIZATION"],
    )

    with _patch_llm(query_response):
        result = query_refiner_node(_make_state())

    assert len(result["search_queries"]) >= 3
    assert any("fallback" in entry.lower() for entry in result["logs"])


def test_query_refiner_no_fallback_log_when_enough_queries():
    query_response = QueryExpansion(
        reasoning="Enough unique queries.",
        search_queries=[
            "long context summarization benchmark",
            "legal document QA benchmark",
            "document understanding llm benchmark",
        ],
    )

    with _patch_llm(query_response):
        result = query_refiner_node(_make_state())

    assert not any("fallback" in entry.lower() for entry in result["logs"])


def test_query_refiner_handles_empty_messages():
    # Node should build a HumanMessage from user_query when messages is empty
    query_response = QueryExpansion(
        reasoning="Derived from user query.",
        search_queries=[
            "long context summarization benchmark",
            "legal document QA benchmark",
            "document understanding llm benchmark",
        ],
    )
    state = cast(
        AgentState,
        {
            "user_query": "I need a model for long legal-document summarization",
            "messages": [],
            "constraints": Constraints(min_context_window=0),
        },
    )

    with _patch_llm(query_response):
        result = query_refiner_node(state)

    assert len(result["search_queries"]) >= 3
