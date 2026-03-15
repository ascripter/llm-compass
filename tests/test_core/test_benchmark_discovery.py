"""Unit tests for Req 2.3 Node 3: Benchmark Discovery."""

from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from llm_compass.agentic_core.nodes.benchmark_discovery import (
    benchmark_discovery_node,
    find_relevant_benchmarks,
)
from llm_compass.agentic_core.state import AgentState
from llm_compass.data.models import Base, BenchmarkDictionary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def sample_benchmarks(db_session):
    """Create sample benchmark records for testing."""
    benchmarks = [
        BenchmarkDictionary(
            id=1,
            name_normalized="mmlu",
            variant="",
            description="Massive Multitask Language Understanding benchmark",
            categories=["knowledge", "reasoning"],
        ),
        BenchmarkDictionary(
            id=2,
            name_normalized="gpqa",
            variant="",
            description="Google-Proof Q&A benchmark for reasoning",
            categories=["reasoning"],
        ),
        BenchmarkDictionary(
            id=3,
            name_normalized="humaneval",
            variant="",
            description="Python code generation benchmark",
            categories=["coding"],
        ),
    ]
    for bench in benchmarks:
        db_session.add(bench)
    db_session.commit()
    for bench in benchmarks:
        db_session.refresh(bench)
    return {b.id: b for b in benchmarks}


_SENTINEL = object()


def _make_state(search_queries=_SENTINEL) -> AgentState:
    """Create a test AgentState with optional search_queries.

    Pass search_queries=None to omit the key (simulates missing).
    Pass search_queries=[] for an empty list.
    Default: populates with sample queries.
    """
    state = cast(AgentState, {
        "user_query": "I need a model for code generation",
        "weighted_benchmarks": [],
    })
    if search_queries is _SENTINEL:
        state["search_queries"] = ["code generation benchmark", "programming task benchmark"]
    elif search_queries is not None:
        state["search_queries"] = search_queries
    return state


def _make_mock_settings():
    """Create mock settings for testing."""
    return MagicMock()


def _make_mock_config(session=None):
    """Create a mock RunnableConfig with a configurable session."""
    return {"configurable": {"session": session or MagicMock()}}


# ---------------------------------------------------------------------------
# Tests for find_relevant_benchmarks function
# ---------------------------------------------------------------------------

@patch('llm_compass.agentic_core.nodes.benchmark_discovery.get_embedding')
def test_find_relevant_benchmarks_returns_weighted_results(mock_get_embedding, sample_benchmarks):
    """Test that find_relevant_benchmarks aggregates and returns properly weighted results."""
    mock_session = MagicMock()
    mock_session.query.return_value.all.return_value = list(sample_benchmarks.values())

    mock_embedding = MagicMock()
    mock_embedding.search_index.side_effect = [
        # Results for "code generation benchmark"
        [
            {"id": 3, "score": 0.9, "item": sample_benchmarks[3]},  # HumanEval
            {"id": 1, "score": 0.6, "item": sample_benchmarks[1]},  # MMLU
        ],
        # Results for "programming task benchmark"
        [
            {"id": 3, "score": 0.8, "item": sample_benchmarks[3]},  # HumanEval again
            {"id": 2, "score": 0.5, "item": sample_benchmarks[2]},  # GPQA
        ],
    ]
    mock_get_embedding.return_value = mock_embedding

    queries = ["code generation benchmark", "programming task benchmark"]
    results = find_relevant_benchmarks(queries, _make_mock_settings(), mock_session, cutoff_score=0.7)

    # Aggregation uses max: HumanEval max(0.9, 0.8) = 0.9; others below cutoff
    assert len(results) == 1
    assert results[0]["id"] == 3
    assert results[0]["name_normalized"] == "humaneval"
    assert abs(results[0]["weight"] - 0.9) < 0.01


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.get_embedding')
def test_find_relevant_benchmarks_filters_by_cutoff(mock_get_embedding, sample_benchmarks):
    """Test that results below cutoff score are filtered out."""
    mock_session = MagicMock()
    mock_session.query.return_value.all.return_value = list(sample_benchmarks.values())

    mock_embedding = MagicMock()
    mock_embedding.search_index.return_value = [
        {"id": 1, "score": 0.5, "item": sample_benchmarks[1]},  # Below cutoff
        {"id": 2, "score": 0.6, "item": sample_benchmarks[2]},  # Below cutoff
    ]
    mock_get_embedding.return_value = mock_embedding

    queries = ["test query"]
    results = find_relevant_benchmarks(queries, _make_mock_settings(), mock_session, cutoff_score=0.7)

    assert len(results) == 0


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.get_embedding')
def test_find_relevant_benchmarks_handles_empty_database(mock_get_embedding):
    """Test graceful handling when no benchmarks exist in database."""
    mock_session = MagicMock()
    mock_session.query.return_value.all.return_value = []

    queries = ["test query"]
    results = find_relevant_benchmarks(queries, _make_mock_settings(), mock_session)

    assert results == []


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.get_embedding')
def test_find_relevant_benchmarks_handles_search_errors(mock_get_embedding, sample_benchmarks):
    """Test that search errors for individual queries don't break the whole process."""
    mock_session = MagicMock()
    mock_session.query.return_value.all.return_value = list(sample_benchmarks.values())

    mock_embedding = MagicMock()
    mock_embedding.search_index.side_effect = [
        Exception("Search failed"),  # First query fails
        [{"id": 3, "score": 0.9, "item": sample_benchmarks[3]}],  # Second succeeds
    ]
    mock_get_embedding.return_value = mock_embedding

    queries = ["failing query", "working query"]
    results = find_relevant_benchmarks(queries, _make_mock_settings(), mock_session, cutoff_score=0.7)

    # Should still return results from successful query
    assert len(results) == 1
    assert results[0]["id"] == 3
    assert results[0]["name_normalized"] == "humaneval"


# ---------------------------------------------------------------------------
# Tests for benchmark_discovery_node function
# ---------------------------------------------------------------------------

@patch('llm_compass.agentic_core.nodes.benchmark_discovery.find_relevant_benchmarks')
def test_benchmark_discovery_node_sets_weighted_benchmarks(mock_find_benchmarks):
    """Test that the node sets weighted_benchmarks in state."""
    mock_find_benchmarks.return_value = [
        {"id": 3, "name_normalized": "humaneval", "variant": "", "weight": 0.9},
        {"id": 1, "name_normalized": "mmlu", "variant": "", "weight": 0.8},
    ]

    state = _make_state(["code generation benchmark"])
    config = _make_mock_config()
    result = benchmark_discovery_node(state, config, settings=_make_mock_settings())

    assert "weighted_benchmarks" in result
    assert len(result["weighted_benchmarks"]) == 2
    assert result["weighted_benchmarks"][0]["name_normalized"] == "humaneval"
    assert result["weighted_benchmarks"][0]["weight"] == 0.9
    assert result["weighted_benchmarks"][1]["name_normalized"] == "mmlu"
    assert result["weighted_benchmarks"][1]["weight"] == 0.8


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.find_relevant_benchmarks')
def test_benchmark_discovery_node_handles_missing_search_queries(mock_find_benchmarks):
    """Test that node handles missing search_queries gracefully."""
    state = _make_state(search_queries=None)  # no search_queries key in state
    config = _make_mock_config()
    result = benchmark_discovery_node(state, config, settings=_make_mock_settings())

    mock_find_benchmarks.assert_not_called()
    assert result["weighted_benchmarks"] == []


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.find_relevant_benchmarks')
def test_benchmark_discovery_node_handles_empty_search_queries(mock_find_benchmarks):
    """Test that node handles empty search_queries list gracefully."""
    state = _make_state(search_queries=[])
    config = _make_mock_config()
    result = benchmark_discovery_node(state, config, settings=_make_mock_settings())

    mock_find_benchmarks.assert_not_called()
    assert result["weighted_benchmarks"] == []


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.find_relevant_benchmarks')
def test_benchmark_discovery_node_handles_exceptions(mock_find_benchmarks):
    """Test that exceptions in find_relevant_benchmarks are caught and handled."""
    mock_find_benchmarks.side_effect = Exception("Database error")

    state = _make_state(["test query"])
    config = _make_mock_config()
    result = benchmark_discovery_node(state, config, settings=_make_mock_settings())

    assert result["weighted_benchmarks"] == []


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.find_relevant_benchmarks')
def test_benchmark_discovery_node_returns_only_its_state_updates(mock_find_benchmarks):
    """Node returns only the keys it owns; LangGraph merges them into the full state."""
    result_benchmarks = [
        {"id": 3, "name_normalized": "humaneval", "variant": "", "weight": 0.9}
    ]
    mock_find_benchmarks.return_value = result_benchmarks

    state = _make_state(["code benchmark"])
    config = _make_mock_config()
    result = benchmark_discovery_node(state, config, settings=_make_mock_settings())

    # Only the keys the node owns should be present in the returned dict
    assert set(result.keys()) == {"weighted_benchmarks", "average_benchmark_similarity", "logs"}
    assert result["weighted_benchmarks"] == result_benchmarks
