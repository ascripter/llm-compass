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
            name="MMLU",
            normalized_name="mmlu",
            description="Massive Multitask Language Understanding benchmark"
        ),
        BenchmarkDictionary(
            id=2,
            name="GPQA",
            normalized_name="gpqa",
            description="Google-Proof Q&A benchmark for reasoning"
        ),
        BenchmarkDictionary(
            id=3,
            name="HumanEval",
            normalized_name="humaneval",
            description="Python code generation benchmark"
        ),
    ]
    for bench in benchmarks:
        db_session.add(bench)
    db_session.commit()
    return {b.id: b for b in benchmarks}


def _make_state(search_queries: list[str] | None = None) -> AgentState:
    """Create a test AgentState with optional search_queries."""
    state = cast(AgentState, {
        "user_query": "I need a model for code generation",
        "search_queries": search_queries or ["code generation benchmark", "programming task benchmark"],
        "weighted_benchmarks": [],
    })
    return state


def _make_mock_settings():
    """Create mock settings for testing."""
    mock_settings = MagicMock()
    return mock_settings


# ---------------------------------------------------------------------------
# Tests for find_relevant_benchmarks function
# ---------------------------------------------------------------------------

@patch('llm_compass.agentic_core.nodes.benchmark_discovery.Database')
@patch('llm_compass.agentic_core.nodes.benchmark_discovery.Embedding')
def test_find_relevant_benchmarks_returns_weighted_results(mock_embedding_class, mock_db_class, sample_benchmarks):
    """Test that find_relevant_benchmarks aggregates and returns properly weighted results."""
    # Mock database session
    mock_session = MagicMock()
    mock_session.query.return_value.all.return_value = list(sample_benchmarks.values())
    mock_db_instance = MagicMock()
    mock_db_instance.get_session.return_value.__enter__.return_value = mock_session
    mock_db_class.return_value = mock_db_instance

    # Mock embedding search results
    mock_embedding_instance = MagicMock()
    mock_embedding_instance.search_index.side_effect = [
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
    mock_embedding_class.return_value = mock_embedding_instance

    queries = ["code generation benchmark", "programming task benchmark"]
    results = find_relevant_benchmarks(queries, cutoff_score=0.7)

    # Should return HumanEval with averaged score: (0.9 + 0.8) / 2 = 0.85
    assert len(results) == 1
    assert results[0]["id"] == "humaneval"
    assert results[0]["name"] == "HumanEval"
    assert abs(results[0]["relevance_weight"] - 0.85) < 0.01


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.Database')
@patch('llm_compass.agentic_core.nodes.benchmark_discovery.Embedding')
def test_find_relevant_benchmarks_filters_by_cutoff(mock_embedding_class, mock_db_class, sample_benchmarks):
    """Test that results below cutoff score are filtered out."""
    # Mock database session
    mock_session = MagicMock()
    mock_session.query.return_value.all.return_value = list(sample_benchmarks.values())
    mock_db_instance = MagicMock()
    mock_db_instance.get_session.return_value.__enter__.return_value = mock_session
    mock_db_class.return_value = mock_db_instance

    # Mock embedding search results with low scores
    mock_embedding_instance = MagicMock()
    mock_embedding_instance.search_index.return_value = [
        {"id": 1, "score": 0.5, "item": sample_benchmarks[1]},  # Below cutoff
        {"id": 2, "score": 0.6, "item": sample_benchmarks[2]},  # Below cutoff
    ]
    mock_embedding_class.return_value = mock_embedding_instance

    queries = ["test query"]
    results = find_relevant_benchmarks(queries, cutoff_score=0.7)

    assert len(results) == 0


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.Database')
@patch('llm_compass.agentic_core.nodes.benchmark_discovery.Embedding')
def test_find_relevant_benchmarks_handles_empty_database(mock_embedding_class, mock_db_class):
    """Test graceful handling when no benchmarks exist in database."""
    # Mock empty database
    mock_session = MagicMock()
    mock_session.query.return_value.all.return_value = []
    mock_db_instance = MagicMock()
    mock_db_instance.get_session.return_value.__enter__.return_value = mock_session
    mock_db_class.return_value = mock_db_instance

    mock_embedding_class.return_value = MagicMock()

    queries = ["test query"]
    results = find_relevant_benchmarks(queries)

    assert results == []


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.Database')
@patch('llm_compass.agentic_core.nodes.benchmark_discovery.Embedding')
def test_find_relevant_benchmarks_handles_search_errors(mock_embedding_class, mock_db_class, sample_benchmarks):
    """Test that search errors for individual queries don't break the whole process."""
    # Mock database session
    mock_session = MagicMock()
    mock_session.query.return_value.all.return_value = list(sample_benchmarks.values())
    mock_db_instance = MagicMock()
    mock_db_instance.get_session.return_value.__enter__.return_value = mock_session
    mock_db_class.return_value = mock_db_instance

    # Mock embedding that raises exception for first query, succeeds for second
    mock_embedding_instance = MagicMock()
    mock_embedding_instance.search_index.side_effect = [
        Exception("Search failed"),  # First query fails
        [{"id": 3, "score": 0.9, "item": sample_benchmarks[3]}],  # Second succeeds
    ]
    mock_embedding_class.return_value = mock_embedding_instance

    queries = ["failing query", "working query"]
    results = find_relevant_benchmarks(queries, cutoff_score=0.7)

    # Should still return results from successful query
    assert len(results) == 1
    assert results[0]["id"] == "humaneval"


# ---------------------------------------------------------------------------
# Tests for benchmark_discovery_node function
# ---------------------------------------------------------------------------

@patch('llm_compass.agentic_core.nodes.benchmark_discovery.find_relevant_benchmarks')
def test_benchmark_discovery_node_sets_weighted_benchmarks(mock_find_benchmarks):
    """Test that the node sets weighted_benchmarks in state."""
    mock_find_benchmarks.return_value = [
        {"id": "humaneval", "name": "HumanEval", "relevance_weight": 0.9},
        {"id": "mmlu", "name": "MMLU", "relevance_weight": 0.8},
    ]

    state = _make_state(["code generation benchmark"])
    result = benchmark_discovery_node(state, _make_mock_settings())

    assert "weighted_benchmarks" in result
    assert len(result["weighted_benchmarks"]) == 2
    assert result["weighted_benchmarks"][0] == {"id": "humaneval", "weight": 0.9}
    assert result["weighted_benchmarks"][1] == {"id": "mmlu", "weight": 0.8}


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.find_relevant_benchmarks')
def test_benchmark_discovery_node_handles_missing_search_queries(mock_find_benchmarks):
    """Test that node handles missing search_queries gracefully."""
    state = _make_state(search_queries=None)
    result = benchmark_discovery_node(state, _make_mock_settings())

    # Should not call find_relevant_benchmarks
    mock_find_benchmarks.assert_not_called()
    assert result["weighted_benchmarks"] == []


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.find_relevant_benchmarks')
def test_benchmark_discovery_node_handles_empty_search_queries(mock_find_benchmarks):
    """Test that node handles empty search_queries list gracefully."""
    state = _make_state(search_queries=[])
    result = benchmark_discovery_node(state, _make_mock_settings())

    # Should not call find_relevant_benchmarks
    mock_find_benchmarks.assert_not_called()
    assert result["weighted_benchmarks"] == []


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.find_relevant_benchmarks')
def test_benchmark_discovery_node_handles_exceptions(mock_find_benchmarks):
    """Test that exceptions in find_relevant_benchmarks are caught and handled."""
    mock_find_benchmarks.side_effect = Exception("Database error")

    state = _make_state(["test query"])
    result = benchmark_discovery_node(state, _make_mock_settings())

    assert result["weighted_benchmarks"] == []


@patch('llm_compass.agentic_core.nodes.benchmark_discovery.find_relevant_benchmarks')
def test_benchmark_discovery_node_preserves_other_state(mock_find_benchmarks):
    """Test that the node preserves other state fields."""
    mock_find_benchmarks.return_value = [
        {"id": "humaneval", "name": "HumanEval", "relevance_weight": 0.9}
    ]

    state = _make_state(["code benchmark"])
    state["user_query"] = "original query"
    state["some_other_field"] = "preserved"

    result = benchmark_discovery_node(state, _make_mock_settings())

    assert result["user_query"] == "original query"
    assert result["some_other_field"] == "preserved"
    assert result["search_queries"] == ["code benchmark"]
    assert result["weighted_benchmarks"] == [{"id": "humaneval", "weight": 0.9}]