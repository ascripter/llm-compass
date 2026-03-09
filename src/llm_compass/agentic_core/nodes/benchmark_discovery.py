"""
Req 2.3 Node 3: Benchmark Discovery

This node executes find_relevant_benchmarks tool using search_queries from Node 2.
Outputs weighted_benchmarks: List[Dict] with id and weight.
"""

from typing import List, Dict, Any
import logging

from sqlalchemy.orm import Session

from llm_compass.config import Settings
from llm_compass.data.embedding import Embedding
from llm_compass.data.models import BenchmarkDictionary
from ..state import AgentState

logger = logging.getLogger(__name__)


def find_relevant_benchmarks(
    queries: List[str], settings: Settings, session: Session, cutoff_score: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Perform vector search against the Benchmark Dictionary for each query.
    Aggregate scores if benchmarks appear in multiple results.
    Return only benchmarks with relevance > cutoff_score.

    Args:
        queries: List of search queries
        cutoff_score: Minimum relevance score to include

    Returns:
        List of dicts: [{"id": "mmlu", "name": "MMLU", "relevance_weight": 0.9}, ...]
    """
    embedding = Embedding(settings)

    # Get all benchmark records
    from llm_compass.data.database import Database

    db = Database(settings)
    with session:
        records = session.query(BenchmarkDictionary).all()
        records_dict = {record.id: record for record in records}

    if not records_dict:
        logger.warning("No benchmark records found in database")
        return []

    all_results = []
    for query in queries:
        try:
            results = embedding.search_index(records_dict, query, top_k=10)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Error searching for query '{query}': {e}")
            continue

    # Aggregate scores for benchmarks that appear multiple times
    benchmark_scores = {}
    for result in all_results:
        bench_id = result["id"]
        score = result["score"]
        if bench_id not in benchmark_scores:
            benchmark_scores[bench_id] = {"total_score": 0.0, "count": 0, "item": result["item"]}
        benchmark_scores[bench_id]["total_score"] += score
        benchmark_scores[bench_id]["count"] += 1

    # Calculate average relevance weight and filter
    weighted_benchmarks = []
    for bench_id, data in benchmark_scores.items():
        avg_score = data["total_score"] / data["count"]
        if avg_score > cutoff_score:
            weighted_benchmarks.append(
                {
                    "id": data["item"].normalized_name,
                    "name": data["item"].name,
                    "relevance_weight": round(avg_score, 3),
                }
            )

    # Sort by relevance_weight descending
    weighted_benchmarks.sort(key=lambda x: x["relevance_weight"], reverse=True)

    logger.info(
        f"Found {len(weighted_benchmarks)} relevant benchmarks from {len(queries)} queries"
    )
    return weighted_benchmarks


def benchmark_discovery_node(
    state: AgentState, *, settings: Settings, session: Session
) -> AgentState:
    """
    Node 3: Execute find_relevant_benchmarks with search_queries.
    Output: weighted_benchmarks as List[Dict] with id and weight.
    """
    search_queries = state.get("search_queries", [])
    if not search_queries:
        logger.warning("No search_queries found in state")
        state["weighted_benchmarks"] = []
        return state

    try:
        results = find_relevant_benchmarks(search_queries, settings=settings, session=session)
        # Transform to the expected output format: [{"id": "...", "weight": 0.9}, ...]
        weighted_benchmarks = [
            {"id": item["id"], "weight": item["relevance_weight"]} for item in results
        ]
        state["weighted_benchmarks"] = weighted_benchmarks
        logger.info(f"Set weighted_benchmarks: {len(weighted_benchmarks)} items")
    except Exception as e:
        logger.error(f"Error in benchmark discovery: {e}")
        state["weighted_benchmarks"] = []

    return state
