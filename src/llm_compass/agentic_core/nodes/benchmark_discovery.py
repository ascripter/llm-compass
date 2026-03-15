"""
Req 2.3 Node 3 (a): Benchmark Discovery

This node executes find_relevant_benchmarks tool using search_queries from Node 2.
Outputs weighted_benchmarks: List[Dict] with id and weight.
"""

from typing import List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.runnables import RunnableConfig
from sqlalchemy.orm import Session

from llm_compass.config import Settings
from llm_compass.data.embedding import get_embedding
from llm_compass.data.models import BenchmarkDictionary
from ..state import AgentState

logger = logging.getLogger(__name__)


def find_relevant_benchmarks(
    queries: List[str], settings: Settings, session: Session, cutoff_score: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Perform vector search against the Benchmark Dictionary for each query.
    Aggregate scores if benchmarks appear in multiple results.
    Return only benchmarks with relevance > cutoff_score.

    Args:
        queries: List of search queries
        cutoff_score: Minimum relevance score to include

    Returns:
        List of dicts: [{"score": 0.9, <BenchmarkDictionary keys/values>}, ...]
    """
    embedding = get_embedding(settings)

    records = session.query(BenchmarkDictionary).all()
    records_dict = {record.id: record for record in records}

    if not records_dict:
        logger.warning("No benchmark records found in database")
        return []

    all_results = []

    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        futures = {
            executor.submit(embedding.search_index, records_dict, q, 5): q for q in queries
        }
        for future in as_completed(futures):
            query = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error searching for query '{query}': {e}")

    # Aggregate scores (max) for benchmarks that appear multiple times
    benchmark_scores = {}
    for result in all_results:
        bench_id = result["id"]
        score = result["score"]
        item = result["item"]
        if bench_id not in benchmark_scores:
            # since item is an sqlalchemy model, we dynamically convert columns to dict
            benchmark_scores[bench_id] = {
                c.name: getattr(item, c.name) for c in item.__table__.columns
            }
            benchmark_scores[bench_id]["score"] = 0.0
        benchmark_scores[bench_id]["score"] = max(
            round(score, 4), benchmark_scores[bench_id]["score"]
        )

    results = [v for v in benchmark_scores.values() if v["score"] > cutoff_score]
    results.sort(key=lambda v: -v["score"])
    logger.debug(
        "Benchmarks found: "
        + " | ".join(
            [f"score={_['score']}: {_['name_normalized']} ({_['variant']})" for _ in results]
        )
    )
    logger.info(f"Found {len(results)} relevant benchmarks from {len(queries)} queries")
    return results


def benchmark_discovery_node(
    state: AgentState, config: RunnableConfig, *, settings: Settings
) -> dict:
    """
    Node 3: Execute find_relevant_benchmarks with search_queries.
    Output: weighted_benchmarks as List[Dict] with id and weight.
    """
    search_queries = state.get("search_queries", [])
    if not search_queries:
        logger.warning("No search_queries found in state")
        return {"weighted_benchmarks": [], "logs": ["No benchmarks found"]}

    session: Session = config["configurable"]["session"]
    try:
        results = find_relevant_benchmarks(search_queries, settings=settings, session=session)
        logger.info(f"Set weighted_benchmarks: {len(results)} items")
        logs = [f"{len(results)} found via similarity search"]
        return {
            "weighted_benchmarks": results,
            "average_benchmark_similarity": sum(_["score"] for _ in results) / len(results),
            "logs": logs,
        }
    except Exception as e:
        logger.error(f"Error in benchmark discovery: {e}")
        return {"weighted_benchmarks": [], "logs": ["No benchmarks found"]}
