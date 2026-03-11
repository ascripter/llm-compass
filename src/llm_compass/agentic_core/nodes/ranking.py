"""
Req 2.3 Node 4: Scoring and Ranking

This node runs after benchmark discovery.
It finds LLMs for relevant benchmarks, scores and ranks them.
"""

from typing import List, Dict, Any, Optional, Tuple
import copy
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from sqlalchemy.orm import Session

from llm_compass.config import Settings
from llm_compass.common.schemas import Constraints
from llm_compass.data.models import LLMMetadata, BenchmarkScore, BenchmarkDictionary
from ..schemas import IntentExtraction, QueryExpansion
from ..state import AgentState
from sqlalchemy import and_, or_, func
import numpy as np


logger = logging.getLogger(__name__)


def _calculate_blended_cost(
    model: LLMMetadata, token_ratio_estimation: Dict[str, Dict[str, float]]
) -> float:
    """
    Calculate blended cost based on I/O ratios and model pricing.

    Formula: sum(token_ratio_estimation[mode][modality] * cost_from_llm_metadata)
    """
    input_ratios = token_ratio_estimation.get("normalized_input_ratios", {})
    output_ratios = token_ratio_estimation.get("normalized_output_ratios", {})

    blended_cost = 0.0

    # Input costs
    blended_cost += input_ratios.get("text", 0) * (model.cost_input_text_1m or 0)
    blended_cost += input_ratios.get("image", 0) * (model.cost_input_image_1024 or 0)
    blended_cost += input_ratios.get("audio", 0) * (model.cost_input_audio_1h or 0)
    blended_cost += input_ratios.get("video", 0) * (model.cost_input_video_1s or 0)

    # Output costs
    blended_cost += output_ratios.get("text", 0) * (model.cost_output_text_1m or 0)
    blended_cost += output_ratios.get("image", 0) * (model.cost_output_image_1024 or 0)
    blended_cost += output_ratios.get("audio", 0) * (model.cost_output_audio_1h or 0)
    blended_cost += output_ratios.get("video", 0) * (model.cost_output_video_1s or 0)

    return blended_cost


def _normalize_scores_to_0_1(scores: List[float]) -> List[float]:
    """Normalize scores to [0,1] range where higher is better."""
    if not scores:
        return []

    min_score, max_score = min(scores), max(scores)
    if max_score == min_score:
        return [0.5] * len(scores)  # All equal, return middle value

    return [(score - min_score) / (max_score - min_score) for score in scores]


def _precompute_bridge_calibration(
    session: Session,
    benchmark_ids: List[int],
) -> Tuple[Dict[int, Optional[Tuple[float, str]]], Dict[int, "BenchmarkDictionary"]]:
    """
    Pre-compute bridge model offsets once per benchmark_id.

    If models lack data for Variant A but have data for Variant B,
    use "Bridge Models" (models with both A and B scores) to calculate a median offset.

    Returns:
        - {benchmark_id: (median_offset, bridge_model_name)} or {benchmark_id: None}
        - {benchmark_id: BenchmarkDictionary ORM object}
    """
    benchmark_orm = {
        b.id: b
        for b in session.query(BenchmarkDictionary)
        .filter(BenchmarkDictionary.id.in_(benchmark_ids))
        .all()
    }
    benchmark_names = list({b.name_normalized for b in benchmark_orm.values()})

    # Find all models that have scores for ≥2 distinct variants of each benchmark name.
    bridge_model_rows = (
        session.query(BenchmarkScore.model_id, BenchmarkDictionary.name_normalized)
        .join(BenchmarkDictionary)
        .filter(BenchmarkDictionary.name_normalized.in_(benchmark_names))
        .group_by(BenchmarkScore.model_id, BenchmarkDictionary.name_normalized)
        .having(func.count(func.distinct(BenchmarkDictionary.variant)) >= 2)
        .all()
    )

    if not bridge_model_rows:
        return {bid: None for bid in benchmark_ids}, benchmark_orm

    bridge_ids_by_name: Dict[str, List[int]] = {}
    all_bridge_model_ids: set = set()
    for model_id, name_normalized in bridge_model_rows:
        bridge_ids_by_name.setdefault(name_normalized, []).append(model_id)
        all_bridge_model_ids.add(model_id)

    # Bulk fetch scores for all bridge models × benchmark names.
    bridge_score_rows = (
        session.query(
            BenchmarkScore.model_id,
            BenchmarkScore.score_value,
            BenchmarkDictionary.variant,
            BenchmarkDictionary.name_normalized,
        )
        .join(BenchmarkDictionary)
        .filter(
            and_(
                BenchmarkScore.model_id.in_(all_bridge_model_ids),
                BenchmarkDictionary.name_normalized.in_(benchmark_names),
            )
        )
        .all()
    )

    # Group as (name_normalized, model_id) → {variant: score_value}
    bridge_scores_by_name_model: Dict[Tuple[str, int], Dict[str, float]] = {}
    for model_id, score_value, variant, name_normalized in bridge_score_rows:
        bridge_scores_by_name_model.setdefault((name_normalized, model_id), {})[
            variant
        ] = score_value

    # Fetch bridge model names (for estimation_note).
    bridge_model_names = {
        model_id: name
        for model_id, name in session.query(LLMMetadata.id, LLMMetadata.name_normalized)
        .filter(LLMMetadata.id.in_(all_bridge_model_ids))
        .all()
    }

    # Compute median offset per benchmark_id.
    result: Dict[int, Optional[Tuple[float, str]]] = {}
    for bid in benchmark_ids:
        bm = benchmark_orm[bid]
        bridge_ids_for_name = bridge_ids_by_name.get(bm.name_normalized, [])

        offsets = []
        for bridge_model_id in bridge_ids_for_name:
            scores = bridge_scores_by_name_model.get((bm.name_normalized, bridge_model_id), {})
            if len(scores) == 2 and bm.variant in scores:
                other_variant = next(v for v in scores if v != bm.variant)
                offsets.append(scores[bm.variant] - scores[other_variant])

        if not offsets:
            result[bid] = None
        else:
            first_bridge_name = bridge_model_names.get(bridge_ids_for_name[0], "unknown")
            result[bid] = (float(np.median(offsets)), first_bridge_name)

    return result, benchmark_orm


def retrieve_and_rank_models(
    benchmark_weights: List[Dict[str, Any]],
    constraints: dict,
    token_ratio_estimation: dict,
    session: Session,
) -> dict:
    """
    Req 2.2.B: The heavy lifting.
    1. Filter models by constraints (SQL).
    2. Fetch scores.
    3. Apply 'Bridge Model' offset logic (Req 1.3.B / 2.2.B Step 3).
    4. Calculate Blended Cost.
    5. Generate 3 lists: Top Perf, Balanced, Budget.
    """

    # Step 1: Model Filtering
    query = session.query(LLMMetadata).filter(LLMMetadata.is_outdated == False)

    # Apply hard constraints
    if "min_context_window" in constraints and constraints["min_context_window"] > 0:
        query = query.filter(LLMMetadata.context_window >= constraints["min_context_window"])

    if "modality_input" in constraints and constraints["modality_input"]:
        # Filter models that support ALL required input modalities
        for modality in constraints["modality_input"]:
            query = query.filter(LLMMetadata.modality_input.contains(f'"{modality}"'))

    if "modality_output" in constraints and constraints["modality_output"]:
        # Filter models that support ALL required output modalities
        for modality in constraints["modality_output"]:
            query = query.filter(LLMMetadata.modality_output.contains(f'"{modality}"'))

    if "deployment" in constraints:
        if constraints["deployment"] == "local":
            query = query.filter(LLMMetadata.is_open_weights == True)
        elif constraints["deployment"] == "cloud":
            query = query.filter(LLMMetadata.is_open_weights == False)

    if "reasoning_model" in constraints and constraints["reasoning_model"]:
        query = query.filter(LLMMetadata.reasoning_type != "none")

    if "tool_calling" in constraints and constraints["tool_calling"]:
        query = query.filter(LLMMetadata.tool_calling != "none")

    if "min_speed_class" in constraints and constraints["min_speed_class"] == "medium":
        query = query.filter(
            or_(LLMMetadata.speed_class == "medium", LLMMetadata.speed_class == "fast")
        )
    elif "min_speed_class" in constraints and constraints["min_speed_class"] == "fast":
        query = query.filter(LLMMetadata.speed_class == "fast")

    filtered_models = query.all()
    logger.debug(
        "retrieve_and_rank_models | %d models pass constraint filter", len(filtered_models)
    )

    if not filtered_models:
        logger.warning("No models passed constraint filter | constraints=%s", constraints)
        return {
            "top_performance": [],
            "balanced": [],
            "budget": [],
            "metadata": {
                "applied_io_ratio": token_ratio_estimation,
                "benchmarks_used": [],
                "benchmark_weights": [],
            },
        }

    # Step 2: Score Retrieval
    benchmark_ids = [bw["id"] for bw in benchmark_weights]
    model_results = []

    model_ids = [m.id for m in filtered_models]

    # Bulk fetch all scores for relevant (model, benchmark) pairs, most recent first.
    all_score_rows = (
        session.query(BenchmarkScore, BenchmarkDictionary)
        .join(BenchmarkDictionary)
        .filter(
            and_(
                BenchmarkScore.model_id.in_(model_ids),
                BenchmarkScore.benchmark_id.in_(benchmark_ids),
            )
        )
        .order_by(BenchmarkScore.date_published.desc().nullslast())
        .all()
    )
    # Keep only the most recent score per (model_id, benchmark_id).
    # TODO: Alternative heuristic for most reliable score
    scores_by_model_benchmark: Dict[Tuple[int, int], Tuple] = {}
    for score, benchmark in all_score_rows:
        key = (score.model_id, score.benchmark_id)
        if key not in scores_by_model_benchmark:
            scores_by_model_benchmark[key] = (score, benchmark)

    # Pre-compute bridge model calibration data once per benchmark (not once per model).
    # Also returns benchmark_orm: {id: BenchmarkDictionary} fetched from DB.
    bridge_calibration, benchmark_orm = _precompute_bridge_calibration(session, benchmark_ids)

    # Bulk fetch "other variant" scores for all models × benchmark names
    # (variants NOT matching the target benchmark_id but sharing the same name_normalized).
    benchmark_names = [b.name_normalized for b in benchmark_orm.values()]
    other_variant_rows = (
        session.query(
            BenchmarkScore.model_id,
            BenchmarkScore.score_value,
            BenchmarkDictionary.name_normalized,
        )
        .join(BenchmarkDictionary)
        .filter(
            and_(
                BenchmarkScore.model_id.in_(model_ids),
                BenchmarkDictionary.name_normalized.in_(benchmark_names),
                BenchmarkDictionary.id.notin_(benchmark_ids),
            )
        )
        .all()
    )
    # Keep max score per (model_id, name_normalized) — matches original semantics.
    other_variant_scores: Dict[Tuple[int, str], float] = {}
    for model_id, score_value, name_normalized in other_variant_rows:
        key = (model_id, name_normalized)
        if key not in other_variant_scores or score_value > other_variant_scores[key]:
            other_variant_scores[key] = score_value

    for model in filtered_models:
        benchmark_results = []
        performance_scores = []

        for bw in benchmark_weights:
            benchmark_id = bw["id"]
            weight = bw["weight"]

            # Dict lookup — replaces per-(model, benchmark) DB query
            score_record = scores_by_model_benchmark.get((model.id, benchmark_id))

            if score_record:
                score, benchmark = score_record
                benchmark_results.append(
                    {
                        "benchmark_id": benchmark.id,
                        "benchmark_name": benchmark.name_normalized,
                        "benchmark_variant": benchmark.variant,
                        "score": score.score_value,
                        "metric_unit": score.metric_unit,
                        "weight_used": weight,
                        "is_estimated": False,
                        "source_url": score.source_url,
                    }
                )
                performance_scores.append(score.score_value * weight)
            else:
                # Try bridge model calibration
                calib = bridge_calibration.get(benchmark_id)
                if calib is not None:
                    median_offset, bridge_model_name = calib
                    bm = benchmark_orm[benchmark_id]
                    other_score = other_variant_scores.get((model.id, bm.name_normalized))
                    if other_score is not None:
                        estimated_score = other_score - median_offset
                        benchmark_results.append(
                            {
                                "benchmark_id": benchmark_id,
                                "benchmark_name": bm.name_normalized,
                                "benchmark_variant": bm.variant,
                                "score": estimated_score,
                                "metric_unit": "%",  # Default assumption
                                "weight_used": weight,
                                "is_estimated": True,
                                "estimation_note": f"Inferred via bridge model '{bridge_model_name}'",
                            }
                        )
                        performance_scores.append(estimated_score * weight)

        if benchmark_results:  # Only include models with at least some benchmark data
            # Step 4: Calculate Blended Cost
            logger.debug(f"include model: {model.name_normalized}")
            blended_cost = _calculate_blended_cost(model, token_ratio_estimation)

            model_results.append(
                {
                    "model_id": model.id,
                    "name_normalized": model.name_normalized,
                    "speed_class": model.speed_class,
                    "speed_tps": model.speed_tps,
                    "benchmark_results": benchmark_results,
                    "blended_cost_1m_usd": blended_cost,
                    "raw_performance_score": sum(performance_scores),
                }
            )

    if not model_results:
        logger.warning(
            "No models had benchmark data after scoring | filtered=%d | benchmarks=%d",
            len(filtered_models),
            len(benchmark_weights),
        )
        return {
            "top_performance": [],
            "balanced": [],
            "budget": [],
            "metadata": {
                "applied_io_ratio": token_ratio_estimation,
                "benchmarks_used": [bw.get("name") for bw in benchmark_weights],
                "benchmark_weights": [bw.get("weight") for bw in benchmark_weights],
            },
        }

    # Calculate normalized indices
    performance_scores = [mr["raw_performance_score"] for mr in model_results]
    blended_costs = [mr["blended_cost_1m_usd"] for mr in model_results]

    # Normalize performance scores (higher is better)
    normalized_performance = _normalize_scores_to_0_1(performance_scores)

    # Normalize blended costs (lower is better, so invert)
    if max(blended_costs) == min(blended_costs):
        normalized_costs = [0.5] * len(blended_costs)
    else:
        normalized_costs = [
            (max(blended_costs) - cost) / (max(blended_costs) - min(blended_costs))
            for cost in blended_costs
        ]

    # Attach normalized indices to model results
    for i, mr in enumerate(model_results):
        mr["performance_index"] = normalized_performance[i]
        mr["blended_cost_index"] = normalized_costs[i]

    # Step 5: Generate ranking lists
    # Each list gets its own deep copy so metadata annotation and field removal
    # on one list do not corrupt the dicts shared with the other lists.
    # Performance List: Ranked by Performance_Index
    top_performance = sorted(
        copy.deepcopy(model_results), key=lambda x: x["performance_index"], reverse=True
    )

    # Budget List: Ranked by 0.2 * Performance_Index + 0.8 * Blended_Cost_Index
    budget = sorted(
        copy.deepcopy(model_results),
        key=lambda x: 0.2 * x["performance_index"] + 0.8 * x["blended_cost_index"],
        reverse=True,
    )

    # Balanced List: Ranked by 0.5 * Performance_Index + 0.5 * Blended_Cost_Index
    balanced = sorted(
        copy.deepcopy(model_results),
        key=lambda x: 0.5 * x["performance_index"] + 0.5 * x["blended_cost_index"],
        reverse=True,
    )

    # Add ranking metadata and clean up
    for rank_list in [top_performance, balanced, budget]:
        for i, mr in enumerate(rank_list):
            if rank_list == top_performance:
                mr["rank_metrics"] = {
                    "performance_index": mr["performance_index"],
                    "blended_cost_index": mr["blended_cost_index"],
                    "blended_score": mr[
                        "performance_index"
                    ],  # Performance list uses performance score
                }
                mr["reason_for_ranking"] = f"Performance Index: {mr['performance_index']:.3f}"
            elif rank_list == budget:
                blended_score = 0.2 * mr["performance_index"] + 0.8 * mr["blended_cost_index"]
                mr["rank_metrics"] = {
                    "performance_index": mr["performance_index"],
                    "blended_cost_index": mr["blended_cost_index"],
                    "blended_score": blended_score,
                }
                mr["reason_for_ranking"] = f"Budget-optimized score: {blended_score:.3f}"
            else:  # balanced
                blended_score = 0.5 * mr["performance_index"] + 0.5 * mr["blended_cost_index"]
                mr["rank_metrics"] = {
                    "performance_index": mr["performance_index"],
                    "blended_cost_index": mr["blended_cost_index"],
                    "blended_score": blended_score,
                }
                mr["reason_for_ranking"] = f"Balanced score: {blended_score:.3f}"

            # Remove internal calculation fields
            mr.pop("raw_performance_score", None)
            mr.pop("blended_cost_1m_usd", None)
            mr.pop("performance_index", None)
            mr.pop("blended_cost_index", None)

    return {
        "top_performance": top_performance,
        "balanced": balanced,
        "budget": budget,
        "metadata": {
            "applied_io_ratio": token_ratio_estimation,
            "benchmarks_used": [bw.get("name", "") for bw in benchmark_weights],
            "benchmark_weights": [bw.get("weight", 0) for bw in benchmark_weights],
        },
    }


def execute_ranking(state: AgentState, config: RunnableConfig) -> dict:
    """
    Wrapper for retrieve_and_rank_models tool.
    Receives a database session via LangGraph config (same pattern as benchmark_discovery_node).
    """
    session: Session = config["configurable"]["session"]
    constraints_val = state.get("constraints")
    constraints: Dict[str, Any] = constraints_val.model_dump() if isinstance(constraints_val, Constraints) else (constraints_val or {})  # type: ignore[assignment]

    token_ratio_val = state.get("token_ratio_estimation")
    if token_ratio_val is None:
        token_ratio_estimation: Dict[str, Any] = {}
    elif isinstance(token_ratio_val, dict):
        token_ratio_estimation = token_ratio_val
    else:
        token_ratio_estimation = token_ratio_val.model_dump()

    logger.debug(
        "execute_ranking ENTRY | benchmark_weights=%d | constraints=%s | token_ratio_keys=%s",
        len(state.get("weighted_benchmarks", [])),
        constraints,
        list(token_ratio_estimation.keys()),
    )

    ranked_results = retrieve_and_rank_models(
        benchmark_weights=state.get("weighted_benchmarks", []),
        constraints=constraints,
        token_ratio_estimation=token_ratio_estimation,
        session=session,
    )

    top_n = len(ranked_results.get("top_performance", []))
    bal_n = len(ranked_results.get("balanced", []))
    bud_n = len(ranked_results.get("budget", []))

    logger.debug(
        "execute_ranking EXIT | top_performance=%d | balanced=%d | budget=%d | benchmarks_used=%d",
        top_n,
        bal_n,
        bud_n,
        len(ranked_results.get("metadata", {}).get("benchmarks_used", [])),
    )

    logs = [f"Ranking: ranked {top_n} models (top_performance / balanced / budget lists)."]
    return {"ranked_results": ranked_results, "logs": logs}
