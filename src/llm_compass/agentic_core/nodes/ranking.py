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
    model: LLMMetadata, 
    token_ratio_estimation: Dict[str, Dict[str, float]]
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


def _apply_bridge_model_calibration(
    session: Session,
    model_id: int,
    benchmark_id: int,
    variant: str
) -> Tuple[Optional[float], Optional[str]]:
    """
    Apply bridge model offset logic for score estimation.
    
    If Model X lacks data for Variant A but has data for Variant B,
    use "Bridge Models" (models with both A and B scores) to calculate offset.
    """
    # Get all models that have scores for both variants of the same benchmark
    bridge_models = (
        session.query(BenchmarkScore.model_id)
        .join(BenchmarkDictionary)
        .filter(
            BenchmarkDictionary.name_normalized == (
                session.query(BenchmarkDictionary.name_normalized)
                .filter(BenchmarkDictionary.id == benchmark_id)
                .scalar()
            )
        )
        .group_by(BenchmarkScore.model_id)
        .having(
            func.count(func.distinct(BenchmarkDictionary.variant)) >= 2
        )
        .all()
    )
    
    if not bridge_models:
        return None, None
    
    bridge_model_ids = [model_id for (model_id,) in bridge_models]
    
    # Calculate average offset between variants
    offsets = []
    for bridge_model_id in bridge_model_ids:
        # Get scores for both variants of the same benchmark for this bridge model
        bridge_scores = (
            session.query(
                BenchmarkScore.score_value,
                BenchmarkDictionary.variant
            )
            .join(BenchmarkDictionary)
            .filter(
                and_(
                    BenchmarkScore.model_id == bridge_model_id,
                    BenchmarkDictionary.name_normalized == (
                        session.query(BenchmarkDictionary.name_normalized)
                        .filter(BenchmarkDictionary.id == benchmark_id)
                        .scalar()
                    )
                )
            )
            .all()
        )
        
        if len(bridge_scores) == 2:
            score_dict = {variant: score for score, variant in bridge_scores}
            variants = list(score_dict.keys())
            if variant in variants:
                # Find the other variant
                other_variant = [v for v in variants if v != variant][0]
                offset = score_dict[variant] - score_dict[other_variant]
                offsets.append(offset)
    
    if not offsets:
        return None, None
    
    # Use median offset for robustness
    median_offset = np.median(offsets)
    
    # Get the model's score for the other variant
    other_variant_scores = (
        session.query(BenchmarkScore.score_value)
        .join(BenchmarkDictionary)
        .filter(
            and_(
                BenchmarkScore.model_id == model_id,
                BenchmarkDictionary.name_normalized == (
                    session.query(BenchmarkDictionary.name_normalized)
                    .filter(BenchmarkDictionary.id == benchmark_id)
                    .scalar()
                ),
                BenchmarkDictionary.variant != variant
            )
        )
        .all()
    )
    
    if not other_variant_scores:
        return None, None
    
    # Take the most recent score for the other variant
    other_score = max(other_variant_scores, key=lambda x: x[0])[0]
    estimated_score = other_score - median_offset
    
    bridge_model_name = session.query(LLMMetadata.name_normalized).filter(
        LLMMetadata.id == bridge_model_ids[0]
    ).scalar()
    
    return estimated_score, f"Inferred via bridge model '{bridge_model_name}'"


def retrieve_and_rank_models(
    benchmark_weights: List[Dict[str, Any]], 
    constraints: dict, 
    token_ratio_estimation: dict, 
    session: Session
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
    if "min_context_window" in constraints:
        query = query.filter(LLMMetadata.context_window >= constraints["min_context_window"])
    
    if "modality_input" in constraints and constraints["modality_input"]:
        # Filter models that support ALL required input modalities
        for modality in constraints["modality_input"]:
            query = query.filter(LLMMetadata.modality_input.contains([modality]))
    
    if "modality_output" in constraints and constraints["modality_output"]:
        # Filter models that support ALL required output modalities  
        for modality in constraints["modality_output"]:
            query = query.filter(LLMMetadata.modality_output.contains([modality]))
    
    if "deployment" in constraints:
        if constraints["deployment"] == "local":
            query = query.filter(LLMMetadata.is_open_weights == True)
        elif constraints["deployment"] == "cloud":
            query = query.filter(LLMMetadata.is_open_weights == False)
    
    if "reasoning_model" in constraints and constraints["reasoning_model"]:
        query = query.filter(LLMMetadata.reasoning_type != "none")
    
    if "tool_calling" in constraints and constraints["tool_calling"]:
        query = query.filter(LLMMetadata.tool_calling != "none")
    
    if "min_speed_class" in constraints:
        speed_order = {"slow": 0, "medium": 1, "fast": 2}
        min_speed_value = speed_order.get(constraints["min_speed_class"], 0)
        query = query.filter(
            or_(
                LLMMetadata.speed_class == constraints["min_speed_class"],
                and_(
                    LLMMetadata.speed_class == "medium",
                    constraints["min_speed_class"] == "slow"
                ),
                and_(
                    LLMMetadata.speed_class == "fast",
                    constraints["min_speed_class"] in ["slow", "medium"]
                )
            )
        )
    
    filtered_models = query.all()
    
    if not filtered_models:
        return {
            "top_performance": [],
            "balanced": [],
            "budget": [],
            "metadata": {
                "applied_io_ratio": token_ratio_estimation,
                "benchmarks_used": [],
                "benchmark_weights": []
            }
        }
    
    # Step 2: Score Retrieval
    benchmark_ids = [bw["id"] for bw in benchmark_weights]
    model_results = []
    
    for model in filtered_models:
        benchmark_results = []
        performance_scores = []
        
        for bw in benchmark_weights:
            benchmark_id = bw["id"]
            weight = bw["weight"]
            
            # Try to get actual score
            score_record = (
                session.query(BenchmarkScore, BenchmarkDictionary)
                .join(BenchmarkDictionary)
                .filter(
                    and_(
                        BenchmarkScore.model_id == model.id,
                        BenchmarkScore.benchmark_id == benchmark_id
                    )
                )
                .order_by(BenchmarkScore.date_published.desc().nullslast())
                .first()
            )
            
            if score_record:
                score, benchmark = score_record
                benchmark_results.append({
                    "benchmark_id": benchmark.id,
                    "benchmark_name": benchmark.name_normalized,
                    "benchmark_variant": benchmark.variant,
                    "score": score.score_value,
                    "metric_unit": score.metric_unit,
                    "weight_used": weight,
                    "is_estimated": False,
                    "source_url": score.source_url
                })
                performance_scores.append(score.score_value * weight)
            else:
                # Try bridge model calibration
                benchmark = session.query(BenchmarkDictionary).filter(
                    BenchmarkDictionary.id == benchmark_id
                ).first()
                
                if benchmark:
                    estimated_score, estimation_note = _apply_bridge_model_calibration(
                        session, model.id, benchmark_id, benchmark.variant
                    )
                    
                    if estimated_score is not None:
                        benchmark_results.append({
                            "benchmark_id": benchmark.id,
                            "benchmark_name": benchmark.name_normalized,
                            "benchmark_variant": benchmark.variant,
                            "score": estimated_score,
                            "metric_unit": "%",  # Default assumption
                            "weight_used": weight,
                            "is_estimated": True,
                            "estimation_note": estimation_note
                        })
                        performance_scores.append(estimated_score * weight)
        
        if benchmark_results:  # Only include models with at least some benchmark data
            # Step 4: Calculate Blended Cost
            blended_cost = _calculate_blended_cost(model, token_ratio_estimation)
            
            model_results.append({
                "model_id": model.id,
                "name_normalized": model.name_normalized,
                "speed_class": model.speed_class,
                "speed_tps": model.speed_tps,
                "benchmark_results": benchmark_results,
                "blended_cost_1m_usd": blended_cost,
                "raw_performance_score": sum(performance_scores)
            })
    
    if not model_results:
        return {
            "top_performance": [],
            "balanced": [],
            "budget": [],
            "metadata": {
                "applied_io_ratio": token_ratio_estimation,
                "benchmarks_used": [bw.get("name") for bw in benchmark_weights],
                "benchmark_weights": [bw.get("weight") for bw in benchmark_weights]
            }
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
        copy.deepcopy(model_results),
        key=lambda x: x["performance_index"],
        reverse=True
    )

    # Budget List: Ranked by 0.2 * Performance_Index + 0.8 * Blended_Cost_Index
    budget = sorted(
        copy.deepcopy(model_results),
        key=lambda x: 0.2 * x["performance_index"] + 0.8 * x["blended_cost_index"],
        reverse=True
    )

    # Balanced List: Ranked by 0.5 * Performance_Index + 0.5 * Blended_Cost_Index
    balanced = sorted(
        copy.deepcopy(model_results),
        key=lambda x: 0.5 * x["performance_index"] + 0.5 * x["blended_cost_index"],
        reverse=True
    )
    
    # Add ranking metadata and clean up
    for rank_list in [top_performance, balanced, budget]:
        for i, mr in enumerate(rank_list):
            if rank_list == top_performance:
                mr["rank_metrics"] = {
                    "performance_index": mr["performance_index"],
                    "blended_cost_index": mr["blended_cost_index"],
                    "blended_score": mr["performance_index"]  # Performance list uses performance score
                }
                mr["reason_for_ranking"] = f"Performance Index: {mr['performance_index']:.3f}"
            elif rank_list == budget:
                blended_score = 0.2 * mr["performance_index"] + 0.8 * mr["blended_cost_index"]
                mr["rank_metrics"] = {
                    "performance_index": mr["performance_index"],
                    "blended_cost_index": mr["blended_cost_index"],
                    "blended_score": blended_score
                }
                mr["reason_for_ranking"] = f"Budget-optimized score: {blended_score:.3f}"
            else:  # balanced
                blended_score = 0.5 * mr["performance_index"] + 0.5 * mr["blended_cost_index"]
                mr["rank_metrics"] = {
                    "performance_index": mr["performance_index"],
                    "blended_cost_index": mr["blended_cost_index"],
                    "blended_score": blended_score
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
            "benchmark_weights": [bw.get("weight", 0) for bw in benchmark_weights]
        }
    }


def execute_ranking(state: AgentState, config: RunnableConfig) -> AgentState:
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

    ranked_results = retrieve_and_rank_models(
        benchmark_weights=state.get("weighted_benchmarks", []),
        constraints=constraints,
        token_ratio_estimation=token_ratio_estimation,
        session=session,
    )
    state["ranked_results"] = ranked_results
    return state