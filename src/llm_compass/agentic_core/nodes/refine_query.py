"""
Req 2.3 Node 2: Query Refiner (LLM)

This node runs after intent validation succeeds.
It performs:
1. Token ratio estimation (modality-aware units + normalized ratios)
2. Search query generation (3 to 5 benchmark-discovery queries)
"""

from typing import Any
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from llm_compass.common.schemas import Constraints
from ..schemas.validate_intent import IntentExtraction, TokenRatioEstimation
from ..schemas.refine_query import QueryExpansion
from ..state import AgentState

logger = logging.getLogger(__name__)

TOKEN_RATIO_SYSTEM_PROMPT = """You are a token volume estimation assistant in an AI routing pipeline.

Given the validated user task and active UI constraints:
1. Estimate realistic input units and output units for text, image, audio, video.
2. Keep estimates practical for a typical single LLM invocation.
3. Follow the output schema strictly.
"""

QUERY_EXPANSION_SYSTEM_PROMPT = """You are a benchmark-discovery query generation assistant.

Given the validated user task and active UI constraints:
1. Generate 3-5 distinct semantic search queries for benchmark retrieval.
2. Queries should focus on measurable evaluation needs (reasoning, coding, summarization, multimodal understanding, etc.).
3. Keep queries concise and specific enough for vector retrieval.
4. Follow the output schema strictly.
"""


def _ensure_query_count(search_queries: list[str], user_query: str) -> tuple[list[str], bool]:
    unique: list[str] = []
    seen: set[str] = set()
    for query in search_queries:
        cleaned = query.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            unique.append(cleaned)
            seen.add(key)

    seed = user_query.strip() or "LLM benchmark recommendation"
    fallback_candidates = [
        f"{seed} benchmark",
        f"{seed} evaluation",
        f"{seed} performance benchmark",
        "llm benchmark leaderboard",
        "model evaluation benchmark suite",
    ]
    used_fallback = False
    for candidate in fallback_candidates:
        key = candidate.lower()
        if len(unique) >= 5:
            break
        if key in seen:
            continue
        if len(unique) < 3:
            used_fallback = True
        unique.append(candidate)
        seen.add(key)

    return unique[:5], used_fallback


def query_refiner_node(state: AgentState) -> dict[str, Any]:
    """
    Refines a validated query (Req 2.3 Node 2).

    Returns:
        dict[str, Any]:
            - token_ratio_estimation: TokenRatioEstimation
            - search_queries: list[str]
            - logs: list[str]
    """
    llm = ChatOpenAI(model="openai/gpt-oss-120b", temperature=0)
    token_estimator = llm.with_structured_output(TokenRatioEstimation)
    query_expander = llm.with_structured_output(QueryExpansion)

    constraints_raw = state.get("constraints")
    constraints = (
        Constraints(**constraints_raw) if isinstance(constraints_raw, dict) else constraints_raw
    )
    if constraints is None:
        constraints = Constraints(min_context_window=0)

    intent_raw = state.get("intent_extraction")
    intent = IntentExtraction(**intent_raw) if isinstance(intent_raw, dict) else intent_raw

    messages = state.get("messages", [])
    if not messages:
        messages = [HumanMessage(content=str(state.get("user_query", "")).strip())]

    context_lines = [
        f"User query: {state.get('user_query', '')}",
        f"UI constraints: {constraints.model_dump_json()}",
    ]
    if intent is not None:
        context_lines.append(
            "Validated intent modalities:"
            f" input={intent.intended_input_modalities}, output={intent.intended_output_modalities}"
        )
    context = "\n".join(context_lines)

    token_messages = [SystemMessage(content=TOKEN_RATIO_SYSTEM_PROMPT)] + messages + [
        HumanMessage(content=context)
    ]
    token_ratio_estimation: TokenRatioEstimation = token_estimator.invoke(token_messages)  # type: ignore[assignment]

    query_messages = [SystemMessage(content=QUERY_EXPANSION_SYSTEM_PROMPT)] + messages + [
        HumanMessage(content=context)
    ]
    query_expansion: QueryExpansion = query_expander.invoke(query_messages)  # type: ignore[assignment]

    search_queries, used_fallback = _ensure_query_count(
        query_expansion.search_queries,
        str(state.get("user_query", "")),
    )

    logs = [
        (
            "Query Refiner: token ratios estimated. "
            f"input={token_ratio_estimation.normalized_input_ratios}, "
            f"output={token_ratio_estimation.normalized_output_ratios}"
        ),
        f"Query Refiner: generated {len(search_queries)} search queries.",
    ]
    if used_fallback:
        logs.append(
            "Query Refiner: query list had fewer than 3 unique entries; fallback queries were appended."
        )

    return {
        "token_ratio_estimation": token_ratio_estimation,
        "search_queries": search_queries,
        "logs": logs,
    }
