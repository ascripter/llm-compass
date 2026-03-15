"""
Req 2.3 Node 3 (b): Benchmark Selection

This node LLM node judges the benchmarks found in Node 3 (a) benchmark_discovery
and decides about their weight

Outputs weighted_benchmarks: List[Dict] with id and weight.
"""

import logging
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from llm_compass.config import Settings
from ..state import AgentState
from ..schemas import BenchmarkJudgments, IntentExtraction


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a benchmark relevance judge in an LLM routing pipeline.

Your job is to judge how well each candidate benchmark matches the user's task.

A benchmark is relevant only if it meaningfully measures the capability needed for the task.
Do not reward shallow keyword overlap.
Judge semantic task fit, not wording similarity.

Evaluate each benchmark on:
1. Core task match.
2. Input modality match.
3. Output modality match.
4. Whether the benchmark measures the needed capability directly or only loosely.

Scoring rule:
- Choose exactly one relevance_class from the schema.
- Use the class definitions in the schema strictly.
- Be slightly conservative: if unsure between two classes, choose the lower one.
- "perfect_match" should be rare and used only when the benchmark is a direct fit for the task.

Output rules:
- You MUST return a judgment for every benchmark provided — including those you consider irrelevant (use "no_match").
- Return only schema-valid structured data.
- Keep rationales short and concrete.
- If a benchmark is not relevant, say why briefly."""


def benchmark_judgment_node(state: AgentState, *, settings: Settings) -> dict:
    """
    Node 3(b): LLM judges each candidate benchmark's relevance to the user task.
    Reads weighted_benchmarks from state (output of benchmark_discovery_node).
    Outputs benchmark_judgements: BenchmarkJudgments.
    """
    weighted_benchmarks = state.get("weighted_benchmarks", [])
    if not weighted_benchmarks:
        logger.warning("No weighted_benchmarks found in state — skipping judgment")
        return {"benchmark_judgements": BenchmarkJudgments(judgments=[])}

    user_query = state.get("user_query", "")

    # Reconstruct intent (may arrive as dict after LangGraph checkpoint)
    intent_raw = state.get("intent_extraction")
    intent: IntentExtraction | None = (
        IntentExtraction(**intent_raw) if isinstance(intent_raw, dict) else intent_raw
    )

    # Build modality context line
    modality_lines = ""
    if intent is not None:
        modality_lines = (
            f"\nInput modalities: {', '.join(intent.intended_input_modalities)}"
            f"\nOutput modalities: {', '.join(intent.intended_output_modalities)}"
        )

    # Format each benchmark as a readable block for the LLM
    benchmark_blocks = []
    for bm in weighted_benchmarks:
        variant_str = f" ({bm['variant']})" if bm.get("variant") else ""
        lines = [f"- ID={bm['id']}: {bm['name_normalized']}{variant_str}"]
        if bm.get("description"):
            description = bm["description"].replace("\n", " ")
            lines.append(f"  Description: {description}")
        if bm.get("categories"):
            lines.append(f"  Categories: {bm['categories']}")
        benchmark_blocks.append("\n".join(lines))

    human_content = (
        f"# User task: {user_query}"
        f"{modality_lines}"
        f"\n\n# Candidate benchmarks to judge:\n" + "\n\n".join(benchmark_blocks)
    )

    llm = settings.make_llm("openai/gpt-5-mini", temperature=0)
    structured_llm = llm.with_structured_output(BenchmarkJudgments)

    logger.debug(
        "benchmark_judgment_node ENTRY | benchmarks=%d | human_message=\n%s",
        len(weighted_benchmarks),
        human_content,
    )

    judgments: BenchmarkJudgments = cast(
        BenchmarkJudgments,
        structured_llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_content)]
        ),
    )

    n_relevant = sum(1 for j in judgments.judgments if j.relevance_weight > 0.0)
    logger.info(
        "benchmark_judgment_node EXIT | judged=%d | relevant=%d",
        len(judgments.judgments),
        n_relevant,
    )

    return {
        "benchmark_judgements": judgments,
        "logs": [f"{n_relevant} are relevant"],
    }
