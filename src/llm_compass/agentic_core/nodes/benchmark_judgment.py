"""
Req 2.3 Node 3 (b): Benchmark Selection

This node LLM node judges the benchmarks found in Node 3 (a) benchmark_discovery
and decides about their weight

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
from ..schemas import BenchmarkJudgments


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
- Return only schema-valid structured data.
- Keep rationales short and concrete.
- If a benchmark is not relevant, say why briefly."""


def benchmark_judgment_node(state: AgentState, settings: Settings):
    # todo
    pass
