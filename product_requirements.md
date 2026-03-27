# Product Requirements

## 0. MVP scope (must / not)
### Must (MVP)
- Answer: “Which LLM is best suited for task X under constraints Y?” using stored benchmark + metadata, with source links.
- Show why: benchmarks chosen, constraints applied (i.e. modality filter), scores used, and what data is missing.
- Support a small, high-quality dataset of benchmarks (curated) rather than “complete coverage”.

### Not in MVP (explicitly out)
- Autonomous discovery agent that scans arXiv/blogs and extracts new benchmarks/models.
- Inter-variant score estimation (“offset calibration”, bridge models).
- Automated deprecation of outdated models.

## 1. Data Foundation

### 1.1. Data Ingestion Strategy
The system employs a dual-channel ingestion approach to build a comprehensive dataset.

#### A. Scheduled Aggregation
- **Frequency:** Bi-Weekly (configurable).
- **Sources (MVP):** 1–3 curated benchmark aggregators + optional provider pages.
- **Sources (extended):** Targeted scraping of known benchmark aggregators (e.g., artificialanalysis.ai, llm-stats.com).
  - **Constraint:** Check websites' `robots.txt` and Terms of Service. If scraping is prohibited, implement a manual data entry workflow or request API access / licensing from provider.
- **Goal:** Maintain up-to-date scores for established models and benchmarks.

#### B. Manual import (required for MVP)
- Allow admin to upload CSV (scores and/or model metadata).
- Validate schema, store original strings, and surface “unmatched entities” for review.

#### C. Agentic Web Research (not in MVP)
- **Trigger:** Scheduled or event-driven.
- **Agent Behavior:** Autonomous research agent scans arXiv, technical blogs, and research papers.
- **Goal:** Discover *new* benchmarks and models not available on aggregator sites.



### 1.2. Database Schema

#### A. Research Memory (Table)
*Tracks the history of the Research Agent to prevent redundant scraping.*

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary Key |
| `source_url` | String | URL or DOI of the paper/site |
| `source_type` | String | Enum: `paper`, `blog`, `aggregator` |
| `extraction_date` | DateTime | Timestamp of processing |
| `processing_status` | String | Status of data extraction (e.g., `success`, `failed`) |

#### B. Benchmark Dictionary (Vector Store)
*Stores semantic definitions to enable "smart" lookup of benchmarks based on user intent.*

| Field | Type | Description |
|-------|------|-------------|
| `benchmark_id` | UUID | Primary Key |
| `name_normalized` | String | Standardized name (e.g., "MMLU") |
| `variant` | String | Version or setting (e.g., "5-shot", "CoT") |
| `description` | Text | Vectorized field describing what it measures |
| `embedding_model_name` | Text | Name of the embedding model that vectorized `description` |
| `embedding_model_version` | Text | Version of the embedding model that vectorized `description` |
| `categories` | List<Str> | Tags (e.g., `["reasoning", "coding", "agentic"]`) |


#### C. Benchmark Scores (Table)
*The core repository of raw performance data. Normalized IDs used for integrity; Views used for readability.*

| Field | Type | Description |
|-------|------|-------------|
| `score_id` | UUID | Primary Key |
| `model_id` | UUID | Foreign Key to LLM Metadata |
| `benchmark_id` | UUID | Foreign Key to Benchmark Dictionary |
| `score_value` | Float | The numerical result |
| `metric_unit` | String | Unit of measurement (e.g., `%`, `elo`, `pass@1`) |
| `source_name` | String | e.g. `ArtificialAnalysis`, `Paper`, `Provider` |
| `source_url` | String | Specific URL where this score was found |
| `date_published` | Date | When the score was published. Nullable. |
| `date_ingested` | Datetime | When the score was ingested into the system |
| `original_model_name` | String | **Raw string** from source (for audit/debugging) |
| `original_benchmark_name`| String | **Raw string** from source (for audit/debugging) |

#### D. LLM Metadata (Table)
*Stores static attributes for filtering and tradeoff analysis.*

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | UUID | Primary Key |
| `name_normalized` | String | Standardized model name (e.g., "llama-3.1-70b-instruct") |
| `name_aliases` | List<Str> | List of alternative identifiers for the model |
| `model_type` | String | Enum: `base`, `instruct`, `thinking` or `generator` |
| `provider` | String | Company/Org (e.g., Meta, OpenAI, Mistral) |
| `release_date` | Date | When the model was released |
| `parameter_count` | Float | Parameters in Billions (e.g., 70.0). Nullable. |
| `architecture` | String | Enum/String: `Dense`, `MoE`, `SSM` (e.g., Mamba), `RNN`. Nullable. |
| `available_quantizations` | List<Str> | e.g., `q4_k_m`, `fp8`, `int8`. Null for API models / unknown. |
| `distillation_source` | String | Stores parent model name if known (e.g., `GPT-4`). Nullable. |
| `modality_input` | List<Str> | Capabilities (e.g., `["text", "image", "audio"]`) |
| `modality_output` | List<Str> | Capabilities (e.g., `["text", "code"]`) |
| `context_window` | Integer | Max input tokens. Nullable. |
| `max_output_tokens` | Integer | Max output tokens per query. Nullable. |
| `cost_input_text_1m` | Float | Price per 1M input tokens of text ($). Nullable. |
| `cost_output_text_1m` | Float | Price per 1M output tokens of text ($). Nullable. |
| `cost_input_image_1k` | Float | Price per input image (1024 x 1024 pixels standard / medium quality) ($). Nullable. |
| `cost_output_image_1k` | Float | Price per output image (1024 x 1024 pixels standard / medium quality) ($). Nullable. |
| `cost_input_audio_1h` | Float | Price per hour of audio input ($). Nullable. |
| `cost_output_audio_1h` | Float | Price per hour of audio output ($). Nullable. |
| `cost_input_video_1s` | Float | Price per second of video input (standard / medium quality, HD resolution 1280 x 720 pixels) ($). Nullable. |
| `cost_output_video_1s` | Float | Price per second of video output (standard / medium quality, HD resolution 1280 x 720 pixels) ($). Nullable. |
| `raw_cost_notes_input` | String | Notes from LLM research (i.e on non-text modality pricing). Nullable. |
| `raw_cost_notes_output` | String | Notes from LLM research (i.e on non-text modality pricing). Nullable. |
| `speed_class` | String | Enum: `fast` (>100 tps), `medium` (40-100 tps), `slow` (<40 tps). |
| `speed_tps` | Float | Approximate value (highly dependant on provider). Nullable. |
| `is_open_weights` | Boolean | Whether this is an open weights model |
| `license` | String | License under which the (open weights) model is distributed. Nullable. |
| `reasoning_type`| String | Enum: `none`, `standard` (reason only when prompted), `native cot`(using Chain-of-Thought natively). Nullable. |
| `tool_calling` | String | Enum: `none`, `standard` (JSON function calling), `agentic` (built-in native tools). Nullable. |
| `is_outdated` | Boolean | Flag to hide superseded models (default: `false`) |
| `superseded_by_model_id` | UUID | For models with `is_outdated=true`: Link to newer model version. Nullable |

### 1.3. Data Processing Rules

#### A. Write-Time Normalization
- **Mechanism:** Use a lightweight/cheap LLM (e.g., GPT-4o-mini, Haiku) during ingestion.
- **Task:** Map incoming strings to standardized entities.
  - *Input:* "llama-2-7b-chat-hf" -> *Output:* "Llama 2 7B Chat"
  - *Input:* "HumanEval pass@1" -> *Output:* "HumanEval" (Variant: "pass@1")
- **Constraint:** The system must strictly preserve the `original_model_name` and `original_benchmark_name` in the Scores table to allow developers to audit normalization errors.

#### B. Score Aggregation Logic (Read-Time)
- **Requirement:** The UI must present a consolidated view to the user.
- **Logic:** Default “best available” selection rule in read path for a given `(model_id, benchmark_id)` tuple: prefer source_type priority (provider/paper > curated aggregator > unknown), then newest `date_published`, else newest `date_ingested`.



## 2. Agentic Core

### 2.1. System Architecture

#### A. Persona Definition
- **Role:** "The Benchmark Analyst"
- **Identity:** An autonomous consultant for AI developers.
- **Prime Directive:** "Recommends models strictly based on retrieved data. Explicitly states 'No data available' instead of guessing. Uses statistical offsets to infer performance across benchmark variants where possible."
- **Tone:** Professional, Objective, Data-Driven.
- **System Prompt Draft:**
> You are The Benchmark Analyst. You assist developers in selecting LLMs based on empirical evidence.
>
> **CORE RULES:**
> 1.  **No Guessing:** If `retrieve_and_rank_models` returns no data for a given model, you must classify it as "Insufficient Data".
> 2.  **Strict Citation:** Every performance claim must reference the specific benchmark used.
> 3.  **Variant Awareness:** If you used the "Offset Calibration" logic (inferred scores), you must tag the result as `estimated: true` in the output.
> 4.  **Format:** * Output must ALWAYS be valid JSON matching the `AgentResponse` schema.

#### B. Structured Output Schema (JSON)
*The Agent does not just "chat back"; it returns a data object the UI can render.*

```json
{
  "user_query": "...",
  "applied_constraints": {"...": "..."},  // from UI widgets
  "status": "ok|needs_clarification|error",
  "traceability": {
    "events": [
      {"stage": "intent_validator", "message": "...", "data": {}}
    ]
  },
  "ui_components": {
    "summary_markdown": "## Executive Summary\nBased on your constraints (Open Weights, Coding Focus), **Llama 3 70B** is the top recommendation...",
    "comparison_table": {
      "title": "Coding Performance (HumanEval + MBPP)",
      "columns": ["Model", "Score (Avg)", "Cost ($/1M)", "Speed (tps)", "Est?"],
      "rows": [
        ["Llama 3 70B", "82.5%", "0.90", "80", false],
        ["DeepSeek Coder", "79.1%", "0.20", "12", true]
      ]
    },
    "recommendation_cards": [
      {
        "category": "Top Performance",
        "model_name": "Llama 3 70B",
        "reason": "Highest pure coding score among open weights."
      },
      {
        "category": "Budget Pick",
        "model_name": "DeepSeek Coder",
        "reason": "Unbeatable price/performance ratio."
      }
    ],
    "citations": [{"id": "...", "label": "...", "url": "https://..."}],
    "warnings": [{"code": "...", "message": "..."}]
  },
  "errors": [{"code": "...", "message": "..."}]
}
```

### 2.2. Tools
*These tools are executed by the Orchestrator based on the Plan.*

#### A. `find_relevant_benchmarks(queries: List[str], cutoff_score: float = 0.5)`
- **Input:** A list of 3-5 distinct search queries (generated by the "Rephrase" node) and a relevance cutoff threshold.
- **Action:**
  1. Perform vector search against the `Benchmark Dictionary` for each query.
  2. Normalize vector similarity scores to a 0-1 range.
  3. **Aggregation:** If a benchmark appears in multiple query results, sum or average its scores to boost its relevance.
  4. **Filtering:** Return only benchmarks with a final relevance score > `cutoff_score`.
- **Output:** List of objects: `[{id: 12, name_normalized: "HumanEval", ..., score: 0.95}, ...]`.


#### B. `retrieve_and_rank_models(benchmark_weights, constraints, token_ratio_estimation)`
- **Input:**
  - `benchmark_weights`: List[Dict] (from Tool A)
  - `constraints`: Dict (from UI)
  - `token_ratio_estimation`: Dict `{"normalized_input_ratios": dict[Modality, float], "normalized_output_ratios": dict[Modality, float]}` (e.g., `{"normalized_input_ratios": {"text": 0.1, "image": 0.75}, "normalized_output_ratios": {"text": 0.15}}`). All values sum to `1.0`
- **Action:**
  1. **Model Filtering:** Query `LLM Metadata` to find ALL models that match the hard constraints from UI input (`is_outdated=false` by default).
  2. **Score Retrieval:** For the filtered models, fetch scores for the *relevant benchmarks*.
  3. **Score Normalization (Inter-Variant Calibration):**
     - *Algorithm:* If Model X lacks data for Variant A but has data for Variant B, use "Bridge Models" (models with both A and B scores) to calculate an offset `Delta = Avg(Score_B) - Avg(Score_A)`. Estimate `Score_A(X) = Score_B(X) - Delta`.
     - *Constraint:* Mark estimated scores as `is_estimated: true`.
  4. **Dynamic Cost Weighting:**
     - *Definition:* `Blended_Cost_1M_USD = sum(token_ratio_estimation[mode][modality] * cost-from-llm_metadata) `, i.e. 
     ```
     Blended_Cost_1M_USD = 
      token_ratio_estimation["normalized_input_ratios"]["text"] * cost_input_text_1m + 
      token_ratio_estimation["normalized_input_ratios"]["image"] * cost_input_image_1024 +
      token_ratio_estimation["normalized_output_ratios"]["text"] * cost_output_text_1m
     ```
     - *Definition:* `Blended_Cost_Index` is a normalized "cheaper is better" score [0,1].
     - *Formula (Per-Query Normalization):* `Blended_Cost_Index = (C_max - Blended_Cost_1M_USD) / (C_max - C_min)`, where `C_min` and `C_max` are the min/max blended costs *within the current filtered candidate set*. If all candidates have equal cost, assign `Blended_Cost_Index = 0.5` to all. Models with fully unknown pricing (`cost_null_fraction = 1.0`) are also assigned `Blended_Cost_Index = 0.5` to avoid inflating their rank.
  5. **Ranking Strategy (Multi-View):**
     - **Performance List:** Ranked by `Performance_Index` (Weighted Average of Normalized Scores).
     - **Balanced List:** Ranked by `0.5 * Performance_Index + 0.5 * Blended_Cost_Index`.
     - **Budget List:** Ranked by `0.2 * Performance_Index + 0.8 * Blended_Cost_Index`.
- **Output:** JSON object containing three ranked lists (`top_performance`, `balanced`, `budget`) with full `benchmark_results` breakdown for transparency. `benchmark.source_url` is required for every non estimated score. `benchmark.estimation_note` is required for every estimated score, i.e.
```json
{
  "top_performance": [
    {
      "model_id": "uuid-123",
      "name_normalized": "Llama 3 70B",
      "provider": "Meta",
      "speed_class": "slow|medium|fast",
      "speed_tps": 80,
      "cost_null_fraction": 0.20, // if not all modalities have costs known
      "rank_metrics": {
          "performance_index": 0.88,
          "blended_cost_index": 0.45,
          "blended_score": 0.88  // (because this is the Performance List)
      },
      "benchmark_results": [
          {
              "benchmark_id": "...",
              "benchmark_name": "HumanEval",
              "benchmark_variant": "...",
              "score": 82.5,
              "metric_unit": "%|elo|pass@1", 
              "weight_used": 0.5,
              "is_estimated": false,
              "source_url": "https://..."
          },
          {
              "benchmark_id": "...",
              "benchmark_name": "MBPP",
              "benchmark_variant": "...",
              "score": 70.2,
              "metric_unit": "%|elo|pass@1", 
              "weight_used": 0.3,
              "is_estimated": true,
              "estimation_note": "Inferred via bridge model 'CodeLlama'",
          },
           {
              "benchmark_id": "...",
              "benchmark_name": "Swe-Bench",
              "benchmark_variant": "...",
              "score": 22.0,
              "metric_unit": "%|elo|pass@1", 
              "weight_used": 0.2,
              "is_estimated": false,
              "source_url": "https://..."
          },
          // ... more items
      ],
      "reason_for_ranking": "Dominates in HumanEval, solid baseline in Swe-Bench."
    },
    // ... more items
  ],
  "balanced": [
    // ... more items
  ],
  "budget": [
    // ... cheap models
  ],
  "metadata": {
    "applied_io_ratio": {...},
    // redundant but easily accessible
    "benchmarks_used": ["HumanEval", "MBPP", "Swe-Bench"],
    "benchmark_weights": [0.5, 0.3, 0.2]
  }
}
```


### 2.3. Agent Workflow (Orchestrator)
*Implementation: **LangGraph**. A strict, deterministic flow is required (Validate -> Rephrase -> Search -> Rank -> Synthesize), not a loose "crew" of chatting agents. The state object must carry the structured UI inputs throughout the chain*


#### Node 1: Intent Validator (LLM)
- **Input:** User Query (original or appended with clarification response) + UI Constraints.
- **Task:** 
  1. **Task Description Check:** Is the task properly defined and concrete enough?  
     - Better: "Summarize 100 pages of legal documents and highlight inconsistencies."  
     - Worse: "Text summary."
  2. **Ambiguity Check (I/O Ratio):** Does the query imply a clear `input/output` token ratio?  
     - Example: "Summarize this PDF" ⇒ High Input / Low Output.
     - Example: "Write a sci‑fi novel from this short outline" ⇒ Low Input / High Output.
  3. **Consistency Check (Query vs. UI Filters):** Does the user's textual intent conflict with active UI filters?
     - *Hard Mismatch Example (impossible with current filters):* Query "Summarize 100‑page PDF" vs. Filter `modality_input=["image"]`.
     - *Weak Mismatch Example (over‑constrained, likely sub‑optimal):* Query "Which model is best for describing video content?" vs. Filter `modality_input=["text", "video"]` since we ONLY need video.
- **Output:**
  - `status`: `"valid"` | `"needs_clarification"`
  - `clarification_question`: String (if `status = "needs_clarification"`).
    - *If Under-Specified:*  
      - Example: "To recommend models, I need more detail. What is the main task: coding, document summarization, data analysis, or something else?"
    - *If Ambiguous I/O:*  
      - Example: "Will this model mostly read long inputs (like documents or codebases), or mostly generate long outputs (like stories or reports)?"
    - *If Mismatch (hard):*
      - Example: "You asked about summarizing PDFs (text) but have 'Image Input' selected. Please refine your task description or adjust the modality selector."
    - *If Mismatch (weak):*
      - Example: "You asked about describing video content, but your filters require both text and video input. For this task, video input alone is sufficient. Do you want to relax the text requirement so more suitable models are considered?"
- **Clarification Limit:** After 3 consecutive `needs_clarification` cycles for the same user session, the agent must stop and return a polite error message (e.g., "I’m still missing key details. Please restate your task with more specifics.").


#### Router: Clarification Gate (Conditional Edge)
- **Logic:**
  - If `status == "needs_clarification"`: **Pause execution.** Return `clarification_question` to UI. Wait for user reply, then re-enter Node 1 with the updated query.
  - If `status == "valid"`: Proceed to **Node 2 (Query Refiner)**.

#### Node 2 (a): Query Refiner (LLM)
- **Input:** Validated User Query + UI Constraints.  
- **Search Query Gen:** Generate 3-5 benchmark search queries from user's input.
- **Output:** `{"search_queries": [...]}`

#### Node 2 (b): Token Ratio Estimation (LLM)
- **Input:** Validated User Query + UI Constraints.  
- **Predict I/O Ratio:** Estimate `input` vs. `output` token volume per modality. Output eight floats summing to 1.0.
     - *Example (RAG):* `{"input_text": 0.95, "output_text": 0.05}`
     - *Example (Long Chat):* `{"input_text": 0.75, "output_text": 0.25}`
     - *Example (Simple Chat):* `{"input_text": 0.5, "output_text": 0.5}`
     - *Example (Novel Writing):* `{"input_text": 0.1, "output_text": 0.9}`
- **Output:** `{ "normalized_input_ratios": {...}, "normalized_output_ratios": {...} }`

#### Node 3 (a): Benchmark Discovery (Tool)
- **Input:** `search_queries` from Node 2.
- **Task:** Execute `find_relevant_benchmarks`.
- **Output:** `weighted_benchmarks: List[Dict]` (e.g., `[{"id": "mmlu", "weight": 0.9}, {"id": "gpqa", "weight": 0.8}]`).

#### Node 3 (b): Benchmark Judgment (Tool)
- **Input:** `weighted_benchmarks` from Node 3a.
- **Task:** LLM judging which pre-filtered benchmarks are actually relevant.
- **Output:** `BenchmarkJudgments: List[dict]` (e.g., `[{"benchmark_id": "mmlu", short_rationale": "...", "relevance_weight": 0.75}, ...]`).

#### Node 4: Scoring & Ranking (Tool)
- **Input:** `BenchmarkJudgments` (from Node 3b) + `constraints` and `token_ratio_estimation` (from Global State / Node 2).
- **Task:** Execute `retrieve_and_rank_models`.
- **Logic:**
  - Fetch models.
  - Apply Inter-Variant Calibration (Offset Logic).
  - Generate the 3 Ranking Lists (Performance, Balanced, Budget).
- **Output:** `ranked_lists: Dict[str, List]` (The three lists).

#### Node 5: Synthesis (LLM + Deterministic)
- **Input:** `ranked_lists` (`RankedLists` from Node 4) + full `AgentState`
- **Task:** Produce the final `SynthesisOutput` (aliased as `UIComponents` in the API
  layer) for the frontend by combining LLM-generated text with deterministically-built
  structured data.

  **LLM-Generated Components:**
  1. **Task Summary:** 1-2 sentence rephrasing of the user's intended task.
  2. **Executive Summary:** 3-5 sentence markdown highlighting top performance
     winner, budget winner, key trade-offs, and most relevant benchmarks.
  3. **Recommendation Reasons:** One reason per category (`top_performance`,
     `balanced`, `budget`) explaining why the model wins, referencing benchmarks.
  4. **Offset Calibration Note:** If any `is_estimated=true` scores exist,
     note which models/benchmarks were estimated. Null otherwise.

  **Deterministic Components (no LLM):**
  1. **Comparison Table:** Base columns: Model, Provider, Blended Score,
     Cost Index, Speed (tps), Est?. Additional columns: top-N benchmark names
     by weight. One row per unique model (deduplicated across lists), sorted
     by `blended_score` desc.
  2. **Recommendation Cards:** Top-1 model from each ranking list. If the same
     model wins multiple categories, collapse to fewer cards.
  3. **Citations:** Unique `(benchmark_name, source_url)` pairs from all
     non-estimated benchmark results. Deduplicated by URL.
  4. **Warnings:** Data quality signals:
     - `LOW_RELEVANCE`: `average_benchmark_similarity < 0.6`
     - `PARTIAL_COST_DATA`: any top-3 model has `cost_null_fraction > 0.3`
     - `ESTIMATED_SCORES`: any top-3 model uses estimated scores
     - `FEW_CANDIDATES`: any ranking list has < 3 models

- **Output:** `SynthesisOutput` stored as `final_response` in `AgentState`.
  Fields: `summary_markdown`, `comparison_table`, `recommendation_cards`,
  `citations`, `warnings`.
- **Assembly:** `summary_markdown` is:
  ```
  ## Your Task
  {task_summary}

  ## Recommendations
  {executive_summary}

  > **Note:** {offset_calibration_note}  (only if estimated scores exist)
  ```


## 3. UI/UX

### 3.1. Design Philosophy
- **Vibe:** "Mission Control" meets "Chat." Dense information, dark mode by default, monospace fonts for data.
- **Layout:** Two-column split (Desktop) or Stacked (Mobile; optional).
  - **Left/Top:** Chat Interface & Active Constraints.
  - **Right/Bottom:** "Traceability View" (Live Agent Thought Process) & Results.

### 3.2. Input Interface (The "Setup" Panel)
*Before (or alongside) the chat, users configure hard constraints to ground the agent.*

#### A. Constraint Controls
1.  **Minimum Context Window (integer field):**
    - Default: 0
    - Label: "Min Context: {Value}"
2.  **Modality Selectors (Multi-Select Chips):**
    - **Input:** `Text` (Default), `Image`, `Audio`, `Video`.
    - **Output:** `Text` (Default), `Image`, `Audio`, `Video`
    - **Logic:** **AND** (Intersection). "Show models that support ALL selected inputs."
    - *Hint Text:* "Select all required capabilities (e.g., Image + Text for visual reasoning)."
3.  **Deployment (Segmented Control):**
    - Options: `Any` | `Cloud API` | `Local / Open Weights`.
    - **Logic:** `Local` filters for `is_open_weights = TRUE`.
4.  **Capabilities (Toggles):**
    - `[ ] Reasoning Model`: Filters for `is_reasoning_model=true` (e.g., o1, R1). Default is `false` (i.e. any model / no filter).
    - `[ ] Tool Calling`: Filters for `has_tool_calling=true`. Default is `false` (i.e. any model / no filter).
5.  **Minimum speed (Segmented Control):**
    - Options: `Any` | `Medium+` | `Fast`
    - *Hint text:* "Filters models by expected throughput class. TPS can vary by provider/hardware."

#### B. Chat Input
- Standard text area.
- Placeholder: "Describe your task (e.g., 'I need a model for RAG on legal documents')..."

#### C. Settings Modal (Global Config)
*Hidden behind a 'Gear' icon.*
- **Ranking Weights:**
  - "Balanced Profile": Slider [Performance <-> Cost] (Default: 50/50).
  - "Budget Profile": Slider [Performance <-> Cost] (Default: 20/80).

### 3.3. Results Interface

#### A. The "Traceability" View (Live Logs; optional)
*Displays the Agent's "Thought Process" (JSON stream `traceability.events[]`) to build trust.*
- **State:** Collapsible sidebar or bottom drawer.
- **Content:**
  - `> Analyzing intent... (Estimated Ratio: 95% Input / 5% Output)`
  - `> Searching benchmarks: "Legal reasoning", "Long context retrieval"...`
  - `> Found 3 relevant benchmarks: LexBench (0.9), RAG-QA (0.8)...`
  - `> Filtering: 12 models matched "Local Only" constraint.`
  - `> Calibrating scores (Using offset logic for 2 models)...`

#### B. Structured Response (The "Chat" Bubble)
*Renders the `ui_components` from the Agent's JSON response.*

1.  **Executive Summary:** Markdown text explaining the top pick.
2.  **Comparison Table (Interactive):**
    - **Columns:** Model Name | Rank Score | Cost/1M | Key Benchmarks (e.g., HumanEval) | Est?
    - **Interactivity:** Click column headers to sort.
    - **Tooltip:** Hovering over "Est?" shows "Score inferred via <Modelname> bridge."
3.  **Recommendation Cards:**
    - Visual highlighting of the "Winner" for each category (Performance vs Budget).
    - **Actions:** "Copy Name", "View Full Details" (Expands row).

### 3.4. Edge Case Handling
- **No Models Found:** If constraints are too strict (e.g., "Video Input" + "Local Only" + "Reasoning Model"), the UI must show a "Relax Constraints" prompt.
- **Missing Data:** If a model row has `N/A` for a benchmark, it is grayed out but visible, with a tooltip explaining "No verified data available."