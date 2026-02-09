# Project Idea

## Target Users

AI developers building agentic systems

## Concept

Another question that's been bothering me during the last weeks and that was asked in many sessions: For which task is which LLM *currently* best suited, and how to keep track of the rapid changes in the field?

The goal is to have a chat client answering questions like "what LLM is currently best suited for a web research agent" or "which models can describe video content" etc.

To my experience, ungrounded LLMs are biased towards replying with "famous" frontier models. It requires detailled querying to get an agent like Perplexity or ChatGPT to even *consider* other models and evaluate them, and then their research rarely seems sufficient (w/o again querying deeper).

### MVP Outline: "Smart Agent" Approach

Main tasks (as perceived by Gemini Pro 3):

**A. Data Foundation**
  - Benchmark Dictionary (Vector Store)
    - Store pairs of `Benchmark Name` + `Rich Description`.
    - *Key:* Descriptions must explicitly state what the benchmark measures (e.g., "MMLU tests general world knowledge") to allow semantic search.
  - Score Repository (Simple DB)
    - Flat table structure: `Model Name` | `Benchmark` | `Score` | `Date`.
  - LLM-Assisted Ingestion
    - Scraper fetches raw data.
    - "Write-time" Normalization: Use a cheap LLM to standardize model names (e.g., "llama-2-7b-chat" → "Llama 2 7B") before saving to DB.

**B. Agentic Core**
  - System Prompt
    - Define Persona: "AI Model Consultant."
    - Define Workflow: Plan strategy → Find benchmarks → Get scores → Recommend.
  - Tools
    - `find_relevant_benchmarks(query)`: Performs semantic search over the Benchmark Dictionary to find relevant metrics based on user intent.
    - `get_scores(benchmark_names)`: SQL/Pandas lookup to retrieve numerical data for specific benchmarks.
  - Reasoning Loop (LangGraph/CrewAI)
    - Node 1 (Strategy): LLM parses user query and decides which benchmarks are relevant.
    - Node 2 (Retrieval): Agent executes tools to get definitions and scores.
    - Node 3 (Synthesis): LLM analyzes tradeoffs (accuracy vs. cost) and generates the final answer.

**C. Minimal Interface**
  - Chat UI (Streamlit/Gradio)
    - Simple text input.
    - Traceability View: Display the agent's intermediate steps (e.g., "Searching for 'reasoning' benchmarks...", "Found: GSM8k, MATH") to demonstrate the "Agentic" aspect.


