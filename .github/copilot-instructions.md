# LLM Compass - AI Coding Agent Guidelines

## Project Overview
An AI-powered LLM recommendation system that answers "Which LLM is best for task X under constraints Y?" using stored benchmark data, metadata, and source links. Built with FastAPI, LangGraph for agentic orchestration, SQLAlchemy + SQLite/FAISS for data, and Streamlit for UI.

## Architecture

### Component Structure
- **`src/llm_compass/agentic_core/`** - LangGraph workflow for query processing
  - `graph.py` - Graph assembly with nodes: validator → refiner → discovery → ranking → synthesis
  - `nodes/` - Individual node implementations
  - `tools.py` - Vector search and model ranking tools
  - `state.py` - AgentState TypedDict for graph state management
  - `schemas/logic.py` - Pydantic schemas for internal tool outputs
- **`src/llm_compass/api/`** - FastAPI endpoints
  - `main.py` - App factory with CORS and error handlers
  - `routers/` - Route handlers (health, query)
  - `schemas/` - Pydantic request/response models
- **`src/llm_compass/data/`** - Data layer
  - `models.py` - SQLAlchemy ORM
  - `database.py` - Connection management
  - `embedding.py` - FAISS vector search for benchmark discovery
  - `ingestion.py` - CSV/data import pipeline
  - `normalizer.py` - Entity name normalization
- **`src/llm_compass/ui/`** - Streamlit frontend
  - `app.py` - Main entry with sidebar/chat layout
  - `components/` - UI widgets (chat, tables, traceability)

### Key Data Models
Three core entities in `data/models.py`:
- `BenchmarkDictionary` - Semantic benchmark definitions with FAISS index IDs
- `LLMMetadata` - Static model attributes (modality, cost, speed, reasoning type)
- `BenchmarkScore` - Performance scores with audit trail (original names preserved)

## Code Style

### Type Hints
- Use `typing` module extensively: `List`, `Optional`, `Literal`, `Mapped`
- SQLAlchemy 2.0 style: `Mapped[str] = mapped_column(String, nullable=False)`
- Pydantic v2 for validation schemas (see `models.py` LLMMetadataSchema)

### Naming Conventions
- SQLAlchemy tables: snake_case (e.g., `benchmark_dictionary`)
- Normalized names: kebab-case lowercase (e.g., `llama-3.1-70b-instruct`)
- Aliases stored as lists in JSON columns
- Pydantic validators: `@field_validator` with `mode="before"`

### Patterns from Codebase
```python
# Comma-separated list validation pattern (models.py:30)
def _comma_separated_list_validator(v: Any, allowed: Optional[tuple[str]] = None) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [item.strip() for item in v.split(",") if item.strip()]
    # ...

# Boolean CSV parsing (models.py field_validator)
@staticmethod
def validate_bool_fields(v):
    if isinstance(v, str):
        v = v.strip().upper()
        if v in ("TRUE", "1", "YES", "T", "Y"):
            return True
        # ...
```

## Build and Test

### Environment Setup
```bash
# Copy environment template
copy .env.example .env  # Windows
# Edit .env with: OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_COMPASS_STORAGE_PATH

# Install dependencies
pip install -r requirements.txt
```

### Build Database
```bash
# Windows
python src\llm_compass\scripts\build_data.py

# Linux/Mac
python src/llm_compass/scripts/build_data.py
```

### Run Tests
```bash
pytest tests/
```

### Run Services
```bash
# API server
python -m uvicorn src.llm_compass.api.main:app --reload --port 8000

# Streamlit UI
streamlit run src/llm_compass/ui/app.py
```

## Project Conventions

### Data Ingestion
1. **Preserve originals**: Always store `original_model_name` and `original_benchmark_name` in `BenchmarkScore` for audit
2. **Normalization at write-time**: Use `normalizer.py` to map incoming strings to standardized entities
3. **FK resolution**: Match scores to `LLMMetadata` and `BenchmarkDictionary` by normalized names

### Agentic Core Rules (from product_requirements.md)
- **No guessing**: If data unavailable, return "Insufficient Data"
- **Strict citation**: Every claim needs benchmark reference with `source_url`
- **Mark estimates**: Tag inferred scores with `is_estimated: true` and `estimation_note`
- **Output format**: Always valid JSON matching `AgentResponse` schema

### Ranking Algorithm
Three-view ranking in `tools.py`:
- `top_performance` - Pure performance index
- `balanced` - 50% performance + 50% cost index
- `budget` - 20% performance + 80% cost index

Cost index computed per-query: `Blended_Cost_Index = (C_max - Blended_Cost_1M) / (C_max - C_min)`

### Vector Search (FAISS)
- Benchmark descriptions embedded and stored in `storage/benchmark_descriptions.faiss`
- `find_relevant_benchmarks()` uses multi-query aggregation with relevance cutoff (default 0.7)
- Index IDs map to `BenchmarkDictionary.id`

## Integration Points

### External APIs
- **OpenRouter** - Primary LLM provider for agent inference
  - Configured via `OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL` env vars
  - Used in agent nodes for intent validation, query refinement, synthesis

### Database
- SQLite for all environments (`sqlite://` for tests, file-based for prod)
- FAISS for vector search (CPU version)

### Data Sources
- Manual CSV import (MVP primary)
- Curated aggregators: vals.ai, artificialanalysis.ai, llm-stats.com (respect robots.txt)
- Research agent discovery (post-MVP)

## Security Considerations
- **No hardcoded secrets**: Use `config.py` Settings dataclass with env var loading
- **Input validation**: All API inputs use Pydantic schemas with validators
- **SQL injection protection**: SQLAlchemy ORM with parameterized queries only
- **CORS**: Restricted to `localhost:3000` and production domain in `api/main.py`

## Testing Strategy
- **Unit tests**: `tests/test_data/`, `tests/test_core/` - Use in-memory SQLite fixtures
- **Integration tests**: Full graph execution with mocked LLM calls
- **Fixture pattern**: `conftest.py` provides `session_fixture()` for DB tests

## Common Tasks

### Adding a New Benchmark
1. Add entry to `research/benchmarks.csv` or use admin CSV upload
2. Run `build_data.py` to regenerate FAISS index
3. Verify vector search returns benchmark with `find_relevant_benchmarks()`

### Adding Model Metadata
1. Edit `research/llm_metadata_cleaned.json`
2. Run `build_data.py` to load into `LLMMetadata` table
3. Ingest scores linking to normalized model name

### Extending the Agent Graph
1. Add node function in `agentic_core/nodes/`
2. Register in `graph.py` with `workflow.add_node()`
3. Define edges with `workflow.add_edge()` or conditional edges
4. Update `state.py` AgentState if new fields needed

## Key Files Reference
- **Data models**: `src/llm_compass/data/models.py`
- **Graph assembly**: `src/llm_compass/agentic_core/graph.py`
- **Config/Env**: `src/llm_compass/config.py`
- **API routes**: `src/llm_compass/api/routers/query.py`
- **Product requirements**: `product_requirements.md`
- **Test fixtures**: `tests/conftest.py`
