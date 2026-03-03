# LLM Compass — FastAPI Design Document

> **Version:** 1.0 · **Date:** 2026-02-25  
> **Purpose:** Define a production-ready REST + WebSocket API layer (FastAPI) that any frontend (React, Vue, Next.js, mobile, etc.) can integrate with, replacing/complementing the current Streamlit UI.

---

## Table of Contents

1. [Design Goals](#1-design-goals)
2. [Tech Stack](#2-tech-stack)
3. [Project Structure](#3-project-structure)
4. [Pydantic Schemas (Request / Response)](#4-pydantic-schemas-request--response)
5. [API Endpoints](#5-api-endpoints)
   - 5.1 [Health & Meta](#51-health--meta)
   - 5.2 [Chat / Agent Query](#52-chat--agent-query-core)
   - 5.3 [Models (LLM Metadata)](#53-models-llm-metadata)
   - 5.4 [Benchmarks](#54-benchmarks)
   - 5.5 [Data Ingestion (Admin)](#55-data-ingestion-admin)
   - 5.6 [Sessions & History](#56-sessions--history)
6. [WebSocket — Real-Time Traceability](#6-websocket--real-time-traceability)
7. [Authentication & Authorization](#7-authentication--authorization)
8. [Error Handling](#8-error-handling)
9. [CORS & Frontend Integration](#9-cors--frontend-integration)
10. [Database Dependency Injection](#10-database-dependency-injection)
11. [Startup & Shutdown Lifecycle](#11-startup--shutdown-lifecycle)
12. [Sequence Diagrams](#12-sequence-diagrams)
13. [OpenAPI / Swagger Notes](#13-openapi--swagger-notes)
14. [Migration Path from Streamlit](#14-migration-path-from-streamlit)

---

## 1. Design Goals

| # | Goal | Rationale |
|---|------|-----------|
| G1 | **Stateless REST** for all CRUD + query operations | Enables horizontal scaling and simple frontend integration. |
| G2 | **WebSocket** for real-time agent traceability streaming | Frontend can render live "Agent Thought Process" logs (Req 3.3.A). |
| G3 | **Pydantic v2 schemas** as the single source of truth | Type-safe contracts shared between API, agent core, and frontend (via generated OpenAPI). |
| G4 | **Session-based context** (server-side) | Support multi-turn clarification flows (Req 2.3 Node 1 — Clarification Gate). |
| G5 | **Admin endpoints** gated by API key | Protect data ingestion & mutation routes. |
| G6 | **Auto-generated OpenAPI 3.1 spec** | Frontend teams can codegen TypeScript types from `/docs`. |

---

## 2. Tech Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Framework | FastAPI | ≥ 0.128 |
| Server | Uvicorn (bundled via `fastapi[standard]`) | — |
| Validation | Pydantic v2 | ≥ 2.8 |
| ORM | SQLAlchemy 2.0 (async optional) | ≥ 2.0 |
| Database | SQLite + FAISS index | — |
| Agent | LangGraph | ≥ 1.0 |
| HTTP Client | httpx (async) | ≥ 0.27 |
| Auth | API-Key header (MVP) → OAuth2/JWT (post-MVP) | — |

> **Note:** All of the above are already in `requirements.txt`.

---

## 3. Project Structure

```
src/llm_compass/
├── api/                          # ← NEW: FastAPI layer
│   ├── __init__.py
│   ├── main.py                   # FastAPI app factory, lifespan, CORS
│   ├── deps.py                   # Dependency injection (DB session, auth, etc.)
│   ├── schemas/                  # Pydantic request/response models
│   │   ├── __init__.py
│   │   ├── common.py             # Shared enums, pagination, errors
│   │   ├── query.py              # QueryRequest, QueryResponse, Clarification
│   │   ├── models.py             # LLMMetadataOut, LLMMetadataFilter
│   │   ├── benchmarks.py         # BenchmarkOut, BenchmarkSearch
│   │   └── ingestion.py          # CSVUploadResponse, IngestionStatus
│   ├── routers/                  # Route modules (one per domain)
│   │   ├── __init__.py
│   │   ├── health.py             # GET /health, GET /info
│   │   ├── query.py              # POST /query, POST /query/{id}/clarify
│   │   ├── models.py             # GET /models, GET /models/{id}
│   │   ├── benchmarks.py         # GET /benchmarks, GET /benchmarks/{id}
│   │   ├── ingestion.py          # POST /admin/ingest/csv
│   │   └── sessions.py           # GET /sessions/{id}
│   └── ws/                       # WebSocket handlers
│       ├── __init__.py
│       └── trace.py              # WS /ws/trace/{session_id}
├── agentic_core/                 # Existing — no changes needed
├── data/                         # Existing — no changes needed
├── ui/                           # Existing Streamlit (can coexist)
├── config.py                     # Existing
└── app.py                        # Existing entry (updated for FastAPI)
```

---

## 4. Pydantic Schemas (Request / Response)

### 4.1 Common

```python
# api/schemas/common.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List

class Modality(str, Enum):
    TEXT  = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class SpeedClass(str, Enum):
    FAST   = "fast"
    MEDIUM = "medium"
    SLOW   = "slow"

class DeploymentType(str, Enum):
    ANY   = "any"
    CLOUD = "cloud"
    LOCAL = "local"

class Pagination(BaseModel):
    page:     int = Field(1, ge=1)
    per_page: int = Field(20, ge=1, le=100)

class PaginatedMeta(BaseModel):
    total:    int
    page:     int
    per_page: int
    pages:    int

class ErrorDetail(BaseModel):
    code:    str
    message: str

class APIError(BaseModel):
    errors: List[ErrorDetail]
```

### 4.2 Query (Agent Interaction)

```python
# api/schemas/query.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

class Constraints(BaseModel):
    """Maps 1:1 to UI constraint panel (Req 3.2)."""
    min_context_window:  int                  = Field(0, ge=0)
    modality_input:      List[Modality]       = ["text"]
    modality_output:     List[Modality]       = ["text"]
    deployment:          DeploymentType        = DeploymentType.ANY
    require_reasoning:   bool                  = False
    require_tool_calling: bool                 = False
    min_speed_class:     Optional[SpeedClass]  = None

class QueryRequest(BaseModel):
    """Primary request body for the /query endpoint."""
    user_query:   str         = Field(..., min_length=5, max_length=2000,
                                       examples=["I need a model for RAG on legal documents"])
    constraints:  Constraints = Constraints()
    session_id:   Optional[str] = None  # Omit for new session

class ClarifyRequest(BaseModel):
    """Follow-up message when the agent asks for clarification."""
    user_reply: str = Field(..., min_length=1, max_length=2000)

# --- Response Sub-Models ---

class TraceEvent(BaseModel):
    stage:   str
    message: str
    data:    Dict[str, Any] = {}

class BenchmarkResult(BaseModel):
    benchmark_id:      str
    benchmark_name:    str
    benchmark_variant: Optional[str] = None
    score:             float
    metric_unit:       str
    weight_used:       float
    is_estimated:      bool          = False
    source_url:        Optional[str] = None
    estimation_note:   Optional[str] = None

class RankMetrics(BaseModel):
    performance_index:   float
    blended_cost_index:  float
    blended_score:       float

class RankedModel(BaseModel):
    model_id:            str
    name_normalized:     str
    provider:            str
    speed_class:         Optional[SpeedClass] = None
    speed_tps:           Optional[float]      = None
    rank_metrics:        RankMetrics
    benchmark_results:   List[BenchmarkResult]
    reason_for_ranking:  str

class ComparisonTable(BaseModel):
    title:   str
    columns: List[str]
    rows:    List[List[Any]]

class RecommendationCard(BaseModel):
    category:   str          # "Top Performance", "Balanced", "Budget Pick"
    model_name: str
    reason:     str

class Citation(BaseModel):
    id:    str
    label: str
    url:   str

class Warning(BaseModel):
    code:    str
    message: str

class UIComponents(BaseModel):
    summary_markdown:     str
    comparison_table:     Optional[ComparisonTable]          = None
    recommendation_cards: List[RecommendationCard]           = []
    citations:            List[Citation]                     = []
    warnings:             List[Warning]                      = []

class RankedLists(BaseModel):
    top_performance: List[RankedModel]
    balanced:        List[RankedModel]
    budget:          List[RankedModel]
    metadata:        Dict[str, Any]     = {}

class QueryResponse(BaseModel):
    """Matches the AgentResponse JSON schema from Req 2.1.B."""
    session_id:          str
    user_query:          str
    applied_constraints: Dict[str, Any]
    status:              Literal["ok", "needs_clarification", "error"]
    clarification_question: Optional[str] = None
    traceability:        Dict[str, List[TraceEvent]] = {"events": []}
    ranked_data:         Optional[RankedLists]       = None
    ui_components:       Optional[UIComponents]      = None
    errors:              List[ErrorDetail]            = []
```

### 4.3 Models (LLM Metadata)

```python
# api/schemas/models.py
class LLMMetadataOut(BaseModel):
    id:                  int
    name_normalized:     str
    provider:            str
    parameter_count:     Optional[int]
    architecture:        Optional[str]
    quantization:        Optional[str]
    modality_input:      List[str]
    modality_output:     List[str]
    context_window:      int
    cost_input_1m:       float
    cost_output_1m:      float
    speed_class:         str
    speed_tps:           Optional[float]
    is_open_weights:     bool
    is_reasoning_model:  bool
    has_tool_calling:    bool
    is_outdated:         bool

class LLMMetadataFilter(BaseModel):
    """Query params for GET /models."""
    provider:         Optional[str]       = None
    is_open_weights:  Optional[bool]      = None
    min_context:      Optional[int]       = None
    modality_input:   Optional[List[str]] = None
    speed_class:      Optional[str]       = None
    is_outdated:      bool                = False

class LLMListResponse(BaseModel):
    data: List[LLMMetadataOut]
    meta: PaginatedMeta
```

### 4.4 Benchmarks

```python
# api/schemas/benchmarks.py
class BenchmarkOut(BaseModel):
    id:              int
    name_normalized: str
    variant:         Optional[str]
    description:     str
    categories:      List[str]

class BenchmarkListResponse(BaseModel):
    data: List[BenchmarkOut]
    meta: PaginatedMeta

class BenchmarkScoreOut(BaseModel):
    model_name:     str
    score_value:    float
    metric_unit:    Optional[str]
    source_name:    Optional[str]
    source_url:     Optional[str]
    date_published: Optional[str]
```

### 4.5 Ingestion

```python
# api/schemas/ingestion.py
class CSVUploadResponse(BaseModel):
    status:           str   # "success" | "partial" | "error"
    rows_processed:   int
    rows_failed:      int
    unmatched_entities: List[str]  # entities that couldn't be normalized
    message:          str
```

---

## 5. API Endpoints

### 5.1 Health & Meta

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | None | Returns `{"status": "ok", "db": "connected"}`. Liveness probe. |
| `GET` | `/info` | None | Returns app version, DB schema version, number of models/benchmarks. |

### 5.2 Chat / Agent Query (Core)

| Method | Path | Auth | Request Body | Response |
|--------|------|------|--------------|----------|
| `POST` | `/api/v1/query` | API Key | `QueryRequest` | `QueryResponse` |
| `POST` | `/api/v1/query/{session_id}/clarify` | API Key | `ClarifyRequest` | `QueryResponse` |

**Flow:**

```
Frontend                          FastAPI                        LangGraph
   │                                 │                               │
   │─── POST /api/v1/query ────────►│                               │
   │    { user_query, constraints }  │                               │
   │                                 │── build_graph().invoke() ───►│
   │                                 │                               │── Node 1: validate_intent
   │                                 │                               │── Router: clarification?
   │                                 │                               │     YES → return early
   │                                 │                               │     NO  → Node 2–5
   │                                 │◄── AgentState ───────────────│
   │◄── QueryResponse ──────────────│                               │
   │    (status: "ok" or                                            │
   │     "needs_clarification")                                     │
   │                                 │                               │
   │  [If needs_clarification]       │                               │
   │─── POST /query/{sid}/clarify ──►│── re-invoke graph ──────────►│
   │    { user_reply }               │                               │
   │◄── QueryResponse (status: ok) ─│                               │
```

**Key Implementation Details:**

```python
# api/routers/query.py (pseudocode)
from fastapi import APIRouter, Depends
from ..schemas.query import QueryRequest, QueryResponse, ClarifyRequest
from ..deps import get_db, require_api_key
from ...agentic_core.graph import build_graph
import uuid

router = APIRouter(prefix="/api/v1", tags=["Query"])

# In-memory session store (MVP) → Redis (prod)
_sessions: dict[str, dict] = {}

@router.post("/query", response_model=QueryResponse)
async def create_query(req: QueryRequest, db=Depends(get_db)):
    session_id = req.session_id or str(uuid.uuid4())
    graph = build_graph()

    initial_state = {
        "user_query": req.user_query,
        "constraints": req.constraints.model_dump(),
        "clarification_needed": False,
        "clarification_question": None,
        "predicted_io_ratio": {},
        "search_queries": [],
        "ranked_results": {},
        "final_response": None,
        "logs": [],
    }

    result = graph.invoke(initial_state)

    # Persist session for potential clarification follow-ups
    _sessions[session_id] = result

    return _build_response(session_id, result)


@router.post("/query/{session_id}/clarify", response_model=QueryResponse)
async def clarify_query(session_id: str, req: ClarifyRequest, db=Depends(get_db)):
    prev_state = _sessions.get(session_id)
    if not prev_state:
        raise HTTPException(404, "Session not found")

    # Append clarification to the original query
    prev_state["user_query"] += f"\n[Clarification]: {req.user_reply}"
    prev_state["clarification_needed"] = False

    graph = build_graph()
    result = graph.invoke(prev_state)
    _sessions[session_id] = result

    return _build_response(session_id, result)
```

### 5.3 Models (LLM Metadata)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/models` | None | List models with filtering & pagination. |
| `GET` | `/api/v1/models/{model_id}` | None | Single model details + its benchmark scores. |

**Query Parameters for `GET /models`:**

| Param | Type | Default | Example |
|-------|------|---------|---------|
| `provider` | string | — | `OpenAI` |
| `is_open_weights` | bool | — | `true` |
| `min_context` | int | 0 | `128000` |
| `modality_input` | string[] | — | `text,image` |
| `speed_class` | string | — | `fast` |
| `is_outdated` | bool | `false` | `false` |
| `page` | int | 1 | `2` |
| `per_page` | int | 20 | `50` |
| `sort_by` | string | `name_normalized` | `cost_input_1m` |
| `sort_order` | string | `asc` | `desc` |

**Example Response:**
```json
{
  "data": [
    {
      "id": 1,
      "name_normalized": "GPT-4o",
      "provider": "OpenAI",
      "parameter_count": null,
      "context_window": 128000,
      "cost_input_1m": 2.50,
      "cost_output_1m": 10.00,
      "speed_class": "fast",
      "is_open_weights": false,
      "is_reasoning_model": false,
      "has_tool_calling": true,
      "is_outdated": false,
      "modality_input": ["text", "image"],
      "modality_output": ["text"]
    }
  ],
  "meta": { "total": 42, "page": 1, "per_page": 20, "pages": 3 }
}
```

### 5.4 Benchmarks

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/benchmarks` | None | List all benchmarks (filterable by `category`). |
| `GET` | `/api/v1/benchmarks/{id}` | None | Single benchmark + all model scores for it. |
| `GET` | `/api/v1/benchmarks/{id}/scores` | None | Paginated scores for a specific benchmark. |

### 5.5 Data Ingestion (Admin)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/v1/admin/ingest/csv` | Admin API Key | Upload CSV file for manual data import. |
| `POST` | `/api/v1/admin/ingest/trigger` | Admin API Key | Trigger scheduled aggregation manually. |
| `GET`  | `/api/v1/admin/ingest/status` | Admin API Key | Check status of last ingestion run. |

**CSV Upload:**

```python
# api/routers/ingestion.py (pseudocode)
from fastapi import APIRouter, UploadFile, File, Depends
from ..deps import require_admin_key

router = APIRouter(prefix="/api/v1/admin", tags=["Admin — Ingestion"])

@router.post("/ingest/csv", response_model=CSVUploadResponse)
async def upload_csv(
    file: UploadFile = File(...),
    entity_type: str = Query(..., regex="^(scores|models|benchmarks)$"),
    _=Depends(require_admin_key),
):
    # Calls data.ingestion.import_manual_csv()
    ...
```

### 5.6 Sessions & History

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/sessions/{session_id}` | API Key | Retrieve full session state (query, constraints, results, trace). |
| `DELETE` | `/api/v1/sessions/{session_id}` | API Key | Delete a session. |

---

## 6. WebSocket — Real-Time Traceability

**Endpoint:** `WS /ws/trace/{session_id}`

Streams `TraceEvent` objects in real-time as the LangGraph executes each node, enabling the frontend to render the "Agent Thought Process" panel (Req 3.3.A).

**Protocol:**

```
Frontend                          FastAPI WS Handler
   │                                 │
   │── WS Connect ──────────────────►│
   │   /ws/trace/{session_id}        │
   │                                 │
   │◄── {"stage": "validator",      │ ◄── graph callback
   │      "message": "Analyzing      │
   │       intent..."}               │
   │                                 │
   │◄── {"stage": "refiner",        │
   │      "message": "Predicted I/O  │
   │       ratio: 90/10"}           │
   │                                 │
   │◄── {"stage": "discovery",      │
   │      "message": "Found 3        │
   │       benchmarks..."}          │
   │                                 │
   │◄── {"stage": "ranking",        │
   │      "message": "Ranking 12     │
   │       models..."}              │
   │                                 │
   │◄── {"stage": "synthesis",      │
   │      "message": "Generating     │
   │       final response..."}      │
   │                                 │
   │◄── {"stage": "done",           │
   │      "message": "Complete"}    │
   │                                 │
   │── WS Close ────────────────────►│
```

**Implementation Sketch:**

```python
# api/ws/trace.py
from fastapi import WebSocket, WebSocketDisconnect

# Global registry: session_id → list of connected WebSocket clients
_trace_clients: dict[str, list[WebSocket]] = {}

async def trace_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    _trace_clients.setdefault(session_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        _trace_clients[session_id].remove(websocket)

async def broadcast_trace(session_id: str, event: dict):
    """Called from within LangGraph node callbacks."""
    for ws in _trace_clients.get(session_id, []):
        await ws.send_json(event)
```

---

## 7. Authentication & Authorization

### MVP: API Key

```
Header: X-API-Key: <key>
```

| Scope | Endpoints | Key Type |
|-------|-----------|----------|
| Public | `/health`, `/info`, `/api/v1/models`, `/api/v1/benchmarks` | None |
| User | `/api/v1/query/**`, `/api/v1/sessions/**`, `/ws/trace/**` | User API Key |
| Admin | `/api/v1/admin/**` | Admin API Key |

```python
# api/deps.py
from fastapi import Header, HTTPException
from ..config import API_KEYS, ADMIN_API_KEYS

async def require_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(401, "Invalid API key")

async def require_admin_key(x_api_key: str = Header(...)):
    if x_api_key not in ADMIN_API_KEYS:
        raise HTTPException(403, "Admin access required")
```

### Post-MVP: OAuth2 + JWT

- Replace API key with bearer tokens.
- Add user accounts, rate limiting, and RBAC.

---

## 8. Error Handling

All errors follow a consistent `APIError` schema:

```json
{
  "errors": [
    {
      "code": "VALIDATION_ERROR",
      "message": "user_query must be at least 5 characters."
    }
  ]
}
```

| HTTP Code | Code Constant | When |
|-----------|---------------|------|
| 400 | `VALIDATION_ERROR` | Pydantic validation fails. |
| 401 | `UNAUTHORIZED` | Missing or invalid API key. |
| 403 | `FORBIDDEN` | Non-admin accessing admin route. |
| 404 | `NOT_FOUND` | Model, Benchmark, or Session ID doesn't exist. |
| 409 | `CONFLICT` | Duplicate entity during ingestion. |
| 422 | `UNPROCESSABLE_ENTITY` | FastAPI default for request body errors. |
| 500 | `INTERNAL_ERROR` | Unhandled exception. |
| 503 | `SERVICE_UNAVAILABLE` | DB or LLM provider unreachable. |

**Global Exception Handler:**

```python
# api/main.py
@app.exception_handler(Exception)
async def global_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"errors": [{"code": "INTERNAL_ERROR", "message": str(exc)}]},
    )
```

---

## 9. CORS & Frontend Integration

```python
# api/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Frontend TypeScript codegen** (recommended):

```bash
# Generate TS types from the running API
npx openapi-typescript http://localhost:8000/openapi.json -o src/api/types.ts
```

---

## 10. Database Dependency Injection

```python
# api/deps.py
from ..data.database import SessionLocal

async def get_db():
    """Yields a SQLAlchemy session, auto-closes after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Usage in any router:

```python
from fastapi import Depends
from ..deps import get_db

@router.get("/models")
async def list_models(db=Depends(get_db)):
    ...
```

---

## 11. Startup & Shutdown Lifecycle

```python
# api/main.py
from contextlib import asynccontextmanager
from ..data.database import init_db, engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("🚀 Initializing database tables...")
    init_db()
    print("✅ LLM Compass API ready.")
    yield
    # --- SHUTDOWN ---
    engine.dispose()
    print("🛑 Database connections closed.")

app = FastAPI(
    title="LLM Compass API",
    version="1.0.0",
    description="Benchmark-driven LLM recommendation engine.",
    lifespan=lifespan,
)
```

---

## 12. Sequence Diagrams

### 12.1 Happy Path — Full Query

```
┌──────────┐      ┌───────────┐      ┌──────────┐      ┌──────────┐
│ Frontend  │      │  FastAPI   │      │ LangGraph │      │ SQLite │
└────┬─────┘      └─────┬─────┘      └─────┬────┘      └─────┬────┘
     │  POST /query      │                  │                  │
     │──────────────────►│                  │                  │
     │                   │  graph.invoke()  │                  │
     │                   │─────────────────►│                  │
     │                   │                  │─ validate_intent │
     │                   │                  │─ refine_query    │
     │                   │                  │─ find_benchmarks─┼──► Vector Search
     │                   │                  │◄─────────────────┼──  Results
     │                   │                  │─ rank_models ────┼──► SQL Filter + Score
     │                   │                  │◄─────────────────┼──  Ranked Lists
     │                   │                  │─ synthesize      │
     │                   │◄─────────────────│  AgentState      │
     │  QueryResponse    │                  │                  │
     │◄──────────────────│                  │                  │
     │  (status: "ok")   │                  │                  │
```

### 12.2 Clarification Flow

```
┌──────────┐      ┌───────────┐      ┌──────────┐
│ Frontend  │      │  FastAPI   │      │ LangGraph │
└────┬─────┘      └─────┬─────┘      └─────┬────┘
     │  POST /query      │                  │
     │──────────────────►│  graph.invoke()  │
     │                   │─────────────────►│
     │                   │                  │─ validate_intent
     │                   │                  │  → needs_clarification!
     │                   │◄─────────────────│  (early return at Router)
     │  QueryResponse    │                  │
     │◄──────────────────│                  │
     │  status: "needs_clarification"       │
     │  clarification_question: "..."       │
     │                   │                  │
     │  [User provides more detail]         │
     │  POST /query/{sid}/clarify           │
     │──────────────────►│  graph.invoke()  │
     │                   │─────────────────►│
     │                   │                  │─ validate → refine → ... → synthesize
     │                   │◄─────────────────│
     │  QueryResponse    │                  │
     │◄──────────────────│                  │
     │  status: "ok"     │                  │
```

### 12.3 Real-Time Traceability (WebSocket)

```
┌──────────┐      ┌───────────┐
│ Frontend  │      │  FastAPI   │
└────┬─────┘      └─────┬─────┘
     │  WS /ws/trace/sid │
     │══════════════════►│  (connection opened)
     │                   │
     │  POST /query      │  (concurrent HTTP request)
     │──────────────────►│
     │                   │─── graph starts ───
     │◄─ TraceEvent ─────│  {"stage":"validator","message":"..."}
     │◄─ TraceEvent ─────│  {"stage":"refiner","message":"..."}
     │◄─ TraceEvent ─────│  {"stage":"discovery","message":"..."}
     │◄─ TraceEvent ─────│  {"stage":"ranking","message":"..."}
     │◄─ TraceEvent ─────│  {"stage":"synthesis","message":"..."}
     │◄─ TraceEvent ─────│  {"stage":"done","message":"Complete"}
     │                   │
     │  QueryResponse    │  (HTTP response arrives)
     │◄──────────────────│
```

---

## 13. OpenAPI / Swagger Notes

FastAPI auto-generates an OpenAPI 3.1 spec at:

| URL | Description |
|-----|-------------|
| `http://localhost:8000/docs` | Swagger UI (interactive) |
| `http://localhost:8000/redoc` | ReDoc (documentation) |
| `http://localhost:8000/openapi.json` | Raw JSON spec (for codegen) |

**Tags used in the spec:**

| Tag | Routers |
|-----|---------|
| `Health` | `/health`, `/info` |
| `Query` | `/api/v1/query/**` |
| `Models` | `/api/v1/models/**` |
| `Benchmarks` | `/api/v1/benchmarks/**` |
| `Admin — Ingestion` | `/api/v1/admin/ingest/**` |
| `Sessions` | `/api/v1/sessions/**` |

---

## 14. Migration Path from Streamlit

The existing Streamlit UI (`src/llm_compass/ui/`) can coexist with FastAPI during migration:

| Phase | Action |
|-------|--------|
| **Phase 1** | Deploy FastAPI alongside Streamlit. Both share the same `agentic_core` and `data` packages. |
| **Phase 2** | Build a React/Next.js frontend consuming the FastAPI endpoints. |
| **Phase 3** | Retire Streamlit UI once the new frontend reaches feature parity. |

**Running both (development):**

```bash
# Terminal 1 — FastAPI
uvicorn src.llm_compass.api.main:app --reload --port 8000

# Terminal 2 — Streamlit (legacy)
streamlit run src/llm_compass/ui/app.py --server.port 8501
```

---

## Appendix: Quick-Start Checklist

- [ ] Create `src/llm_compass/api/` directory structure (Section 3).
- [ ] Implement Pydantic schemas (Section 4).
- [ ] Wire up `api/main.py` with lifespan, CORS, and exception handlers.
- [ ] Implement `/health` and `/api/v1/query` routers first.
- [ ] Add WebSocket `/ws/trace` for traceability streaming.
- [ ] Add API key auth via `deps.py`.
- [ ] Generate OpenAPI spec → share with frontend team.
- [ ] Add integration tests under `tests/test_api/`.

---

*This document is the single source of truth for the FastAPI integration layer. Update it as endpoints evolve.*