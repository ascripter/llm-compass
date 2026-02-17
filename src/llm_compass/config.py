"""
Global configuration settings.

Loads environment variables (DB credentials, API keys) and defines
application constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

# Database Connection String (PostgreSQL + pgvector)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/llm_benchmarks")

# LLM Provider Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# App Settings
MVP_BENCHMARK_SOURCES = ["seed_data/benchmarks.csv", "seed_data/scores.csv"]
