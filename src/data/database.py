"""
Database connection and session management.
Req 1.2: Central access point for PostgreSQL + pgvector.
"""

import os
from sqlmodel import create_engine, Session
from sqlalchemy.orm import sessionmaker

# Database URL from environment variable
# e.g. postgresql://user:pass@localhost:5432/benchmarks_db
DATABASE_URL = os.getenv("DATABASE_URL")

# Create engine with echo=True for debugging during MVP
engine = create_engine(DATABASE_URL, echo=False)


def get_session():
    """
    Dependency for getting a database session.
    Usage:
        with get_session() as session:
            ...
    """
    with Session(engine) as session:
        yield session


def init_db():
    """
    Creates tables if they don't exist.
    Run this on startup.
    """
    from .models import SQLModel

    SQLModel.metadata.create_all(engine)
