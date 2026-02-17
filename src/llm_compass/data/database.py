"""
Database connection and session management.
Req 1.2: Central access point for PostgreSQL + pgvector.
"""

import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from pgvector.psycopg import register_vector

# Database URL from environment variable
# e.g. postgresql://user:pass@localhost:5432/database_name
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL environment variable is not set")

# Create engine with echo=True for debugging during MVP
engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@event.listens_for(engine, "connect")
def register_vector_type(dbapi_connection, connection_record):
    register_vector(dbapi_connection)


def get_session():
    """
    Dependency for getting a database session.
    Usage:
        with get_session() as session:
            ...
    """
    with SessionLocal() as session:
        yield session


def init_db():
    """
    Creates tables if they don't exist.
    Run this on startup.
    """
    from .models import Base

    Base.metadata.create_all(engine)
