"""
Pytest fixtures for integration testing.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool


def pytest_configure(config):
    """Configure logging for the test session."""
    from llm_compass.config import get_settings

    settings = get_settings()
    settings.setup_app_logging("test")


# Use in-memory SQLite for simple logic tests,
# or a separate Postgres container for vector tests.
@pytest.fixture(name="session")
def session_fixture():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    # Import Base from models to create tables
    from llm_compass.data.models import Base

    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with SessionLocal() as session:
        yield session
