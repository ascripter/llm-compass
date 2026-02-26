"""
Database connection and session management.
Req 1.2: Central access point for PostgreSQL + pgvector.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from llm_compass.config import Settings


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        url = self.settings.get_db_url()
        self.engine = create_engine(url, echo=True)  # echo=False in prod
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    # Database URL from environment variable
    # e.g. postgresql://user:pass@localhost:5432/database_name
    # DATABASE_URL = os.getenv("DATABASE_URL")
    # if DATABASE_URL is None:
    #     raise ValueError("DATABASE_URL environment variable is not set")

    def get_session(self):
        """
        Dependency for getting a database session (FastAPI)
        Usage:
            with get_session() as session:
                ...
        """
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def init_db(self):
        """
        Creates tables if they don't exist.
        Run this always on startup.
        """
        from .models import Base

        Base.metadata.create_all(self.engine)
