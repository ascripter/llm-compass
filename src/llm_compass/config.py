"""
Global configuration settings.

Loads environment variables (DB credentials, API keys) and defines
application constants.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


_MANDATORY_ENV_VARS = ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "LLM_COMPASS_STORAGE_PATH")


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    openrouter_api_key: str
    openrouter_base_url: str
    storage_path: Path

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        load_dotenv_file: bool = True,
        dotenv_path: str | Path | None = None,
        project_root: Path | None = None,
        create_storage_dir: bool = True,
    ) -> "Settings":
        # By default, load_dotenv() does not override already-set env vars
        if load_dotenv_file:
            load_dotenv(dotenv_path=dotenv_path)

        source = os.environ if env is None else env

        missing = [k for k in _MANDATORY_ENV_VARS if not source.get(k)]
        if missing:
            raise ValueError(f"Missing env vars: {', '.join(missing)}")

        storage = Path(source["LLM_COMPASS_STORAGE_PATH"]).expanduser().resolve()
        if create_storage_dir:
            storage.mkdir(parents=True, exist_ok=True)

        return cls(
            project_root=project_root or Path(__file__).absolute().parent,
            openrouter_api_key=source["OPENROUTER_API_KEY"],
            openrouter_base_url=source["OPENROUTER_BASE_URL"],
            storage_path=storage,
        )

    def get_benchmark_description_csv(self) -> Path:
        return Path(self.storage_path, "benchmark_descriptions.csv")

    def get_faiss_path(self) -> Path:
        return Path(self.storage_path, "benchmark_descriptions.faiss")

    def get_db_path(self) -> Path:
        return Path(self.storage_path, "llm_compass.sqlite")

    def get_db_url(self) -> str:
        return f"sqlite:///{self.get_db_path()}"


# lru_cache effectively Singleton behavior on get_settings()
# (lightweight solution instead of dependency injection pattern for settings)
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
