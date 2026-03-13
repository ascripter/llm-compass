"""
Global configuration settings.

Loads environment variables (DB credentials, API keys) and defines
application constants.
"""

import os
import logging
import logging.config
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

from dotenv import load_dotenv
from pydantic import SecretStr


_MANDATORY_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "LLM_COMPASS_STORAGE_PATH",
    "LLM_COMPASS_LOG_PATH",
)


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    repo_root: Path
    openrouter_api_key: str
    openrouter_base_url: str
    storage_path: Path
    log_path: Path
    log_file_level_backend: str
    log_console_level_backend: str
    log_file_level_frontend: str
    log_console_level_frontend: str
    log_file_level_dev: str
    log_console_level_dev: str

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        load_dotenv_file: bool = True,
        dotenv_path: str | Path | None = None,
        project_root: Path | None = None,
        repo_root: Path | None = None,
        create_storage_dir: bool = True,
        create_log_dir: bool = True,
        log_file_level_backend: str = "DEBUG",
        log_console_level_backend: str = "INFO",
        log_file_level_frontend: str = "DEBUG",
        log_console_level_frontend: str = "INFO",
        log_file_level_dev: str = "DEBUG",
        log_console_level_dev: str = "DEBUG",
    ) -> "Settings":
        # By default, load_dotenv() does not override already-set env vars
        if load_dotenv_file:
            load_dotenv(dotenv_path=dotenv_path)

        source = os.environ if env is None else env

        missing = [k for k in _MANDATORY_ENV_VARS if not source.get(k)]
        if missing:
            raise ValueError(f"Missing env vars: {', '.join(missing)}")

        storage = Path(source["LLM_COMPASS_STORAGE_PATH"]).expanduser().resolve()
        log_path = Path(source["LLM_COMPASS_LOG_PATH"]).expanduser().resolve()
        if create_storage_dir:
            storage.mkdir(parents=True, exist_ok=True)
        if create_log_dir:
            log_path.mkdir(parents=True, exist_ok=True)

        return cls(
            project_root=project_root or Path(__file__).absolute().parent,
            repo_root=repo_root or Path(__file__).absolute().parent.parent.parent,
            openrouter_api_key=source["OPENROUTER_API_KEY"],
            openrouter_base_url=source["OPENROUTER_BASE_URL"],
            storage_path=storage,
            log_path=log_path,
            log_file_level_backend=log_file_level_backend,
            log_console_level_backend=log_console_level_backend,
            log_file_level_frontend=log_file_level_frontend,
            log_console_level_frontend=log_console_level_frontend,
            log_file_level_dev=log_file_level_dev,
            log_console_level_dev=log_console_level_dev,
        )

    def get_benchmark_description_csv(self) -> Path:
        return Path(self.storage_path, "benchmark_descriptions.csv")

    def get_faiss_path(self) -> Path:
        return Path(self.storage_path, "benchmark_descriptions.faiss")

    def get_db_path(self) -> Path:
        return Path(self.storage_path, "llm_compass.sqlite")

    def get_db_url(self) -> str:
        return f"sqlite:///{self.get_db_path()}"

    def make_llm(self, model: str, **kwargs):
        """Create a ChatOpenAI instance pre-configured for OpenRouter."""
        from langchain_openai import ChatOpenAI  # lazy import — optional at config level

        return ChatOpenAI(
            model=model,
            api_key=SecretStr(self.openrouter_api_key),
            base_url=self.openrouter_base_url,
            **kwargs,
        )

    def setup_app_logging(self, name: str) -> logging.Logger:
        """
        Configures the root logger for the calling application.
        """
        assert name in ("backend", "frontend", "dev")
        filename = f"{name}.log"

        config = {
            "version": 1,
            "disable_existing_loggers": False,  # crucial: keeps module loggers active
            "formatters": {
                "console": {"format": "%(name)s - %(levelname)s: %(message)s"},
                "file": {"format": "[%(asctime)s] %(name)s - %(levelname)s: %(message)s"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": getattr(self, f"log_console_level_{name}"),
                    "formatter": "console",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": getattr(self, f"log_file_level_{name}"),
                    "formatter": "file",
                    "filename": str(Path(self.log_path, filename)),
                    "maxBytes": 10485760,  # Rotates at 10MB
                    "backupCount": 5,  # Keeps 5 historical log files
                    "encoding": "utf8",
                },
            },
            "loggers": {},
            "root": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
            },
        }

        # The following modules will only log WARNING and ERROR level
        silence_modules = [
            "httpcore.http11",
            "httpcore.connection",
            "sqlalchemy.engine.Engine",
            "uvicorn.access",
            "uvicorn.error",
        ]
        for module in silence_modules:
            config["loggers"][module] = {"level": "WARNING", "propagate": False}

        # Apply the configuration to the process
        logging.config.dictConfig(config)

        # Return a generic logger instance just in case the caller wants it immediately
        return logging.getLogger(__name__)


# lru_cache effectively Singleton behavior on get_settings()
# (lightweight solution instead of dependency injection pattern for settings)
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
