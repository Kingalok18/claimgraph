"""Settings: YAML files in configs/ plus .env. Environment variables win over .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _env(values: dict[str, str | None], key: str, default: str = "") -> str:
    """Read one setting. Inline comments in .env are stripped by python-dotenv."""
    if key in os.environ:
        return os.environ[key]
    return values.get(key) or default


@dataclass
class Settings:
    root: Path
    models: dict[str, Any]
    pipeline: dict[str, Any]
    database_url: str
    test_mysql_url: str
    qdrant_url: str
    grobid_url: str
    ollama_base_url: str
    openalex_mailto: str
    hf_token: str
    data_dir: Path
    papers_dir: Path
    prompts_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    uploads_dir: Path = field(init=False)
    qdrant_path: Path = field(init=False)
    cache_dir: Path = field(init=False)
    labels_dir: Path = field(init=False)
    datasets_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.prompts_dir = self.root / "prompts"
        self.results_dir = self.root / "results"
        self.models_dir = self.root / "models"
        self.uploads_dir = self.papers_dir / "uploads"
        self.qdrant_path = self.data_dir / "qdrant"
        self.cache_dir = self.data_dir / "cache"
        self.labels_dir = self.data_dir / "labels"
        self.datasets_dir = self.data_dir / "datasets"
        self.logs_dir = self.data_dir / "logs"

    @property
    def nli_finetuned_path(self) -> Path:
        return self.root / self.models["nli"]["finetuned_path"]

    @property
    def app_name(self) -> str:
        return self.pipeline.get("app_name", "ClaimGraph")

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.cache_dir, self.labels_dir, self.datasets_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_settings(root: Path | None = None) -> Settings:
    root = Path(root) if root else ROOT
    values = dotenv_values(root / ".env") if (root / ".env").exists() else {}
    pipeline = _load_yaml(root / "configs" / "pipeline.yaml")
    models = _load_yaml(root / "configs" / "models.yaml")

    data_dir = Path(os.environ.get("CLAIMGRAPH_DATA_DIR", root / "data"))
    papers_dir = Path(os.environ.get("CLAIMGRAPH_PAPERS_DIR", root / pipeline["papers_dir"]))

    database_url = _env(values, "DATABASE_URL", f"sqlite:///{(data_dir / 'claimgraph.db').as_posix()}")
    if database_url.startswith("sqlite:///") and not database_url.startswith("sqlite:////"):
        # Relative SQLite paths are relative to the project folder, not the shell's folder.
        relative = database_url.removeprefix("sqlite:///")
        if not Path(relative).is_absolute():
            database_url = f"sqlite:///{(root / relative).as_posix()}"

    return Settings(
        root=root,
        models=models,
        pipeline=pipeline,
        database_url=database_url,
        test_mysql_url=_env(values, "TEST_MYSQL_URL"),
        qdrant_url=_env(values, "QDRANT_URL"),
        grobid_url=_env(values, "GROBID_URL", "http://localhost:8070"),
        ollama_base_url=_env(values, "OLLAMA_BASE_URL", "http://localhost:11434"),
        openalex_mailto=_env(values, "OPENALEX_MAILTO"),
        hf_token=_env(values, "HF_TOKEN"),
        data_dir=data_dir,
        papers_dir=papers_dir,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings() -> None:
    """Forget the cached settings. Tests call this after changing the environment."""
    get_settings.cache_clear()
