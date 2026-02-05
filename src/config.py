from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]


def get_data_root() -> Path:
    """Get data directory from DATA_DIR env var or default to <repo>/data."""
    env = os.environ.get("DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return get_project_root() / "data"


def get_processed_dir() -> Path:
    """Return data/processed directory."""
    return get_data_root() / "processed"


def get_raw_dir() -> Path:
    """Return data/raw directory."""
    return get_data_root() / "raw"


def get_results_dir() -> Path:
    """Return data/processed/results directory, creating it if needed."""
    out = get_processed_dir() / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out
