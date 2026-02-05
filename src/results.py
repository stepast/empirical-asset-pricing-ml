from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import get_results_dir


def make_run_id(dt: datetime | None = None) -> str:
    """Generate timestamp ID for tagging outputs (e.g., 20260202_143015)."""
    dt = dt or datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def save_per_year(
    per_year: pd.DataFrame,
    *,
    model_name: str,
    feature_set: str,
    run_id: str,
) -> Path:
    """Save per-year results CSV."""
    out_dir = get_results_dir()
    fname = f"per_year__{model_name}__{feature_set}__{run_id}.csv"
    out_path = out_dir / fname
    per_year.to_csv(out_path, index=False)
    return out_path


def save_summary(
    summary: pd.DataFrame,
    *,
    feature_set: str,
    run_id: str,
    index: bool = False,
) -> Path:
    """Save summary CSV."""
    out_dir = get_results_dir()
    fname = f"summary__{feature_set}__{run_id}.csv"
    out_path = out_dir / fname
    summary.to_csv(out_path, index=index)
    return out_path


def save_overall_json(
    overall: dict[str, Any],
    *,
    model_name: str,
    feature_set: str,
    run_id: str,
) -> Path:
    """Save overall metrics as JSON."""
    out_dir = get_results_dir()
    fname = f"overall__{model_name}__{feature_set}__{run_id}.json"
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(overall, f, indent=2, sort_keys=True)
    return out_path


def save_run_metadata(
    metadata: dict[str, Any],
    *,
    feature_set: str,
    run_id: str,
) -> Path:
    """Save run metadata (config and environment) as JSON."""
    out_dir = get_results_dir()
    fname = f"run_metadata__{feature_set}__{run_id}.json"
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return out_path

def print_summary_table(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    common_cols: list[str] | None = None,
    value_cols: list[str] | None = None,
) -> None:
    """Print formatted summary table with optional shared header."""
    if title:
        print(f"\n{title}")

    if df.empty:
        print("(no results)")
        return

    df_display = df.copy()

    # --------------------------------------------------
    # Print common (run-level) columns once
    # --------------------------------------------------
    if common_cols:
        common_cols = [c for c in common_cols if c in df_display.columns]
        if common_cols:
            print("\nRun configuration")
            print("-" * 60)
            first_row = df_display.iloc[0]
            for c in common_cols:
                print(f"{c:25}: {first_row[c]}")
            print("-" * 60)

        df_display = df_display.drop(columns=common_cols)

    # --------------------------------------------------
    # Keep only per-model columns if specified
    # --------------------------------------------------
    if value_cols:
        value_cols = [c for c in value_cols if c in df_display.columns]
        df_display = df_display[value_cols]

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------
    with pd.option_context(
        "display.max_rows", 200,
        "display.max_columns", 50,
        "display.width", 120,
    ):
        for col in df_display.columns:
            if pd.api.types.is_numeric_dtype(df_display[col]):
                df_display[col] = df_display[col].round(6)

        print(df_display.to_string(index=False))
