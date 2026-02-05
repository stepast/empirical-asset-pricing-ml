from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def iter_gkx_splits(
    df: pd.DataFrame,
    date_col: str = "eom",
    train_years: int = 18,
    val_years: int = 12,
    test_start_year: int = 1987,
    test_end_year: int | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """
    GKX-style splits with:
      - expanding training window
      - fixed-length rolling validation window
      - 1-year rolling test window

    Initial layout: for given test_start_year T (e.g. 1987)
      train: [T - val_years - train_years, ..., T - val_years - 1]
      val  : [T - val_years,               ..., T - 1]
      test : [T]

    For each subsequent test year T+1:
      train: lower bound stays fixed, upper bound increases by 1
      val  : moves forward by 1 year
      test : moves forward by 1 year
    """
    if date_col not in df.columns:
        raise KeyError(f"date_col '{date_col}' not found in DataFrame")

    years = df[date_col].dt.year.to_numpy()
    min_year = years.min()
    max_year = years.max()

    if test_end_year is None:
        test_end_year = max_year

    # Initial training start is fixed, determined by test_start_year and window lengths
    initial_train_start = test_start_year - val_years - train_years
    if initial_train_start < min_year:
        raise ValueError(
            f"Not enough history for test_start_year={test_start_year} "
            f"with train_years={train_years}, val_years={val_years}. "
            f"Need data from at least {initial_train_start}, have {min_year}."
        )

    # For each test year, we expand the training end year, starting
    # from (test_start_year - val_years - 1) and moving forward.
    for test_year in range(test_start_year, test_end_year + 1):
        if test_year > max_year:
            break

        # Train starts fixed at initial_train_start
        train_start = initial_train_start
        # Train end is just before the validation window begins
        train_end = test_year - val_years - 1

        # Validation window: fixed length, immediately before test year
        val_start = test_year - val_years
        val_end = test_year - 1

        # Build masks
        train_mask = (years >= train_start) & (years <= train_end)
        val_mask = (years >= val_start) & (years <= val_end)
        test_mask = (years == test_year)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(test_idx) == 0:
            continue

        yield {
            "test_year": test_year,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        }