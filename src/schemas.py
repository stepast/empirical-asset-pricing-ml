from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FeatureConfig:
    """Controls which inputs are fed to the model."""
    macro_cols: list[str]
    char_cols: list[str]
    ff49_col: str = "ff49"
    date_col: str = "eom"
    quantile_range: tuple[float, float] = (25.0, 75.0)
    interactions: bool = False
    char_method: Literal["zscore", "rank"] = "rank"
    char_impute: Literal["median", "none"] = "median"


@dataclass
class TrainConfig:
    """Controls time-splitting, training randomness, and MLP early stopping."""
    # ---- Time split / evaluation window ----
    date_col: str = "eom"
    train_years: int = 25
    val_years: int = 5
    test_start_year: int = 2000
    test_end_year: int = 2024
    seed: int = 0
    # ---- Early stopping / scheduler ----
    patience: int = 10
    min_delta: float = 1e-6
    use_plateau_scheduler: bool = True
    plateau_factor: float = 0.5
    plateau_patience: int = 1
    plateau_min_lr: float = 1e-6
    plateau_threshold: float | None = None  # if None, use min_delta
