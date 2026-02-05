from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FeatureConfig:
    """Controls which inputs are fed to the model / policy."""
    macro_cols: list[str]
    char_cols: list[str]
    ff49_col: str = "ff49"
    date_col: str = "eom"
    quantile_range: tuple[float, float] = (25.0, 75.0)
    interactions: bool = False


@dataclass
class TrainConfig:
    """
    Controls time-splitting and training randomness.

    Notes
    -----
    - The split fields are shared across both repos.
    - Some additional training hyperparameters are only used in the portfolio-policy repo.
      (They are harmless defaults in the GKX repo.)
    """
    # ---- Time split / evaluation window (shared) ----
    date_col: str = "eom"
    train_years: int = 25
    val_years: int = 5
    test_start_year: int = 2000
    test_end_year: int | None = None
    seed: int = 0

    # ---- Portfolio-policy training hyperparameters (used in BSV) ----
    risk_aversion: float = 5.0
    epochs: int = 200
    lr: float = 1e-3
    l1: float = 0.0
    l2: float = 0.0
    weight_decay: float = 0.0

    policy_mode: Literal["long_only_softmax", "long_short_tilt"] = "long_only_softmax"
    gross_leverage: float | None = None
    turnover_penalty: float = 0.0
    max_weight: float | None = None

    # ---- Early stopping / scheduler (used in BSV; harmless defaults in GKX) ----
    patience: int = 10
    min_delta: float = 1e-6

    use_plateau_scheduler: bool = True
    plateau_factor: float = 0.5
    plateau_patience: int = 1
    plateau_min_lr: float = 1e-6
    plateau_threshold: float | None = None  # if None, use min_delta
