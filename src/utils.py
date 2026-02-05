from __future__ import annotations

import random
import numpy as np
import torch


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-sample R² following GKX: 1 - SSE_model / SSE_benchmark.
    
    Benchmark is zero prediction (historical mean).
    """
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = (y_true ** 2).sum()
    return 1.0 - numerator / denominator


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
