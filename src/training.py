from __future__ import annotations

import math
from typing import Any, Literal
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer

from .models import MLPReturnPredictor
from .schemas import TrainConfig
from .splitting import iter_gkx_splits
from .utils import r2_oos, seed_everything


def run_gkx_sklearn_grid_with_features(
    *,
    df: pd.DataFrame,
    feature_builder: Callable[[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]],
    base_estimator,
    train: TrainConfig,
    param_grid: dict[str, Any] | None = None,
    estimator_name: str = "",
    tune_hyperparams: bool = True,
    verbose: bool = True,
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Run GKX-style rolling out-of-sample evaluation for sklearn estimators.
    
    Trains on expanding window, tunes on validation, tests on 1-year horizon.
    """
    scorer = make_scorer(r2_oos, greater_is_better=True)
    param_grid = param_grid or {}
    name = estimator_name or type(base_estimator).__name__

    per_year_records = []
    preds_by_year = {}
    sse_total = 0.0
    sst_total = 0.0

    for split in iter_gkx_splits(
        df=df,
        date_col=train.date_col,
        train_years=train.train_years,
        val_years=train.val_years,
        test_start_year=train.test_start_year,
        test_end_year=train.test_end_year,
    ):
        test_year = split["test_year"]
        tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]

        if len(tr) == 0 or len(te) == 0:
            continue

        feats = feature_builder(tr, va, te)

        X_train = feats["X_train"].astype(np.float32)
        y_train = feats["y_train"].astype(np.float32)
        X_test = feats["X_test"].astype(np.float32)
        y_test = feats["y_test"].astype(np.float32)
        X_val = feats.get("X_val", np.empty((0, X_train.shape[1]))).astype(np.float32)
        y_val = feats.get("y_val", np.empty(0)).astype(np.float32)

        if X_train.size == 0 or X_test.size == 0:
            continue

        n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
        do_tune = tune_hyperparams and param_grid and n_val > 0

        if do_tune:
            X_cv = np.vstack([X_train, X_val])
            y_cv = np.concatenate([y_train, y_val])
            test_fold = np.concatenate([np.full(n_train, -1), np.zeros(n_val)])
            
            grid = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                scoring=scorer,
                cv=PredefinedSplit(test_fold),
                n_jobs=1,
                refit=True,
            )
            grid.fit(X_cv, y_cv)
            fitted = grid.best_estimator_
            best_params = grid.best_params_
            best_val_score = grid.best_score_
        else:
            X_fit = np.vstack([X_train, X_val]) if n_val > 0 else X_train
            y_fit = np.concatenate([y_train, y_val]) if n_val > 0 else y_train
            fitted = clone(base_estimator).fit(X_fit, y_fit)
            best_params = {}
            best_val_score = float("nan")

        y_hat = fitted.predict(X_test).astype(np.float32)
        r2_year = r2_oos(y_test, y_hat)

        resid = y_test - y_hat
        sse_total += np.dot(resid, resid)
        sst_total += np.dot(y_test, y_test)

        if return_predictions:
            preds_by_year[test_year] = (y_test.copy(), y_hat.copy())

        per_year_records.append({
            "test_year": test_year,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "r2_oos": r2_year,
            **best_params,
        })

        if verbose:
            global_r2 = 1 - sse_total / sst_total if sst_total > 0 else float("nan")
            msg = (
                f"{name} | {test_year} | p={X_test.shape[1]} | "
                f"train={n_train}, val={n_val}, test={n_test} | "
                f"R2_test={r2_year:.4f}, R2_global={global_r2:.4f}"
            )
            if do_tune:
                msg += f" | R2_val={best_val_score:.4f} {best_params}"
            print(msg)

    per_year_df = pd.DataFrame(per_year_records).sort_values("test_year").reset_index(drop=True)
    global_r2 = 1 - sse_total / sst_total if sst_total > 0 else float("nan")

    result = {
        "per_year": per_year_df,
        "overall": {"global_r2": global_r2},
        "global_r2": global_r2,
    }
    if return_predictions:
        result["predictions_by_year"] = preds_by_year

    return result


def _make_loader(
    X: np.ndarray,
    y: np.ndarray | None,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    device: torch.device = None,
) -> torch.utils.data.DataLoader:
    """Create DataLoader from numpy arrays."""
    device = device or torch.device("cpu")
    pin_memory = device.type == "cuda"

    X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    
    if y is None:
        dataset = torch.utils.data.TensorDataset(X_t)
    else:
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        y_t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        dataset = torch.utils.data.TensorDataset(X_t, y_t)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


@torch.inference_mode()
def _eval_loss_and_r2(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Compute mean loss and R2 on a DataLoader."""
    model.eval()
    non_blocking = device.type == "cuda"

    total_loss = 0.0
    sse = 0.0
    sst = 0.0
    n_total = 0

    for batch in loader:
        xb = batch[0].to(device, non_blocking=non_blocking)
        yb = batch[1].to(device, non_blocking=non_blocking).view(-1)
        yhat = model(xb).view(-1)

        loss_val = criterion(yhat, yb)
        total_loss += loss_val.item() * len(yb)
        
        diff = yb - yhat
        sse += diff.square().sum().item()
        sst += yb.square().sum().item()
        n_total += len(yb)

    mean_loss = total_loss / n_total if n_total > 0 else float("nan")
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    
    return mean_loss, r2


def train_one_split_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    input_dim: int,
    device: torch.device,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dims: Sequence[int],
    dropout: float = 0.0,
    use_batchnorm: bool = True,
    loss: Literal["mse", "huber"] = "mse",
    huber_delta: float = 1.0,
    patience: int = 5,
    min_delta: float = 1e-5,
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.0,
    use_plateau_scheduler: bool = True,
    plateau_factor: float = 0.3,
    plateau_patience: int = 0,
    plateau_min_lr: float = 1e-6,
    plateau_threshold: float | None = 1e-5,
    verbose: bool = False,
) -> tuple[nn.Module, dict[str, float]]:
    """Train MLP on one split with early stopping based on validation loss."""
    
    model = MLPReturnPredictor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if loss == "mse":
        criterion = nn.MSELoss()
    elif loss == "huber":
        criterion = nn.SmoothL1Loss(beta=huber_delta)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    def l1_penalty() -> torch.Tensor:
        if l1_lambda <= 0:
            return torch.zeros((), device=device)
        penalty = torch.zeros((), device=device)
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                continue
            penalty = penalty + p.abs().sum()
        return penalty
    
    def l2_penalty() -> torch.Tensor:
        if l2_lambda <= 0:
            return torch.zeros((), device=device)
        penalty = torch.zeros((), device=device)
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                continue
            penalty = penalty + (p * p).sum()
        return penalty


    train_loader = _make_loader(
        X_train, y_train, batch_size,
        shuffle=True,
        drop_last=use_batchnorm,
        device=device,
    )
    val_loader = _make_loader(X_val, y_val, batch_size, device=device)

    scheduler = None
    if use_plateau_scheduler:
        threshold = min_delta if plateau_threshold is None else plateau_threshold
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=threshold,
            min_lr=plateau_min_lr,
        )

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0
    non_blocking = device.type == "cuda"

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []

        for xb_cpu, yb_cpu in train_loader:
            xb = xb_cpu.to(device, non_blocking=non_blocking)
            yb = yb_cpu.to(device, non_blocking=non_blocking).view(-1)

            optimizer.zero_grad(set_to_none=True)
            yhat = model(xb).view(-1)
            
            data_loss = criterion(yhat, yb)
            total_loss = data_loss
            if l1_lambda > 0 or l2_lambda > 0:
                total_loss = (total_loss + l1_lambda * l1_penalty() 
                            + l2_lambda * l2_penalty())
                            
            total_loss.backward()
            optimizer.step()
            train_losses.append(data_loss.item())

        val_loss, val_r2 = _eval_loss_and_r2(model, val_loader, criterion, device)

        if scheduler and not math.isnan(val_loss):
            scheduler.step(val_loss)

        if verbose:
            print(f"Epoch {epoch:03d} | train={np.mean(train_losses):.6f} | val={val_loss:.6f} | r2={val_r2:.4f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    final_val_loss, final_val_r2 = _eval_loss_and_r2(model, val_loader, criterion, device)
    return model, {"val_loss": final_val_loss, "val_r2": final_val_r2}


def _tune_mlp_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    input_dim: int,
    device: torch.device,
    n_epochs: int = 100,
    param_grid: dict[str, list] | None = None,
    verbose: bool = False,
    **fixed_kwargs,
) -> dict[str, Any]:
    """Simple grid search over MLP hyperparameters."""
    param_grid = param_grid or {
        "lr": [1e-3, 3e-4],
        "l1_lambda": [0.0, 1e-4, 1e-3],
    }

    best_val_r2 = -float("inf")
    best_params = None

    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        trial_kwargs = {**fixed_kwargs, **params}

        _, metrics = train_one_split_mlp(
            X_train, y_train, X_val, y_val,
            input_dim=input_dim,
            device=device,
            n_epochs=n_epochs,
            verbose=False,
            **trial_kwargs,
        )

        if metrics["val_r2"] > best_val_r2:
            best_val_r2 = metrics["val_r2"]
            best_params = params

        if verbose:
            print(f"Params: {params} → val_r2={metrics['val_r2']:.4f}")

    return best_params


@torch.inference_mode()
def _predict_in_batches(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Generate predictions in batches."""
    model.eval()
    loader = _make_loader(X, None, batch_size, device=device)
    predictions = []

    for (xb,) in loader:
        xb = xb.to(device, non_blocking=device.type == "cuda")
        yhat = model(xb).view(-1)
        predictions.append(yhat.cpu().numpy())

    return np.concatenate(predictions)


def run_gkx_mlp(
    *,
    df: pd.DataFrame,
    feature_builder: Callable,
    train: TrainConfig,
    n_epochs: int = 200,
    batch_size: int = 10000,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    hidden_dims: Sequence[int] = (32, 16, 8),
    dropout: float = 0.0,
    use_batchnorm: bool = True,
    loss: Literal["mse", "huber"] = "mse",
    huber_delta: float = 1.0,
    patience: int = 5,
    min_delta: float = 1e-5,
    ensemble_n: int = 5,
    l1_lambda: float = 0.0,
    deterministic: bool = True,
    use_plateau_scheduler: bool = True,
    plateau_factor: float = 0.3,
    plateau_patience: int = 0,
    plateau_min_lr: float = 1e-6,
    plateau_threshold: float | None = 1e-5,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run MLP with ensemble averaging across multiple random seeds."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_year_records = []
    sse_total = 0.0
    sst_total = 0.0

    for split in iter_gkx_splits(
        df=df,
        date_col=train.date_col,
        train_years=train.train_years,
        val_years=train.val_years,
        test_start_year=train.test_start_year,
        test_end_year=train.test_end_year,
    ):
        test_year = split["test_year"]
        tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]

        if len(tr) == 0 or len(te) == 0:
            continue

        feats = feature_builder(tr, va, te)
        X_train = feats["X_train"].astype(np.float32)
        y_train = feats["y_train"].astype(np.float32)
        X_val = feats.get("X_val", np.empty((0, X_train.shape[1]))).astype(np.float32)
        y_val = feats.get("y_val", np.empty(0)).astype(np.float32)
        X_test = feats["X_test"].astype(np.float32)
        y_test = feats["y_test"].astype(np.float32)

        if X_train.size == 0 or X_test.size == 0:
            continue

        input_dim = X_train.shape[1]
        n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)

        ensemble_preds = []
        for seed_idx in range(ensemble_n):
            if deterministic:
                seed_everything(seed_idx)

            model, _ = train_one_split_mlp(
                X_train, y_train, X_val, y_val,
                input_dim=input_dim,
                device=device,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                hidden_dims=hidden_dims,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                loss=loss,
                huber_delta=huber_delta,
                patience=patience,
                min_delta=min_delta,
                l1_lambda=l1_lambda,
                use_plateau_scheduler=use_plateau_scheduler,
                plateau_factor=plateau_factor,
                plateau_patience=plateau_patience,
                plateau_min_lr=plateau_min_lr,
                plateau_threshold=plateau_threshold,
                verbose=False,
            )

            y_hat_seed = _predict_in_batches(model, X_test, batch_size, device)
            ensemble_preds.append(y_hat_seed)

        y_hat = np.mean(ensemble_preds, axis=0).astype(np.float32)
        r2_year = r2_oos(y_test, y_hat)

        resid = y_test - y_hat
        sse_total += np.dot(resid, resid)
        sst_total += np.dot(y_test, y_test)

        per_year_records.append({
            "test_year": test_year,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "r2_oos": r2_year,
        })

        if verbose:
            global_r2 = 1 - sse_total / sst_total if sst_total > 0 else float("nan")
            print(f"MLP | {test_year} | train={n_train}, val={n_val}, test={n_test} | "
                  f"R2_test={r2_year:.4f}, R2_global={global_r2:.4f}")

    per_year_df = pd.DataFrame(per_year_records).sort_values("test_year").reset_index(drop=True)
    global_r2 = 1 - sse_total / sst_total if sst_total > 0 else float("nan")

    return {
        "per_year": per_year_df,
        "overall": {"global_r2": global_r2},
        "global_r2": global_r2,
    }
