from __future__ import annotations
from typing import Any
from collections.abc import Sequence

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler

def ff49_dummies_for_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    col: str = "ff49",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct Fama–French industry dummy variables (one-hot encodings)
    
    The set of industry categories is determined using the TRAIN sample only
    to avoid look-ahead bias

    """

    # Identify industries present in TRAIN only
    train_ff = df.iloc[train_idx][col].astype("Int64")
    cats = np.sort(train_ff.dropna().unique())

    # Edge case: no industries available in TRAIN
    if len(cats) == 0:
        n_tr, n_va, n_te = len(train_idx), len(val_idx), len(test_idx)
        return (
            np.zeros((n_tr, 0), dtype=np.float32),
            np.zeros((n_va, 0), dtype=np.float32),
            np.zeros((n_te, 0), dtype=np.float32),
        )

    # Freeze the category set to TRAIN industries only
    cat_type = pd.api.types.CategoricalDtype(categories=cats)

    def _make(idx: np.ndarray) -> pd.DataFrame:
        """
        Build dummy variables for a given index set using
        the TRAIN-defined category set.
        """
        s = df.iloc[idx][col].astype("Int64")
        cat = s.astype(cat_type)
        return pd.get_dummies(cat, prefix=col, dtype=np.float32)

    # Build TRAIN dummies and record canonical column order
    d_train = _make(train_idx)
    dummy_cols = list(d_train.columns)

    def _make_block(idx: np.ndarray) -> np.ndarray:
        """
        Build dummy matrix for a split and reindex to TRAIN columns,
        filling missing categories with zeros.
        """
        d = _make(idx).reindex(columns=dummy_cols, fill_value=0.0)
        return d.to_numpy(dtype=np.float32)

    return (
        _make_block(train_idx),
        _make_block(val_idx),
        _make_block(test_idx),
    )

def scale_macros_with_robust(
    X: pd.DataFrame,
    df: pd.DataFrame,
    macro_cols: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    date_col: str = "eom",
    quantile_range: tuple[float, float] = (25.0, 75.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust scaling for macro predictors in a panel setting.

    - Fit RobustScaler on TRAIN months only, using *one macro observation per date*
      (each month counts once).
    - Apply the fitted transform to TRAIN/VAL/TEST rows.
    - Non-macro columns are left unchanged.

    """
    macro_cols = list(macro_cols)

    # Work on copies because we'll overwrite macro columns in-place
    X_train = X.iloc[train_idx].copy()
    X_val   = X.iloc[val_idx].copy()
    X_test  = X.iloc[test_idx].copy()

    # Build a per-date macro table from TRAIN only (one row per month)
    train_dates = df.iloc[train_idx][date_col].to_numpy()

    train_macro_unique = (
        pd.DataFrame(X_train[macro_cols].to_numpy(), columns=macro_cols)
        .assign(_date=train_dates)
        .groupby("_date", sort=True)[macro_cols]
        .first()
    )

    # RobustScaler centers by median and scales by IQR 
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=quantile_range,
    )
    scaler.fit(train_macro_unique)

    # Transform macro columns in each split; leave other columns unchanged
    for X_split in (X_train, X_val, X_test):
        X_split.loc[:, macro_cols] = scaler.transform(X_split[macro_cols])

    return (
        X_train.to_numpy(dtype=np.float32, copy=False),
        X_val.to_numpy(dtype=np.float32, copy=False),
        X_test.to_numpy(dtype=np.float32, copy=False),
    )


def prepare_features_and_target(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "ret_exc_lead1m",
    required_cols: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Construct base feature matrix X and target y, then remove rows that are not usable.

    """
    feature_cols = list(feature_cols)

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop missing target and required columns first (keeps feature processing aligned)
        
    req = list(required_cols)
    if req:
        df_req = df[req].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        ok_req = np.isfinite(df_req.to_numpy(dtype=float)).all(axis=1)
    else:
        ok_req = np.ones(len(df), dtype=bool)
    
    mask = y.notna() & ok_req

    X = X.loc[mask]
    y = y.loc[mask]
    df_sub = df.loc[mask]

    # Ensure numeric features; convert non-numeric to NaN, and inf -> NaN
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Keep only rows where ALL features and target are finite
    finite_mask = np.isfinite(X.to_numpy(dtype=float)).all(axis=1) & np.isfinite(y.to_numpy(dtype=float))

    # Reset index together to keep alignment stable for downstream indexing
    X_clean = X.loc[finite_mask].reset_index(drop=True)
    y_clean = y.loc[finite_mask].reset_index(drop=True)
    df_clean = df_sub.loc[finite_mask].reset_index(drop=True)

    return X_clean, y_clean, df_clean


from .schemas import FeatureConfig


def make_feature_builder(
    X_base: pd.DataFrame,
    y: pd.Series,
    df_aligned: pd.DataFrame,
    X_columns: list[str],
    cfg: FeatureConfig,
):
    """
    - Macros are never included as standalone regressors.
    - If cfg.interactions=False: macros are not used at all.
    - If cfg.interactions=True: macros are used only to form (chars × macros) interactions.
    """
    macro_cols = list(cfg.macro_cols)
    char_cols = list(cfg.char_cols)

    # sanity: required columns exist in X_base
    needed = list(char_cols)
    if cfg.interactions:   # only require macros if we will use them
        needed += list(macro_cols)

    missing = [c for c in needed if c not in X_columns]
    if missing:
        raise ValueError(f"Missing expected feature columns in X_base: {missing[:10]}")

    def _builder(train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, Any]:
        # 1) FF industry dummies (TRAIN defines category set)
        d_tr, d_va, d_te = ff49_dummies_for_split(
            df_aligned, train_idx, val_idx, test_idx, col=cfg.ff49_col
        )

        # 2) Base characteristics block (NO macros)
        Xc_tr = X_base.iloc[train_idx][char_cols].to_numpy(dtype=np.float32, copy=False)
        Xc_va = X_base.iloc[val_idx][char_cols].to_numpy(dtype=np.float32, copy=False)
        Xc_te = X_base.iloc[test_idx][char_cols].to_numpy(dtype=np.float32, copy=False)

        # 3) Optional interactions (chars × scaled macros), macros not included standalone
        if cfg.interactions and macro_cols:

            X_macro_only = X_base[macro_cols]

            Xm_tr, Xm_va, Xm_te = scale_macros_with_robust(
                X=X_macro_only,
                df=df_aligned,
                macro_cols=macro_cols,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                date_col=cfg.date_col,
                quantile_range=cfg.quantile_range,
            )

            def _inter(chars: np.ndarray, macros: np.ndarray) -> np.ndarray:
                n, P = chars.shape
                J = macros.shape[1]
                return (chars[:, :, None] * macros[:, None, :]).reshape(n, P * J).astype(np.float32, copy=False)

            inter_tr = _inter(Xc_tr, Xm_tr)
            inter_va = _inter(Xc_va, Xm_va)
            inter_te = _inter(Xc_te, Xm_te)
        else:
            inter_tr = np.zeros((len(train_idx), 0), dtype=np.float32)
            inter_va = np.zeros((len(val_idx), 0), dtype=np.float32)
            inter_te = np.zeros((len(test_idx), 0), dtype=np.float32)

        # 4) Concatenate: chars + FF dummies + interactions
        X_train = np.hstack([Xc_tr, d_tr, inter_tr]).astype(np.float32, copy=False)
        X_val   = np.hstack([Xc_va, d_va, inter_va]).astype(np.float32, copy=False)
        X_test  = np.hstack([Xc_te, d_te, inter_te]).astype(np.float32, copy=False)

        # 5) y blocks
        y_train = y.iloc[train_idx].to_numpy(dtype=np.float32, copy=False)
        y_val   = y.iloc[val_idx].to_numpy(dtype=np.float32, copy=False)
        y_test  = y.iloc[test_idx].to_numpy(dtype=np.float32, copy=False)

        # 6) Final safety: drop non-finite rows within each split
        def _finite(Xb: np.ndarray, yb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            m = np.isfinite(Xb).all(axis=1) & np.isfinite(yb)
            return Xb[m], yb[m]

        X_train, y_train = _finite(X_train, y_train)
        X_val, y_val     = _finite(X_val, y_val)
        X_test, y_test   = _finite(X_test, y_test)

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val,     "y_val": y_val,
            "X_test": X_test,   "y_test": y_test,
        }

    return _builder
