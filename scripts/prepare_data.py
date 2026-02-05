from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence, Literal

import logging
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Basic configuration
# -------------------------------------------------------------------

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().parents[0]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"

JKP_FILE = "JKP_US_raw_full.parquet"
MACRO_FILE = "macro_predictors.csv"
OUTPUT_FILE = "JKP_US_processed.parquet"

START_SAMPLE_YEAR = 1957

ID_VARS = [
    "id", "permno", "permco", "gvkey", "iid",
    "date", "eom",
    "curcd", "fx", "excntry",
    "me", "size_grp", "me_company",
    "common", "exch_main", "primary_sec", "obs_main",
    "comp_exchg", "comp_tpci",
    "source_crsp", "crsp_exchcd", "crsp_shrcd",
    "ff49", "sic", "naics", "gics",
]

RET_VARS = [
    "ret_exc_lead1m", "ret_exc", "ret", "ret_lead1m",
    "ret_local", "ret_lag_dif",
    "prc_local", "prc_high", "prc_low",
    "bidask", "dolvol",
    "shares", "tvol", "adjfct",
]

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Core helpers
# -------------------------------------------------------------------

def _ensure_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric and replace inf with NaN."""
    out = df.copy()
    cols = list(cols)
    out[cols] = (
        out[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    return out


def _impute_cs_median(df: pd.DataFrame, cols: Sequence[str], date_col: str) -> pd.DataFrame:
    """
    Cross-sectionally impute NaNs with the per-month median.
    Any remaining NaNs (all-missing month/col) become 0.

    Memory-safe version: avoids groupby.transform('median') which materializes
    a full (n_rows x n_cols) intermediate.
    """
    out = df.copy()
    cols = list(cols)

    # Compute one median row per date (small: n_months x n_cols)
    med_by_date = out.groupby(date_col, sort=False)[cols].median()

    # Join back only once using the date key (avoids giant transform temp)
    # This creates a median-aligned frame of same shape, but built column-wise
    # without the transform internal consolidation blow-up.
    med_aligned = out[[date_col]].join(med_by_date, on=date_col)[cols]

    out[cols] = out[cols].where(~out[cols].isna(), med_aligned).fillna(0.0)
    return out


def standardize_cross_sectional(
    df: pd.DataFrame,
    char_cols: Sequence[str],
    date_col: str = "eom",
    method: Literal["zscore", "rank"] = "rank",
    impute: Literal["median", "none"] = "median",
) -> pd.DataFrame:
    """
    Cross-sectionally transform characteristics by date.

    Parameters
    ----------
    method:
        "zscore": per-month (x - mean) / std
        "rank":   per-month rank transform mapped to [-1, 1]
    impute:
        "median": cross-sectional per-month median imputation (then 0 for all-missing)
        "none":   no imputation (NaNs remain)

    Notes
    -----
    - All characteristics are coerced to numeric; inf is treated as missing.
    - For rank transform:
        - if a month has <=1 non-missing obs, mapped values are set to 0 (and NaNs remain if impute="none").
        - with impute="median", NaNs are filled with that month’s median mapped value.
    """
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in DataFrame.")

    char_cols = list(char_cols)
    out = _ensure_numeric(df, char_cols)

    g = out.groupby(date_col)[char_cols]

    if method == "zscore":
        means = g.transform("mean")
        stds = g.transform("std").replace(0.0, 1.0).fillna(1.0)
        out[char_cols] = ((out[char_cols] - means) / stds).astype("float64")

    elif method == "rank":
        r = g.rank(method="average", na_option="keep")
        n = g.transform("count")

        mapped = 2.0 * ((r - 1.0) / (n - 1.0) - 0.5)
        mapped = mapped.where(n > 1, 0.0)

        out[char_cols] = mapped.astype(np.float32)

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'zscore' or 'rank'.")

    if impute == "median":
        out = _impute_cs_median(out, char_cols, date_col=date_col)
        out[char_cols] = out[char_cols].astype(np.float32)
    elif impute == "none":
        pass
    else:
        raise ValueError(f"Unknown impute: {impute!r}. Use 'median' or 'none'.")

    return out


def load_and_process_macro() -> tuple[pd.DataFrame, list[str]]:
    """
    Load macro predictors, minimally process them, and align by month-end.

    Steps:
        - read CSV
        - filter to years >= START_SAMPLE_YEAR
        - sort by date
        - lag each macro series by one month (to avoid look-ahead)
        - create 'eom' month-end dates

    NOTE: no standardization / scaling is done here.
    That is left to the modeling stage, where you can choose the
    training window used to compute means/stds.
    """
    path = RAW_DIR / MACRO_FILE
    if not path.exists():
        raise FileNotFoundError(f"Macro predictors file not found at {path}")

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise KeyError("Expected column 'date' in macro predictors file.")

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= START_SAMPLE_YEAR].sort_values("date").reset_index(drop=True)

    macro_cols = [c for c in df.columns if c != "date"]
    if not macro_cols:
        raise ValueError("No macro predictor columns found.")

    # Lag by one month to avoid look-ahead
    df[macro_cols] = df[macro_cols].shift(1)

    # Align to month-end
    df["eom"] = df["date"] + pd.offsets.MonthEnd(0)

    # Ensure one row per month to avoid duplicating panel rows on merge
    if df["eom"].duplicated().any():
        dups = df.loc[df["eom"].duplicated(), "eom"].unique()
        raise ValueError(f"Macro data has duplicate months in 'eom': {dups[:5]} (showing up to 5).")

    logger.info(
        "Loaded macro predictors: %d rows, %d variables (unscaled, lagged)",
        df.shape[0],
        len(macro_cols),
    )
    return df, macro_cols


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load JKP panel
    jkp_path = RAW_DIR / JKP_FILE
    if not jkp_path.exists():
        raise FileNotFoundError(f"JKP file not found at {jkp_path}")

    jkp = pd.read_parquet(jkp_path)

    if "eom" not in jkp.columns:
        raise KeyError("Expected column 'eom' in JKP data.")

    jkp["eom"] = pd.to_datetime(jkp["eom"])
    jkp = jkp.sort_values(["id", "eom"])
    jkp = jkp[jkp["eom"].dt.year >= START_SAMPLE_YEAR].reset_index(drop=True)
    
    # Add next month raw returns
    jkp["ret_lead1m"] = jkp.groupby("id")["ret"].shift(-1)

    logger.info("JKP panel loaded: %d rows, %d columns", jkp.shape[0], jkp.shape[1])

    # 2) Identify characteristic columns (all non-ID, non-return columns)
    all_cols = set(jkp.columns)
    excluded = set(ID_VARS) | set(RET_VARS)

    char_cols = sorted(all_cols - excluded)
    if not char_cols:
        raise ValueError("No characteristic columns found after excluding ID/return columns.")

    logger.info("Using full characteristic set: %d columns", len(char_cols))

    # 3) Standardize firm characteristics cross-sectionally
    jkp = standardize_cross_sectional(
        jkp,
        char_cols,
        date_col="eom",
        method="rank",    
        impute="median",   
    )

    # Optional sanity check: rank-mapped should be in [-1, 1] up to eps
    mx = float(np.nanmax(jkp[char_cols].to_numpy()))
    mn = float(np.nanmin(jkp[char_cols].to_numpy()))
    if mx > 1.000001 or mn < -1.000001:
        raise ValueError(f"Rank-mapped characteristics out of bounds: min={mn:.6f}, max={mx:.6f}")

    # 4) Load and process macro series (unscaled, lagged)
    macro, macro_cols = load_and_process_macro()

    # 5) Merge macro into panel
    n_before = jkp.shape[0]
    jkp = jkp.merge(macro.drop(columns=["date"]), on="eom", how="left", validate="m:1")
    if jkp.shape[0] != n_before:
        raise RuntimeError("Row count changed after macro merge (should not happen with validate='m:1').")

    # 6) Save processed file (macros unscaled)
    out_path = PROC_DIR / OUTPUT_FILE
    jkp.to_parquet(out_path, index=False)
    logger.info("Saved processed data to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    main()
