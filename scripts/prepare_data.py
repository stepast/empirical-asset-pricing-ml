from __future__ import annotations

from pathlib import Path
import sys

import logging
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
    "ret_exc_lead1m", "ret_exc", "ret", 
    "ret_local", "ret_lag_dif",
    "prc_local", "prc_high", "prc_low",
    "bidask", "dolvol",
    "shares", "tvol", "adjfct",
]

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Core helpers
# -------------------------------------------------------------------

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
    
    # Add next month raw returns (calendar-aligned)
    jkp["eom_next"] = jkp["eom"] + pd.offsets.MonthEnd(1)
    lead = jkp[["id", "eom", "ret"]].rename(
        columns={"eom": "eom_next", "ret": "ret_lead1m"}
    )
    jkp = jkp.merge(lead, on=["id", "eom_next"], how="left", validate="1:1")
    jkp = jkp.drop(columns=["eom_next"])

    logger.info("JKP panel loaded: %d rows, %d columns", jkp.shape[0], jkp.shape[1])

    # 2) Identify characteristic columns (all non-ID, non-return columns)
    all_cols = set(jkp.columns)
    excluded = set(ID_VARS) | set(RET_VARS)

    char_cols = sorted(all_cols - excluded)
    if not char_cols:
        raise ValueError("No characteristic columns found after excluding ID/return columns.")

    logger.info("Using full characteristic set: %d columns", len(char_cols))

    # 3) Load and process macro series (unscaled, lagged)
    macro, macro_cols = load_and_process_macro()

    # 4) Merge macro into panel
    n_before = jkp.shape[0]
    jkp = jkp.merge(macro.drop(columns=["date"]), on="eom", how="left", validate="m:1")
    if jkp.shape[0] != n_before:
        raise RuntimeError("Row count changed after macro merge (should not happen with validate='m:1').")

    # 5) Save processed file (macros unscaled, characteristics unstandardized)
    out_path = PROC_DIR / OUTPUT_FILE
    jkp.to_parquet(out_path, index=False)
    logger.info("Saved processed data to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    main()
