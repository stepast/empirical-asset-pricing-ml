"""
Download raw data inputs:
1) JKP global factor panel (USA) from WRDS
2) Macro predictors from Amit Goyal’s Web site

Run once to populate data/raw/.
"""

from pathlib import Path
import io
import logging

import numpy as np
import pandas as pd
import requests
import wrds

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().parents[0]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

def download_jkp_usa() -> pd.DataFrame:
    db = wrds.Connection()
    try:
        sql = """
            SELECT *
            FROM contrib.global_factor
            WHERE common=1
              AND exch_main=1
              AND primary_sec=1
              AND obs_main=1
              AND excntry='USA'
        """
        return db.raw_sql(sql)
    finally:
        db.close()


def download_macro_predictors(sheet_id: str, tab_name: str) -> pd.DataFrame:
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/gviz/tq?tqx=out:csv&sheet={tab_name}"
    )

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text), thousands=",")

    df = (
        df.sort_values("yyyymm")
          .assign(
              date=lambda x: pd.to_datetime(x["yyyymm"], format="%Y%m"),
              dp=lambda x: np.log(x["D12"]) - np.log(x["Index"]),
              dy=lambda x: np.log(x["D12"]) - np.log(x["Index"].shift(1)),
              ep=lambda x: np.log(x["E12"]) - np.log(x["Index"]),
              de=lambda x: np.log(x["D12"]) - np.log(x["E12"]),
              tms=lambda x: x["lty"] - x["tbl"],
              dfy=lambda x: x["BAA"] - x["AAA"],
          )
          .rename(columns={"b/m": "bm"})
          .loc[:, [
              "date", "dp", "dy", "ep", "de", "svar", "bm",
              "ntis", "tbl", "lty", "ltr", "tms", "dfy", "infl"
          ]]
          .dropna()
    )

    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Downloading JKP USA data from WRDS...")
    jkp = download_jkp_usa()
    jkp_path = RAW_DIR / "JKP_US_raw_full.parquet"
    jkp.to_parquet(jkp_path, index=False)

    logging.info("Downloading macro predictors...")
    sheet_id = "1bM7vCWd3WOt95Sf9qjLPZjoiafgF_8EG"
    tab_name = "macro_predictors"  
    macro = download_macro_predictors(sheet_id, tab_name)
    macro_path = RAW_DIR / "macro_predictors.csv"
    macro.to_csv(macro_path, index=False)

    logging.info("Done.")


if __name__ == "__main__":
    main()


