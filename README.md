## Application of machine learning methods to stock return forecasting in Python

This repository contains a reproducible research pipeline for forecasting U.S. equity returns using machine learning methods, based on "Empirical Asset Pricing via Machine Learning" by Shihao Gu, Bryan Kelly, and Dacheng Xiu (GKX). The project uses the Global Factor Data made available by Theis Ingerslev Jensen, Bryan Kelly, and Lasse Heje Pedersen, together with a set of macroeconomic predictors collected by Amit Goyal.

Data note: Raw data are not included. The equity panel is accessed via WRDS. You must have valid WRDS credentials and permissions.

## Project structure

```text
.
├── data/
│   ├── raw/                   # raw downloads (ignored by git; .gitkeep included)
│   └── processed/             # processed panel + outputs (ignored by git; .gitkeep included)
│       └── results/           # saved experiment outputs
├── scripts/
│   ├── data_download.py       # downloads raw inputs (WRDS + macro predictors)
│   ├── prepare_data.py        # builds processed parquet panel
│   └── run_experiments.py     # runs OOS forecasting experiments and saves results
└── src/
    ├── config.py              # project, data, and results paths
    ├── preprocessing.py       # feature preparation + split-safe transforms
    ├── splitting.py           # GKX-style rolling train/val/test splits
    ├── training.py            # model training and (optional) hyperparameter tuning
    ├── models.py              # model definitions 
    ├── results.py             # result saving utilities + run metadata
    ├── schemas.py             # FeatureConfig / TrainConfig dataclasses
    └── utils.py               # metrics and helper functions
```

Setup

Option A: Conda (recommended)

conda env create -f environment.yml
conda activate gkx-ml

WRDS credentials

The script scripts/data_download.py uses the wrds Python package and expects valid WRDS credentials.
Common options include:
- setting environment variables (e.g., WRDS_USERNAME), or
- using a .pgpass file or interactive login, depending on your setup.

---

Run the pipeline

1) Download raw inputs
python scripts/data_download.py

2) Build processed panel
python scripts/prepare_data.py

3) Run experiments (rolling OOS evaluation)
python scripts/run_experiments.py

Results are written to:
data/processed/results/

---

Configuration (feature set choices)

SIZE_FILTER_QUANTILE:
Cross-sectional size filter applied each month. Example: 0.50 keeps stocks with market equity (ME) above the monthly median. Use 0.0 to disable.

MACRO_INTERACTIONS:
If True, includes interactions between firm characteristics and macroeconomic predictors.

USE_JKP_CHARACTERISTICS:
If True, restricts characteristics to the JKP set loaded from data/raw/Factor_Details.xlsx.

---

Results and outputs

For each run, the following files are written to data/processed/results/:

- summary__<feature_set>__<run_id>.csv
- per_year__<model>__<feature_set>__<run_id>.csv
- overall__<model>__<feature_set>__<run_id>.json
- run_metadata__<feature_set>__<run_id>.json

---

Methodology (high level)

Walk-forward evaluation with expanding training window, fixed validation window, and rolling one-year out-of-sample tests (GKX-style).
Feature standardization is performed split-safely during model training (per-month within each train/val/test split).
Splits are based on label month (t+1) to avoid look-ahead from forward returns.

Models included:
- OLS
- Elastic Net
- Principal Component Regression
- Random Forest
- MLP

---

References

Gu, Shihao; Kelly, Bryan; Xiu, Dacheng. "Empirical Asset Pricing via Machine Learning". The Review of Financial Studies, 33(5), 2020, 2223–2273.
Jensen, Theis Ingerslev; Kelly, Bryan; Pedersen, Lasse Heje. "Is There a Replication Crisis in Finance?". The Journal of Finance, 78(5), 2023, 2465-2518.

