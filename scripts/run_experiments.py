from __future__ import annotations

"""Run GKX-style ML experiments for return prediction.

Usage:
    python scripts/run_experiments.py

Expects:
    - Processed panel at data/processed/JKP_US_processed.parquet
    - Factor_Details.xlsx in data/raw/ (if using JKP subset)

Outputs:
    Timestamped results saved to data/processed/results/
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.schemas import FeatureConfig, TrainConfig

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

MACRO_INTERACTIONS = False
USE_JKP_CHARACTERISTICS = False
SIZE_FILTER_QUANTILE = 0.50

TRAIN_YEARS = 25
VAL_YEARS = 5
TEST_START_YEAR = 2000
TEST_END_YEAR = None

TRAIN_CFG = TrainConfig(
    train_years=TRAIN_YEARS,
    val_years=VAL_YEARS,
    test_start_year=TEST_START_YEAR,
    test_end_year=TEST_END_YEAR,
    seed=0,
    patience=10,
    min_delta=1e-4,
    use_plateau_scheduler=True,
)


def load_jkp_characteristics(raw_dir: Path) -> list[str]:
    """Load JKP characteristic list from Factor_Details.xlsx."""
    path = raw_dir / "Factor_Details.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Factor_Details.xlsx not found at {path}")

    df = pd.read_excel(path)
    if "abr_jkp" not in df.columns:
        raise KeyError("Column 'abr_jkp' not found")

    chars = df.loc[df["abr_jkp"].notna(), "abr_jkp"].astype(str).unique().tolist()
    print(f"Loaded {len(chars)} JKP characteristics")
    return chars


def main() -> None:
    from src.config import get_processed_dir, get_raw_dir, get_results_dir
    from src.preprocessing import make_feature_builder, prepare_features_and_target
    # print_summary_table now in results.py
    from src.results import (
        print_summary_table,
        make_run_id,
        save_overall_json,
        save_per_year,
        save_run_metadata,
        save_summary,
    )
    from src.training import run_gkx_mlp, run_gkx_sklearn_grid_with_features

    raw_dir = get_raw_dir()
    proc_dir = get_processed_dir()
    results_dir = get_results_dir()

    feature_set_name = (
        f"{'jkp_chars' if USE_JKP_CHARACTERISTICS else 'all_chars'}__"
        f"mktcap_above_p{int(SIZE_FILTER_QUANTILE*100)}__"
        f"{'macro_int' if MACRO_INTERACTIONS else 'no_macro_int'}"
    )

    print("Project root:", REPO_ROOT)
    print("Raw dir     :", raw_dir)
    print("Processed   :", proc_dir)
    print("Results     :", results_dir)

    # -----------------------------------------------------------------------
    # Load processed data
    # -----------------------------------------------------------------------
    processed_path = proc_dir / "JKP_US_processed.parquet"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed panel not found: {processed_path}\n"
            "Run scripts/prepare_data.py first."
        )

    df = pd.read_parquet(processed_path)
    print(f"\nLoaded: {len(df):,} rows, {len(df.columns)} columns")

    # -------------------------------------------------------------------
    # Filter by size
    # -------------------------------------------------------------------
    if SIZE_FILTER_QUANTILE > 0:
        cutoffs = df.groupby("eom")["me"].transform(lambda x: x.quantile(SIZE_FILTER_QUANTILE))
        df = df[df["me"] >= cutoffs].copy()
        print(f"After ME >= p{SIZE_FILTER_QUANTILE:.0%}: {len(df):,} rows")
        
    # -------------------------------------------------------------------
    # Identify macro / characteristic columns
    # -------------------------------------------------------------------
    macro_csv = raw_dir / "macro_predictors.csv"
    if not macro_csv.exists():
        raise FileNotFoundError(
            f"macro_predictors.csv not found at {macro_csv}. "
            "Place it in data/raw."
        )

    macro_sample = pd.read_csv(macro_csv, nrows=1)
    macro_cols = [c for c in macro_sample.columns if c != "date"]

    id_vars = {
        "id", "permno", "permco", "gvkey", "iid",
        "date", "eom",
        "curcd", "fx", "excntry",
        "me", "size_grp", "me_company",
        "common", "exch_main", "primary_sec", "obs_main",
        "comp_exchg", "comp_tpci",
        "source_crsp", "crsp_exchcd", "crsp_shrcd",
        "ff49", "sic", "naics", "gics",
    }

    ret_vars = {
        "ret_exc_lead1m", "ret_exc", "ret", "ret_lead1m",
        "ret_local", "ret_lag_dif",
        "prc_local", "prc_high", "prc_low",
        "bidask", "dolvol",
        "shares", "tvol", "adjfct",
    }

    excluded = id_vars | ret_vars | set(macro_cols)
    char_cols_all = sorted(set(df.columns) - excluded)

    # -----------------------------------------------------------------------
    # Select characteristics
    # -----------------------------------------------------------------------
    if USE_JKP_CHARACTERISTICS:
        jkp_chars = load_jkp_characteristics(raw_dir)
        char_cols = [c for c in char_cols_all if c in jkp_chars]
        print(f"Using {len(char_cols)}/{len(jkp_chars)} JKP characteristics")
    else:
        char_cols = char_cols_all
        print(f"Using all {len(char_cols)} characteristics")

    print(f"Macro predictors: {len(macro_cols)}")

    # -----------------------------------------------------------------------
    # Prepare features
    # -----------------------------------------------------------------------
    feature_cols = char_cols + macro_cols if MACRO_INTERACTIONS else char_cols
    
    X, y, df_clean = prepare_features_and_target(
        df,
        feature_cols=feature_cols,
        target_col = "ret_exc_lead1m"
    )
    
    assert len(df_clean) == X.shape[0] == y.shape[0], "X/y/df_clean misalignment"

    print(f"\nFeature matrix: {X.shape}")
    print(f"Target: {y.shape}")
    
    min_date = df_clean["eom"].min()
    max_date = df_clean["eom"].max()
    n_obs = len(df_clean)
    n_permno = df_clean["permno"].nunique() if "permno" in df_clean.columns else None

    feat_cfg = FeatureConfig(
        macro_cols=macro_cols,
        char_cols=char_cols,
        interactions=MACRO_INTERACTIONS,
        ff49_col="ff49",
        date_col=TRAIN_CFG.date_col,
    )

    feature_builder = make_feature_builder(
        X_base=X,
        y=y,
        df_aligned=df_clean,
        X_columns=list(X.columns),
        cfg=feat_cfg,
    )

    print("\nConfiguration:")
    print(f"  Feature set : {feature_set_name}")
    print(f"  ME filter   : p{SIZE_FILTER_QUANTILE:.0%}")
    print(f"  Chars       : {len(char_cols)}")
    print(f"  Macro       : {len(macro_cols)}")
    print(f"  Interactions: {MACRO_INTERACTIONS}")
    print(f"  Split       : train={TRAIN_CFG.train_years}y, val={TRAIN_CFG.val_years}y, test from {TRAIN_CFG.test_start_year}")

    # -----------------------------------------------------------------------
    # Run models
    # -----------------------------------------------------------------------
    run_id = make_run_id()
    rows = []

    def record(model_name: str, res: dict) -> None:
        rows.append({"model": model_name, "global_r2": res["global_r2"]})
        save_per_year(res["per_year"], model_name=model_name, feature_set=feature_set_name, run_id=run_id)
        save_overall_json(res["overall"], model_name=model_name, feature_set=feature_set_name, run_id=run_id)

    # OLS
    res_ols = run_gkx_sklearn_grid_with_features(
        df=df_clean,
        feature_builder=feature_builder,
        train=TRAIN_CFG,
        base_estimator=LinearRegression(),
        tune_hyperparams=False,
        estimator_name="OLS",
    )
    record("OLS", res_ols)

    # Elastic Net
    res_enet = run_gkx_sklearn_grid_with_features(
        df=df_clean,
        feature_builder=feature_builder,
        train=TRAIN_CFG,
        base_estimator=ElasticNet(max_iter=8000, tol=1e-3),
        param_grid={"alpha": [1e-3, 1e-2, 1e-1], "l1_ratio": [0.7, 0.9, 1.0]},
        tune_hyperparams=True,
        estimator_name="ENet",
    )
    record("ENet", res_enet)

    # PCR
    pipeline = Pipeline([
        ("pca", PCA(svd_solver="full", random_state=0)),
        ("reg", LinearRegression()),
    ])
    res_pcr = run_gkx_sklearn_grid_with_features(
        df=df_clean,
        feature_builder=feature_builder,
        train=TRAIN_CFG,
        base_estimator=pipeline,
        param_grid={"pca__n_components": [10, 25, 50, 75, 100]},
        tune_hyperparams=True,
        estimator_name="PCR",
    )
    record("PCR", res_pcr)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=500,
        max_features=0.03,
        min_samples_leaf=200,
        max_depth=10,
        max_samples=0.8,
        bootstrap=True,
        random_state=0,
        n_jobs=8,
    )
    res_rf = run_gkx_sklearn_grid_with_features(
        df=df_clean,
        feature_builder=feature_builder,
        train=TRAIN_CFG,
        base_estimator=rf,
        tune_hyperparams=False,
        estimator_name="RF_fixed",
    )
    record("RF_fixed", res_rf)

    # MLP
    res_mlp = run_gkx_mlp(
        df=df_clean,
        feature_builder=feature_builder,
        train=TRAIN_CFG,
        n_epochs=200,
        batch_size=10000,
        lr=1e-3,
        weight_decay=0.0,
        hidden_dims=(32, 16, 8),
        dropout=0.0,
        use_batchnorm=True,
        loss="mse",
        patience=5,
        min_delta=1e-5,
        ensemble_n=5,
        deterministic=True,
        l1_lambda=1e-3,
    )
    record("MLP", res_mlp)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    summary_df = pd.DataFrame(rows).sort_values("global_r2", ascending=False).reset_index(drop=True)
    summary_df.insert(0, "feature_set", feature_set_name)

    summary_df["train_years"] = TRAIN_CFG.train_years
    summary_df["val_years"] = TRAIN_CFG.val_years
    summary_df["test_start_year"] = TRAIN_CFG.test_start_year
    if TRAIN_CFG.test_end_year:
        summary_df["test_end_year"] = TRAIN_CFG.test_end_year

    summary_df["size_filter_quantile"] = SIZE_FILTER_QUANTILE
    summary_df["use_jkp_characteristics"] = USE_JKP_CHARACTERISTICS
    summary_df["n_chars"] = len(char_cols)
    summary_df["n_macro"] = len(macro_cols)
    summary_df["macro_interactions"] = MACRO_INTERACTIONS
    
    summary_df["n_obs"] = n_obs
    summary_df["min_eom"] = str(min_date)
    summary_df["max_eom"] = str(max_date)
    if n_permno is not None:
        summary_df["n_permno"] = n_permno


    summary_path = save_summary(summary_df, feature_set=feature_set_name, run_id=run_id)

    metadata = {
        "run_id": run_id,
        "feature_set": feature_set_name,
        "script": "run_experiments.py",
        "config": {
            "train_years": TRAIN_CFG.train_years,
            "val_years": TRAIN_CFG.val_years,
            "test_start_year": TRAIN_CFG.test_start_year,
            "size_filter_quantile": SIZE_FILTER_QUANTILE,
            "macro_interactions": MACRO_INTERACTIONS,
            "use_jkp_characteristics": USE_JKP_CHARACTERISTICS,
            "n_chars": len(char_cols),
            "n_macro": len(macro_cols),
            "test_end_year": TRAIN_CFG.test_end_year,
            "n_obs": n_obs,
            "min_eom": str(min_date),
            "max_eom": str(max_date),
            "n_permno": n_permno,
        },
    }
    meta_path = save_run_metadata(metadata, feature_set=feature_set_name, run_id=run_id)

    print_summary_table(
            summary_df,
            title="\n" + "="*60 + "\nResults\n" + "="*60,
            common_cols=[
                "feature_set",
                "train_years",
                "val_years",
                "test_start_year",
                "size_filter_quantile",
                "use_jkp_characteristics",
                "n_chars",
                "n_macro",
                "macro_interactions",
            ],
            value_cols=["model", "global_r2"],
        )
    
    print(f"\nSaved: {summary_path.name}")
    print(f"Saved: {meta_path.name}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
