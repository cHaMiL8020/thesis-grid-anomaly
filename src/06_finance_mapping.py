# src/06_finance_mapping.py

#!/usr/bin/env python3
"""
Finance-aware backtest over detected anomalies (2022).

Reads:
  - reports/tables/anomalies_2022.csv  (from step 05)

Computes:
  - A simple policy: buy/dispatch vs sell/defer based on:
      * predicted ΔPrice
      * predicted ΔLoad
      * predicted CF_Solar / CF_Wind (low/high VRE)
      * anomaly combined_score
  - Utility per timestep and cumulative utility vs baseline.

Writes:
  - reports/tables/finance_backtest_2022.csv
  - reports/figures/utility_vs_time_2022.png
"""

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Non-interactive backend for headless environments
plt.switch_backend("Agg")

# Default IO paths (can be overridden via CLI)
ANOM_CSV_DEFAULT = "reports/tables/anomalies_2022.csv"
OUT_CSV_DEFAULT = "reports/tables/finance_backtest_2022.csv"
OUT_PNG_DEFAULT = "reports/figures/utility_vs_time_2022.png"

# Hyperparameters (tunable)
CFG_DEFAULT: Dict[str, float] = {
    "price_up_thresh": 0.0,   # act if predicted ΔPrice >= this
    "load_up_thresh": 0.0,    # act if predicted ΔLoad  >= this
    "vre_low_cf": 0.25,       # "low renewables" if both CFs below this
    "vre_high_cf": 0.45,      # "high renewables" if either CF above this
    "anomaly_bonus": 5.0,     # €/MWh per unit combined_score
    "imbalance_penalty": 2.0, # €/MWh penalty when action contradicts realized direction
    "base_position": 0.0,     # baseline (0 = do nothing)
    "position_size": 1.0,     # magnitude of +/- action
}


# ------------------------- helpers -------------------------


def _fail(msg: str) -> None:
    """Print an error message and exit with non-zero status."""
    import sys
    sys.stderr.write(f"[ERROR] {msg}\n")
    sys.exit(1)


def _warn(msg: str) -> None:
    """Print a warning message to stderr."""
    import sys
    sys.stderr.write(f"[WARN] {msg}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finance-aware backtest over detected anomalies for 2022."
    )
    parser.add_argument(
        "--anoms-csv",
        default=ANOM_CSV_DEFAULT,
        help=f"Path to anomalies CSV (default: {ANOM_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--out-csv",
        default=OUT_CSV_DEFAULT,
        help=f"Path to output finance backtest CSV (default: {OUT_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--out-png",
        default=OUT_PNG_DEFAULT,
        help=f"Path to output utility figure (default: {OUT_PNG_DEFAULT}).",
    )
    return parser.parse_args()


def _ensure_parent_dir(path: str) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)


# ---------------------------- main ----------------------------


def main() -> None:
    args = _parse_args()
    cfg = CFG_DEFAULT.copy()

    anom_csv = args.anoms_csv
    out_csv = args.out_csv
    out_png = args.out_png

    if not os.path.exists(anom_csv):
        _fail(f"Missing anomalies CSV '{anom_csv}'. Run anomaly detection first.")

    # Load anomaly table
    df = (
        pd.read_csv(anom_csv, parse_dates=["Time (UTC)"])
        .set_index("Time (UTC)")
        .sort_index()
    )

    # Required columns from anomaly detection
    required: List[str] = [
        "Price_true",
        "Price_pred",
        "Load_MW_true",
        "Load_MW_pred",
        "CF_Solar_pred",
        "CF_Wind_pred",
        "combined_score",
    ]

    # Backward compatibility:
    # if CF preds missing (older script), use truth or zeros
    for alt in ["CF_Solar_pred", "CF_Wind_pred"]:
        if alt not in df.columns:
            truth_name = alt.replace("_pred", "_true")
            if truth_name in df.columns:
                _warn(
                    f"Column '{alt}' missing, falling back to '{truth_name}' "
                    "for finance backtest."
                )
                df[alt] = df[truth_name]
            else:
                _warn(
                    f"Columns '{alt}' and '{truth_name}' both missing; "
                    f"filling '{alt}' with zeros."
                )
                df[alt] = 0.0

    # If combined_score missing (very unlikely with current 05), try anomaly_score
    if "combined_score" not in df.columns and "anomaly_score" in df.columns:
        _warn(
            "Column 'combined_score' missing; falling back to 'anomaly_score'."
        )
        df["combined_score"] = df["anomaly_score"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        _fail(f"Missing required columns in {anom_csv}: {missing}")

    # ------------------------------------------------------
    # Predicted deltas (simple first-order differences)
    # ------------------------------------------------------
    df["dPrice_pred"] = df["Price_pred"].diff().fillna(0.0)
    df["dLoad_pred"] = df["Load_MW_pred"].diff().fillna(0.0)

    # ------------------------------------------------------
    # Action rules
    # ------------------------------------------------------
    cond_buy_dispatch = (
        (df["dPrice_pred"] >= cfg["price_up_thresh"])
        & (df["dLoad_pred"] >= cfg["load_up_thresh"])
        & (df["CF_Solar_pred"] < cfg["vre_low_cf"])
        & (df["CF_Wind_pred"] < cfg["vre_low_cf"])
    )

    cond_sell_defer = (
        (df["dPrice_pred"] <= -cfg["price_up_thresh"])
        & (
            (df["CF_Solar_pred"] > cfg["vre_high_cf"])
            | (df["CF_Wind_pred"] > cfg["vre_high_cf"])
        )
    )

    # +1 buy/dispatch, -1 sell/defer, 0 hold
    df["action"] = np.where(
        cond_buy_dispatch,
        +cfg["position_size"],
        np.where(cond_sell_defer, -cfg["position_size"], cfg["base_position"]),
    )

    # ------------------------------------------------------
    # Realized direction & imbalance penalty
    # ------------------------------------------------------
    df["dPrice_true"] = df["Price_true"].diff().fillna(0.0)

    # "Wrong direction" only matters if we actually took a position
    wrong_dir = (
        (np.sign(df["dPrice_true"]) != np.sign(df["dPrice_pred"]))
        & (np.abs(df["dPrice_true"]) > 0)
        & (df["action"] != 0.0)
    )
    df["imbalance"] = cfg["imbalance_penalty"] * wrong_dir.astype(float)

    # ------------------------------------------------------
    # Utility model
    # ------------------------------------------------------
    df["utility"] = (
        -df["Price_true"] * df["action"]  # cost/revenue
        + cfg["anomaly_bonus"] * df["combined_score"]  # anomaly bonus
        - df["imbalance"]  # imbalance penalty
    )

    df["utility_baseline"] = 0.0
    df["utility_diff"] = df["utility"] - df["utility_baseline"]
    df["utility_cum"] = df["utility"].cumsum()
    df["utility_baseline_cum"] = df["utility_baseline"].cumsum()

    out_cols = [
        "Price_true",
        "Price_pred",
        "dPrice_true",
        "dPrice_pred",
        "Load_MW_true",
        "Load_MW_pred",
        "CF_Solar_pred",
        "CF_Wind_pred",
        "combined_score",
        "action",
        "imbalance",
        "utility",
        "utility_baseline",
        "utility_diff",
        "utility_cum",
        "utility_baseline_cum",
    ]

    # ------------------------------------------------------
    # Save backtest CSV
    # ------------------------------------------------------
    _ensure_parent_dir(out_csv)
    df[out_cols].to_csv(out_csv)
    print(f"[INFO] Saved finance backtest CSV to '{out_csv}'")

    # ------------------------------------------------------
    # Plot cumulative utility vs time
    # ------------------------------------------------------
    _ensure_parent_dir(out_png)
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["utility_cum"], label="Policy")
    plt.plot(df.index, df["utility_baseline_cum"], label="Baseline")
    plt.title("Cumulative utility — 2022")
    plt.ylabel("€ (relative units)")
    plt.xlabel("time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"[INFO] Saved utility plot to '{out_png}'")
    print("[INFO] Finance mapping (06_finance_mapping.py) completed.")


if __name__ == "__main__":
    main()
