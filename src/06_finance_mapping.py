# src/06_finance_mapping.py

#!/usr/bin/env python3
"""
Finance-aware backtest over detected anomalies.
Supports both raw ML output (wide) and ASP-refined output (long).
"""

import argparse
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

ANOM_CSV_DEFAULT = "reports/tables/anomalies_2022.csv"
OUT_CSV_DEFAULT = "reports/tables/finance_backtest_2022.csv"
OUT_PNG_DEFAULT = "reports/figures/utility_vs_time_2022.png"

CFG_DEFAULT: Dict[str, float] = {
    "price_up_thresh": 0.0,   
    "anomaly_bonus": 5.0,     
    "imbalance_penalty": 2.0, 
    "position_size": 1.0,     
    "base_position": 0.0,
}

def _fail(msg: str) -> None:
    import sys
    sys.stderr.write(f"[ERROR] {msg}\n")
    sys.exit(1)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finance backtest.")
    parser.add_argument("--anoms-csv", default=ANOM_CSV_DEFAULT)
    parser.add_argument("--out-csv", default=OUT_CSV_DEFAULT)
    parser.add_argument("--out-png", default=OUT_PNG_DEFAULT)
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    cfg = CFG_DEFAULT.copy()

    if not os.path.exists(args.anoms_csv):
        _fail(f"Input file not found: {args.anoms_csv}")

    df_input = pd.read_csv(args.anoms_csv, parse_dates=["Time (UTC)"])
    
    # ------------------------------------------------------
    # FORMAT DETECTION & PIVOT
    # ------------------------------------------------------
    # If 'target' is present, it's the "Long" format from ASP Step 07
    if "target" in df_input.columns:
        print(f"[INFO] Detected refined (long-format) data. Pivoting...")
        
        # 1. Pivot the flags (0 or 1)
        df_flags = df_input.pivot(index="Time (UTC)", columns="target", values="final_flag").fillna(0)
        df_flags.columns = [f"{c}_anom" for c in df_flags.columns]
        
        # 2. Pivot the scores
        df_scores = df_input.pivot(index="Time (UTC)", columns="target", values="anomaly_score").fillna(0)
        # We take the max score across targets for the 'combined_score'
        df_flags["combined_score"] = df_scores.max(axis=1)

        # 3. Join with original values from the raw anomaly file
        if not os.path.exists(ANOM_CSV_DEFAULT):
            _fail(f"Original anomalies file '{ANOM_CSV_DEFAULT}' is required to get Price/Load values.")
        
        df_orig = pd.read_csv(ANOM_CSV_DEFAULT, parse_dates=["Time (UTC)"]).set_index("Time (UTC)")
        # Keep only the columns we need from the original
        val_cols = [c for c in df_orig.columns if "_true" in c or "_pred" in c]
        df = df_flags.join(df_orig[val_cols]).sort_index()
    else:
        df = df_input.set_index("Time (UTC)").sort_index()

    # Required columns check
    required = ["Price_true", "Price_pred", "combined_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        _fail(f"Missing required columns in {args.anoms_csv}: {missing}")

    # Logic: Predicted Price Delta
    df["dPrice_pred"] = df["Price_pred"].diff().fillna(0.0)
    df["dPrice_true"] = df["Price_true"].diff().fillna(0.0)

    # ------------------------------------------------------
    # POLICY: Only trade if ASP confirmed a Price anomaly
    # ------------------------------------------------------
    cond_buy = (df.get("Price_anom", 0) == 1) & (df["dPrice_pred"] > cfg["price_up_thresh"])
    cond_sell = (df.get("Price_anom", 0) == 1) & (df["dPrice_pred"] < -cfg["price_up_thresh"])

    df["action"] = cfg["base_position"]
    df.loc[cond_buy, "action"] = cfg["position_size"]
    df.loc[cond_sell, "action"] = -cfg["position_size"]

    # Imbalance Penalty
    wrong_dir = (np.sign(df["dPrice_true"]) != np.sign(df["dPrice_pred"])) & (df["action"] != 0)
    df["imbalance"] = wrong_dir.astype(float) * cfg["imbalance_penalty"]

    # Utility Calculation
    df["utility"] = (-df["Price_true"] * df["action"]) + (cfg["anomaly_bonus"] * df["combined_score"]) - df["imbalance"]
    df["utility_cum"] = df["utility"].cumsum()

    # Save and Plot
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv)
    
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["utility_cum"], label="Neuro-Symbolic Policy")
    plt.title("Cumulative Utility Backtest")
    plt.legend()
    plt.savefig(args.out_png)
    print(f"[INFO] Finance backtest complete. Saved to {args.out_csv}")

if __name__ == "__main__":
    main()