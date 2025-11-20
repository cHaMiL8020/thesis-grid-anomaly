# src/06_finance_mapping.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Inputs
ANOM_CSV = "reports/tables/anomalies_2022.csv"
OUT_CSV  = "reports/tables/finance_backtest_2022.csv"
OUT_PNG  = "reports/figures/utility_vs_time_2022.png"

# Hyperparameters (tunable)
CFG = {
    "price_up_thresh":   0.0,   # act if predicted ΔPrice >= this
    "load_up_thresh":    0.0,   # act if predicted ΔLoad  >= this
    "vre_low_cf":        0.25,  # "low renewables" if both CFs below this
    "vre_high_cf":       0.45,  # "high renewables" if either CF above this
    "anomaly_bonus":     5.0,   # €/MWh per unit combined_score
    "imbalance_penalty": 2.0,   # €/MWh penalty when action contradicts realized direction
    "base_position":     0.0,   # baseline (0 = do nothing)
    "position_size":     1.0,   # magnitude of +/- action
}


def main():
    if not os.path.exists(ANOM_CSV):
        raise FileNotFoundError(f"Missing {ANOM_CSV}. Run `make detect` first.")

    df = (
        pd.read_csv(ANOM_CSV, parse_dates=["Time (UTC)"])
          .set_index("Time (UTC)")
          .sort_index()
    )

    # Required columns from anomaly detection
    required = [
        "Price_true", "Price_pred",
        "Load_MW_true", "Load_MW_pred",
        "CF_Solar_pred", "CF_Wind_pred",
        "combined_score",
    ]

    # Backward compatibility: if CF preds missing (older script), use truth or zeros
    for alt in ["CF_Solar_pred", "CF_Wind_pred"]:
        if alt not in df.columns:
            truth_name = alt.replace("_pred", "_true")
            df[alt] = df[truth_name] if truth_name in df.columns else 0.0

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {ANOM_CSV}: {missing}")

    # Predicted deltas
    df["dPrice_pred"] = df["Price_pred"].diff().fillna(0.0)
    df["dLoad_pred"]  = df["Load_MW_pred"].diff().fillna(0.0)

    # Action rules
    cond_buy_dispatch = (
        (df["dPrice_pred"] >= CFG["price_up_thresh"]) &
        (df["dLoad_pred"]  >= CFG["load_up_thresh"]) &
        (df["CF_Solar_pred"] < CFG["vre_low_cf"]) &
        (df["CF_Wind_pred"]  < CFG["vre_low_cf"])
    )

    cond_sell_defer = (
        (df["dPrice_pred"] <= -CFG["price_up_thresh"]) &
        (
            (df["CF_Solar_pred"] > CFG["vre_high_cf"]) |
            (df["CF_Wind_pred"]  > CFG["vre_high_cf"])
        )
    )

    # +1 buy/dispatch, -1 sell/defer, 0 hold
    df["action"] = np.where(
        cond_buy_dispatch,
        +CFG["position_size"],
        np.where(cond_sell_defer, -CFG["position_size"], CFG["base_position"]),
    )

    # Realized direction
    df["dPrice_true"] = df["Price_true"].diff().fillna(0.0)
    wrong_dir = (
        (np.sign(df["dPrice_true"]) != np.sign(df["dPrice_pred"])) &
        (np.abs(df["dPrice_true"]) > 0)
    )
    imbalance = CFG["imbalance_penalty"] * wrong_dir.astype(float)

    # Utility model
    df["utility"] = (
        - df["Price_true"] * df["action"]   # cost/revenue
        + CFG["anomaly_bonus"] * df["combined_score"]  # anomaly bonus
        - imbalance                                    # imbalance penalty
    )

    df["utility_baseline"] = 0.0
    df["utility_diff"] = df["utility"] - df["utility_baseline"]

    out_cols = [
        "Price_true", "Price_pred",
        "dPrice_true", "dPrice_pred",
        "Load_MW_true", "Load_MW_pred",
        "CF_Solar_pred", "CF_Wind_pred",
        "combined_score",
        "action",
        "utility", "utility_baseline", "utility_diff",
    ]

    os.makedirs("reports/tables", exist_ok=True)
    df[out_cols].to_csv(OUT_CSV)
    print(f"Saved {OUT_CSV}")

    # Plot cumulative utility vs time
    os.makedirs("reports/figures", exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["utility"].cumsum(), label="Policy")
    plt.plot(df.index, df["utility_baseline"].cumsum(), label="Baseline")
    plt.title("Cumulative utility — 2022")
    plt.ylabel("€ (relative units)")
    plt.xlabel("time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
