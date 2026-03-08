# src/11_plot_master_timeline.py

#!/usr/bin/env python3
"""
Master timeline visualization for Neuro-Symbolic Grid Anomaly Detection.

This version contrasts ML-suggested anomalies (Neural) against 
ASP-confirmed anomalies (Symbolic) and plots updated Finance PnL.

Panels:
1) Actual vs predicted (dCeNN-ELM)
2) Residuals
3) Neuro-Symbolic Flag Contrast (Neural vs Symbolic)
4) Finance PnL (Refined Strategy)

ASP-refined events are shaded as vertical bands, coloured by severity.
"""

import argparse
import sys
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Use non-interactive backend for headless/server environments
plt.switch_backend("Agg")

# Default IO paths matching the repository structure
DEFAULT_ANOM_RAW = "reports/tables/anomalies_2022.csv"
DEFAULT_ANOM_REF = "reports/tables/anomalies_refined.csv"
DEFAULT_EVENTS = "reports/tables/anomaly_events_2022.csv"
DEFAULT_FINANCE = "reports/tables/finance_backtest_2022.csv"

# ------------------------ utilities ------------------------

def _fail(msg: str) -> None:
    sys.stderr.write(f"[ERROR] {msg}\n")
    sys.exit(1)

def _warn(msg: str) -> None:
    sys.stderr.write(f"[WARN] {msg}\n")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a Neuro-Symbolic master timeline for one signal."
    )
    parser.add_argument(
        "--signal",
        default="Price",
        choices=["CF_Solar", "CF_Wind", "Load_MW", "Price"],
        help="Target to visualize. Default: Price."
    )
    parser.add_argument(
        "--anomalies-raw",
        default=DEFAULT_ANOM_RAW,
        help=f"Point-level raw anomalies CSV (default: {DEFAULT_ANOM_RAW})."
    )
    parser.add_argument(
        "--anomalies-refined",
        default=DEFAULT_ANOM_REF,
        help=f"ASP refined anomalies CSV (default: {DEFAULT_ANOM_REF})."
    )
    parser.add_argument(
        "--events",
        default=DEFAULT_EVENTS,
        help=f"Event-level anomalies CSV (default: {DEFAULT_EVENTS})."
    )
    parser.add_argument(
        "--finance",
        default=DEFAULT_FINANCE,
        help=f"Finance backtest CSV (default: {DEFAULT_FINANCE})."
    )
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--score-threshold", type=float, default=1.0)
    return parser.parse_args()

# ------------------------ data loaders ------------------------

def load_data_repo(args, signal):
    """Load and slice all relevant dataframes."""
    try:
        # 1. Load Raw Anomaly Data (Neural results)
        df_raw = pd.read_csv(args.anomalies_raw, parse_dates=["Time (UTC)"]).set_index("Time (UTC)").sort_index()
        if df_raw.index.tz is None:
            df_raw.index = df_raw.index.tz_localize('UTC')
        
        # 2. Load Refined Anomaly Data (Symbolic results)
        df_ref_long = pd.read_csv(args.anomalies_refined, parse_dates=["Time (UTC)"])
        df_ref_long["Time (UTC)"] = pd.to_datetime(df_ref_long["Time (UTC)"], utc=True)
        
        # Filter for confirmed anomalies ONLY (is_vetoed == 0) and match lowercase target
        sig_lower = signal.lower()
        df_ref = df_ref_long[(df_ref_long["target"] == sig_lower) & 
                             (df_ref_long["is_vetoed"] == 0)].set_index("Time (UTC)").sort_index()
        
        # 3. Load Events (for shading)
        df_events = pd.read_csv(args.events, parse_dates=["start_ts", "end_ts"])
        df_events["start_ts"] = pd.to_datetime(df_events["start_ts"], utc=True)
        df_events["end_ts"] = pd.to_datetime(df_events["end_ts"], utc=True)
        df_events = df_events[df_events["signal_id"] == signal]
        
        # 4. Load Finance (Refined Utility)
        df_fin = pd.read_csv(args.finance, parse_dates=["Time (UTC)"]).set_index("Time (UTC)").sort_index()
        if df_fin.index.tz is None:
            df_fin.index = df_fin.index.tz_localize('UTC')
        
        # Slicing time window
        start_ts = pd.to_datetime(args.start, utc=True) if args.start else None
        end_ts = pd.to_datetime(args.end, utc=True) if args.end else None
        
        if start_ts:
            df_raw = df_raw.loc[start_ts:]
            df_fin = df_fin.loc[start_ts:]
            df_events = df_events[df_events["start_ts"] >= start_ts]
        if end_ts:
            df_raw = df_raw.loc[:end_ts]
            df_fin = df_fin.loc[:end_ts]
            df_events = df_events[df_events["end_ts"] <= end_ts]
            
        return df_raw, df_ref, df_events, df_fin
    except Exception as exc:
        _fail(f"Failed to load project CSVs: {exc}")

# ------------------------ plotting ------------------------

def plot_master_timeline(df_raw, df_ref, df_events, df_fin, signal, args):
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    
    # Panel 1: Actual vs Predicted
    axes[0].plot(df_raw.index, df_raw[f"{signal}_true"], label="Actual", color='black', alpha=0.6)
    axes[0].plot(df_raw.index, df_raw[f"{signal}_pred"], label="Predicted (dCeNN-ELM)", color='blue', linestyle="--")
    axes[0].set_ylabel(f"{signal}")
    axes[0].legend(loc="upper left")
    axes[0].set_title(f"Master Timeline Analysis: {signal}")

    # Panel 2: Residuals
    res = df_raw[f"{signal}_true"] - df_raw[f"{signal}_pred"]
    axes[1].fill_between(df_raw.index, 0, res, color='gray', alpha=0.3, label="Residual")
    axes[1].set_ylabel("Error")
    axes[1].legend(loc="upper left")

    # Panel 3: NEURO-SYMBOLIC FLAG CONTRAST
    # 1. Neural Flag (Orange) - Statistical detection
    anom_col = f"{signal}_anom"
    if anom_col in df_raw.columns:
        axes[2].fill_between(df_raw.index, 0, df_raw[anom_col], color='orange', alpha=0.3, label="Neural Flag (ML)")
    
    # 2. Symbolic Flag (Green) - ASP Confirmed Only
    # Use valid_mask to align refined timestamps with the full raw index
    valid_mask = df_raw.index.isin(df_ref.index).astype(int)
    axes[2].fill_between(df_raw.index, 0, valid_mask, color='green', alpha=0.7, label="Symbolic Flag (ASP-Confirmed)")
    
    axes[2].set_ylabel("Anomalies")
    axes[2].legend(loc="upper left")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Normal", "Anomaly"])

    # Panel 4: Financial Impact (Refined)
    if "utility_cum" in df_fin.columns:
        axes[3].plot(df_fin.index, df_fin["utility_cum"], label="Strategy PnL (ASP-Verified)", color="green")
        axes[3].axhline(0, color="black", linestyle="--", alpha=0.5)
        axes[3].set_ylabel("Cum. Utility (€)")
        axes[3].legend(loc="upper left")

    # Shade Events by Severity
    sev_colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green", "UNKNOWN": "gray"}
    for _, ev in df_events.iterrows():
        color = sev_colors.get(ev["severity_label"], "gray")
        for ax in axes:
            ax.axvspan(ev["start_ts"], ev["end_ts"], color=color, alpha=0.15)

    plt.tight_layout()
    
    # Save output
    out_path = args.output or f"reports/figures/master_timeline_{signal}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved master timeline to '{out_path}'")

def main():
    args = _parse_args()
    df_raw, df_ref, df_events, df_fin = load_data_repo(args, args.signal)
    plot_master_timeline(df_raw, df_ref, df_events, df_fin, args.signal, args)

if __name__ == "__main__":
    main()