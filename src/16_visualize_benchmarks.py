import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import yaml
import importlib.util

# Academic Plotting Configuration
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 200, 'lines.linewidth': 1.5})

# File Paths
ANOM_RAW = "reports/tables/anomalies_2022.csv"
ANOM_REF = "reports/tables/anomalies_refined.csv"
EVENTS_CSV = "reports/tables/anomaly_events_2022.csv"
OUT_DIR = "reports/figures/event_comparisons"

def plot_all_signals_comparison():
    """Generates comparison plots for each target signal during detected events."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load Data (Corrected read_csv call)
    df_raw = pd.read_csv(ANOM_RAW, index_col=0, parse_dates=True)
    df_ref = pd.read_csv(ANOM_REF, parse_dates=["Time (UTC)"])
    events = pd.read_csv(EVENTS_CSV)
    
    # Target Mapping
    signals = ["CF_Solar", "CF_Wind", "Actual_Load_MW", "Price_EUR_MWh"]
    
    # 2. Iterate through top events for each signal
    for signal in signals:
        # Filter events for this specific signal
        sig_events = events[events['signal_id'] == signal].sort_values(by="peak_score", ascending=False).head(3)
        
        if sig_events.empty:
            print(f"[INFO] No events found for {signal}")
            continue

        for i, (_, event) in enumerate(sig_events.iterrows()):
            start = pd.to_datetime(event["start_ts"]) - pd.Timedelta(hours=6)
            end = pd.to_datetime(event["end_ts"]) + pd.Timedelta(hours=6)
            
            # Slice the data for the window
            window = df_raw.loc[start:end]
            if window.empty: continue

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                          gridspec_kw={'height_ratios': [3, 1]})

            # --- Panel 1: Prediction Comparison ---
            # Note: Assuming benchmarks are stored in df_raw from step 15
            ax1.plot(window.index, window[f"{signal}_true"], 'k-', label="Actual Data", alpha=0.9)
            ax1.plot(window.index, window[f"{signal}_pred"], 'g--', label="Proposed (dCeNN-ELM)")
            
            # If benchmark columns exist (from 15_run_benchmarks), plot them
            for b_mark in ['LSTM', 'Ridge']:
                col = f"{signal}_{b_mark}_pred"
                if col in window.columns:
                    ax1.plot(window.index, window[col], label=f"Baseline ({b_mark})", alpha=0.6)

            ax1.set_ylabel(signal.replace("_", " "))
            ax1.set_title(f"Event Comparison: {signal} (Severity: {event['severity_label']})")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)

            # --- Panel 2: Neuro-Symbolic Flag Contrast ---
            # Raw ML Anomaly Flag
            ax2.fill_between(window.index, 0, window[f"{signal}_anom"], color='orange', alpha=0.3, label="Neural Flag")
            
            # ASP Refined Flag (Check if this window has a refined flag)
            ref_win = df_ref[(df_ref['Time (UTC)'] >= start) & (df_ref['Time (UTC)'] <= end) & (df_ref['target'] == signal.lower())]
            if not ref_win.empty:
                ax2.fill_between(ref_win['Time (UTC)'], 0, ref_win['final_flag'], color='green', alpha=0.7, label="Symbolic (ASP) Flag")

            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(["Normal", "Anomaly"])
            ax2.set_ylim(-0.1, 1.1)
            ax2.legend(loc="upper left")

            plt.tight_layout()
            save_path = f"{OUT_DIR}/event_{signal}_{i+1}.png"
            plt.savefig(save_path)
            plt.close()
            print(f"[INFO] Saved comparison plot for {signal} to {save_path}")

if __name__ == "__main__":
    plot_all_signals_comparison()