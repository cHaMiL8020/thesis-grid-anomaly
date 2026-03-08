import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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

    # 1. Load Data
    if not os.path.exists(ANOM_RAW) or not os.path.exists(ANOM_REF):
        print(f"[ERROR] Required tables not found. Run detection and ASP steps first.")
        return

    # Load raw data and ensure index is UTC datetime
    df_raw = pd.read_csv(ANOM_RAW, index_col=0, parse_dates=True)
    if df_raw.index.tz is None:
        df_raw.index = df_raw.index.tz_localize('UTC')

    # Load refined data and ensure Time is UTC datetime
    df_ref = pd.read_csv(ANOM_REF, parse_dates=["Time (UTC)"])
    df_ref["Time (UTC)"] = pd.to_datetime(df_ref["Time (UTC)"], utc=True)
    
    events = pd.read_csv(EVENTS_CSV)
    
    # Corrected Signal Mapping to match your pipeline's CSV headers
    # These names must match the columns in anomalies_2022.csv (e.g. Load_MW_true)
    signals = ["CF_Solar", "CF_Wind", "Load_MW", "Price"]
    
    # 2. Iterate through top events for each signal
    for signal in signals:
        # Filter events for this specific signal
        sig_events = events[events['signal_id'] == signal].sort_values(by="peak_score", ascending=False).head(3)
        
        if sig_events.empty:
            print(f"[INFO] No events found for {signal}")
            continue

        for i, (_, event) in enumerate(sig_events.iterrows()):
            # Define window with 6-hour padding
            start = pd.to_datetime(event["start_ts"], utc=True) - pd.Timedelta(hours=6)
            end = pd.to_datetime(event["end_ts"], utc=True) + pd.Timedelta(hours=6)
            
            # Slice the data for the window
            window = df_raw.loc[start:end]
            if window.empty: continue

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                          gridspec_kw={'height_ratios': [3, 1]})

            # --- Panel 1: Prediction Comparison ---
            ax1.plot(window.index, window[f"{signal}_true"], 'k-', label="Actual Data", alpha=0.9)
            ax1.plot(window.index, window[f"{signal}_pred"], 'g--', label="Proposed (dCeNN-ELM)")
            
            # Plot benchmarks if they exist (LSTM/Ridge)
            for b_mark in ['LSTM', 'Ridge']:
                col = f"{signal}_{b_mark}_pred"
                if col in window.columns:
                    ax1.plot(window.index, window[col], label=f"Baseline ({b_mark})", alpha=0.6)

            ax1.set_ylabel(signal.replace("_", " "))
            ax1.set_title(f"Event Comparison: {signal} (Severity: {event['severity_label']})")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)

            # --- Panel 2: Neuro-Symbolic Flag Contrast ---
            # A) Raw Neural Anomaly Flag (Orange)
            anom_col = f"{signal}_anom"
            if anom_col in window.columns:
                ax2.fill_between(window.index, 0, window[anom_col], 
                                 color='orange', alpha=0.3, label="Neural Flag (ML)")
            
            # B) ASP Refined Flag (Green)
            # Match lowercase target (e.g. 'load_mw') and filter for confirmed anomalies (is_vetoed == 0)
            target_slug = signal.lower()
            ref_win = df_ref[(df_ref['Time (UTC)'] >= start) & 
                             (df_ref['Time (UTC)'] <= end) & 
                             (df_ref['target'] == target_slug) &
                             (df_ref['is_vetoed'] == 0)]

            if not ref_win.empty:
                # Plot the confirmed flag (height 1)
                ax2.fill_between(ref_win['Time (UTC)'], 0, 1, 
                                 color='green', alpha=0.7, label="Symbolic Flag (ASP Confirmed)")

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