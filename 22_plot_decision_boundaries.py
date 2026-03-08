import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# --- Academic Plotting Configuration ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 200, 'lines.linewidth': 1.2})

# Paths
ANOM_RAW = "reports/tables/anomalies_2022.csv"
THRESHOLDS_JSON = "artifacts/thresholds.json"
EVENTS_CSV = "reports/tables/anomaly_events_2022.csv"
OUT_DIR = "reports/figures/decision_boundaries"

def load_thresholds():
    """Loads the hour-bucketed thresholds from the JSON artifact."""
    if not os.path.exists(THRESHOLDS_JSON):
        print(f"[ERROR] Thresholds file not found at {THRESHOLDS_JSON}")
        return None
    with open(THRESHOLDS_JSON, 'r') as f:
        return json.load(f)

def get_threshold_series(df, signal, thresholds_dict):
    """
    Maps hour-dependent thresholds to a time-series series.
    Matches the 'bucket_by': 'hour' logic.
    """
    target_thr = thresholds_dict['targets'][signal]
    # Extract hour from index
    hours = df.index.hour.astype(str)
    return hours.map(target_thr).astype(float)

def plot_decision_boundaries():
    """Generates visualizations of the Conformal Prediction uncertainty envelopes."""
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Load Data
    if not os.path.exists(ANOM_RAW):
        print(f"[ERROR] Anomaly data not found at {ANOM_RAW}")
        return

    df = pd.read_csv(ANOM_RAW, index_col=0, parse_dates=True)
    if df.index.tz is None: 
        df.index = df.index.tz_localize('UTC')
    
    events = pd.read_csv(EVENTS_CSV)
    thresholds_dict = load_thresholds()
    if not thresholds_dict: return
    
    signals = ["CF_Solar", "CF_Wind", "Load_MW", "Price"]
    
    # --- Part A: Event-Specific Prediction Bands ---
    for signal in signals:
        # Sort and take top events to visualize the "Uncertainty Envelope"
        sig_events = events[events['signal_id'] == signal].sort_values(by="peak_score", ascending=False).head(2)
        
        for i, (_, event) in enumerate(sig_events.iterrows()):
            start = pd.to_datetime(event["start_ts"], utc=True) - pd.Timedelta(hours=4)
            end = pd.to_datetime(event["end_ts"], utc=True) + pd.Timedelta(hours=4)
            
            window = df.loc[start:end].copy()
            if window.empty: continue
            
            # Retrieve hour-dependent threshold for this window
            thr_series = get_threshold_series(window, signal, thresholds_dict)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot Actual vs Prediction
            ax.plot(window.index, window[f"{signal}_true"], 'k-', label="Actual Data", alpha=0.8)
            ax.plot(window.index, window[f"{signal}_pred"], 'b--', label="dCeNN-ELM Prediction")
            
            # Shade the Conformal Uncertainty Envelope
            # Lower/Upper Bound = Pred +/- Threshold
            upper_bound = window[f"{signal}_pred"] + thr_series
            lower_bound = window[f"{signal}_pred"] - thr_series
            
            ax.fill_between(window.index, lower_bound, upper_bound, 
                             color='blue', alpha=0.15, label=f"Uncertainty Envelope (α={thresholds_dict['meta']['alpha']})")
            
            # Highlight points that are anomalies (Actual outside the envelope)
            anoms = window[window[f"{signal}_anom"] == 1]
            if not anoms.empty:
                ax.scatter(anoms.index, anoms[f"{signal}_true"], color='red', s=25, label="Detected Anomaly", zorder=5)

            ax.set_title(f"Decision Boundary: {signal} Anomaly Event\n(Shaded area = Conformal Neural Layer Tolerance)")
            ax.set_ylabel(signal.replace("_", " "))
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{OUT_DIR}/boundary_{signal}_{i+1}.png")
            plt.close()

    # --- Part B: Global Residual Violin Plot ---
    # Standardized comparison of error distributions across signals
    plt.figure(figsize=(12, 6))
    melt_data = []
    for sig in signals:
        residuals = (df[f"{sig}_true"] - df[f"{sig}_pred"]).dropna()
        # Z-score residuals for standardized visualization
        z_residuals = (residuals - residuals.mean()) / (residuals.std() + 1e-9)
        for val in z_residuals.sample(min(2000, len(z_residuals))):
            melt_data.append({"Signal": sig, "Standardized Residual": val})
            
    df_violin = pd.DataFrame(melt_data)
    
    # Corrected sns.violinplot to resolve FutureWarning
    sns.violinplot(data=df_violin, x="Signal", y="Standardized Residual", 
                   hue="Signal", palette="muted", inner="quart", legend=False)
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Statistical Rigor: Distribution of Prediction Residuals (Z-Scored)")
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(f"{OUT_DIR}/global_residual_violin.png")
    plt.close()
    
    print(f"[INFO] Decision boundary plots saved to {OUT_DIR}")

if __name__ == "__main__":
    plot_decision_boundaries()