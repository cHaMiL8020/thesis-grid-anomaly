import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- Academic Plotting Configuration ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 200})

# Paths
ANOM_RAW = "reports/tables/anomalies_2022.csv"
ANOM_REF = "reports/tables/anomalies_refined.csv"
OUT_FIG = "reports/figures/veto_reasoning_analysis.png"

def run_pre_viz_check(df_raw, df_ref, signals):
    """
    Validates data integrity before plotting to ensure the Neuro-Symbolic 
    results are consistent and the CSV files contain required columns.
    """
    print("[INFO] Running pre-visualization integrity check...")
    
    # 1. Check for required columns
    if 'is_vetoed' not in df_ref.columns:
        print("[ERROR] 'is_vetoed' column missing in anomalies_refined.csv.")
        print("[FIX] Ensure src/07_apply_asp.py is updated to save veto atoms.")
        return False

    # 2. Check for signal presence
    missing_signals = []
    for sig in signals:
        if f"{sig}_anom" not in df_raw.columns:
            missing_signals.append(sig)
    
    if missing_signals:
        print(f"[WARN] Missing neural columns for: {missing_signals}")

    # 3. Check for Symbolic Vetoes
    veto_count = df_ref['is_vetoed'].sum()
    if veto_count == 0:
        print("[WARN] Zero symbolic vetoes found. Check if market_rules.lp and 07_apply_asp.py are case-sensitive.")
    else:
        print(f"[SUCCESS] Found {veto_count} symbolic vetoes for analysis.")

    return True

def plot_veto_analysis():
    """
    Analyzes why the ASP layer rejected neural anomalies.
    Provides Explainable AI (XAI) insights into the symbolic filtering.
    """
    # 1. Load Data
    if not os.path.exists(ANOM_RAW) or not os.path.exists(ANOM_REF):
        print(f"[ERROR] Data files not found. Ensure the pipeline has run up to Step 07.")
        return

    df_raw = pd.read_csv(ANOM_RAW, parse_dates=["Time (UTC)"])
    df_ref = pd.read_csv(ANOM_REF, parse_dates=["Time (UTC)"])
    
    signals = ["CF_Solar", "CF_Wind", "Load_MW", "Price"]
    
    # --- PRE-VIZ CHECK ---
    if not run_pre_viz_check(df_raw, df_ref, signals):
        print("[ERROR] Integrity check failed. Aborting visualization.")
        return

    rejection_data = []

    for signal in signals:
        col_name = f"{signal}_anom"
        if col_name not in df_raw.columns:
            continue

        # Get instances where Neural Layer flagged an anomaly
        total_neural = df_raw[df_raw[col_name] == 1].shape[0]
        
        # ASP target names are lowercase; check for confirmed vs vetoed
        sig_name_ref = signal.lower()
        confirmed = df_ref[(df_ref["target"] == sig_name_ref) & (df_ref["is_vetoed"] == 0)].shape[0]
        vetoed = df_ref[(df_ref["target"] == sig_name_ref) & (df_ref["is_vetoed"] == 1)].shape[0]
        
        # Cross-check consistency: if the sums don't match, warn the user
        if (confirmed + vetoed) != total_neural and total_neural > 0:
             print(f"[WARN] Mismatch for {signal}: Neural({total_neural}) != Confirmed({confirmed}) + Vetoed({vetoed})")
        
        rejection_data.append({
            "Signal": signal.replace("_", " "),
            "Confirmed (Symbolic)": confirmed,
            "Vetoed (Logic Violation)": vetoed
        })

    if not rejection_data:
        print("[ERROR] No matching signal data found for plotting.")
        return

    df_viz = pd.DataFrame(rejection_data).set_index("Signal")

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    df_viz.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], ax=ax, alpha=0.85)

    # 3. Styling
    ax.set_title("Explainable AI: Symbolic Veto Analysis\n(Neural Layer Suggestions vs. ASP Logical Validation)", fontsize=14)
    ax.set_ylabel("Number of Detected Points")
    ax.set_xlabel("Market & Grid Signals")
    ax.legend(title="Decision Source")
    plt.xticks(rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate with percentages
    for i, (idx, row) in enumerate(df_viz.iterrows()):
        total = row.sum()
        if total > 0:
            veto_pct = (row['Vetoed (Logic Violation)'] / total) * 100
            ax.text(i, total + 0.5, f"Veto Rate: {veto_pct:.1f}%", ha='center', weight='bold', color='#c0392b')

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG)
    print(f"[INFO] Veto analysis visualization saved to: {OUT_FIG}")

if __name__ == "__main__":
    plot_veto_analysis()