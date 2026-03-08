import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Academic Plotting Configuration ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 200})

# Paths
ANOM_RAW = "reports/tables/anomalies_2022.csv"
ANOM_REF = "reports/tables/anomalies_refined.csv"
OUT_FIG = "reports/figures/utility_waterfall_comparison.png"

# Backtest Configuration (Must match src/06_finance_mapping.py)
CFG = {
    "price_up_thresh": 0.0,
    "anomaly_bonus": 5.0,
    "imbalance_penalty": 2.0,
    "position_size": 1.0,
}

def calculate_utility(df, flag_col):
    """
    Calculates utility based on the formula in step 06.
    Utility = (Directional Gain) + (Anomaly Bonus) - (Imbalance Penalty)
    """
    # Logic: Predicted Price Delta
    d_pred = df["Price_pred"].diff().fillna(0.0)
    d_true = df["Price_true"].diff().fillna(0.0)
    
    # Policy: Trade if flag is active
    action = np.zeros(len(df))
    action[(df[flag_col] == 1) & (d_pred > CFG["price_up_thresh"])] = CFG["position_size"]
    action[(df[flag_col] == 1) & (d_pred < -CFG["price_up_thresh"])] = -CFG["position_size"]
    
    # Return Calculation (Price change * direction)
    returns = d_true * action
    
    # Imbalance Penalty (Mismatch between pred and true direction)
    wrong_dir = (np.sign(d_true) != np.sign(d_pred)) & (action != 0)
    penalties = wrong_dir.astype(float) * CFG["imbalance_penalty"]
    
    # Anomaly Bonus (Weighted by detection score)
    bonus = (df[flag_col] == 1).astype(float) * CFG["anomaly_bonus"] * df["combined_score"]
    
    return returns + bonus - penalties, penalties

def plot_utility_comparison():
    """Generates a comparison of Neural vs. Neuro-Symbolic utility and risk reduction."""
    # 1. Load and Align Data
    df_raw = pd.read_csv(ANOM_RAW, parse_dates=["Time (UTC)"]).set_index("Time (UTC)")
    df_ref_long = pd.read_csv(ANOM_REF, parse_dates=["Time (UTC)"])
    
    # Extract only confirmed Price flags from the Refined (Symbolic) set
    # target is lowercase 'price' based on Step 07 mapping
    df_price_ref = df_ref_long[(df_ref_long["target"] == "price") & (df_ref_long["is_vetoed"] == 0)]
    df_price_ref = df_price_ref.set_index("Time (UTC)")
    
    # Join Symbolic flags into the Raw dataframe
    df_raw["Symbolic_Flag"] = 0
    df_raw.loc[df_price_ref.index, "Symbolic_Flag"] = 1
    
    # 2. Calculate Utilities
    util_neural, pen_neural = calculate_utility(df_raw, "Price_anom")
    util_ns, pen_ns = calculate_utility(df_raw, "Symbolic_Flag")
    
    # 3. Create Comparison Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # --- Top Panel: Cumulative Timeline ---
    ax1.plot(df_raw.index, util_neural.cumsum(), label="Purely Neural Strategy", color="#e74c3c", alpha=0.8)
    ax1.plot(df_raw.index, util_ns.cumsum(), label="Neuro-Symbolic (Proposed)", color="#2ecc71", linewidth=2)
    ax1.set_title("Cumulative Financial Utility: Strategy Comparison", fontsize=14)
    ax1.set_ylabel("Cumulative Utility (€)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Bottom Panel: Risk Reduction (Waterfall/Bar) ---
    summary = pd.DataFrame({
        "Metric": ["Total Penalties", "Net Profit"],
        "Purely Neural": [pen_neural.sum(), util_neural.sum()],
        "Neuro-Symbolic": [pen_ns.sum(), util_ns.sum()]
    }).set_index("Metric")
    
    summary.T.plot(kind="bar", stacked=False, ax=ax2, color=["#f39c12", "#3498db"], rot=0)
    
    # Highlight the specific reduction in penalties
    reduction = pen_neural.sum() - pen_ns.sum()
    ax2.annotate(f"Risk Reduced by {reduction:,.0f}€", 
                 xy=(1, pen_ns.sum()), xytext=(0.5, pen_neural.sum() * 1.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    
    ax2.set_title("Strategy Performance Summary (Penalty Mitigation)", fontsize=12)
    ax2.set_ylabel("Euro (€)")
    ax2.legend(title="Financial Metric")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG)
    print(f"[INFO] Utility comparison saved to {OUT_FIG}")

if __name__ == "__main__":
    plot_utility_comparison()