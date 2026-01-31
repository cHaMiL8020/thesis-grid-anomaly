import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Academic Plotting Configuration ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})

# Paths
REPORT_CSV = "reports/tables/global_performance.csv"
OUT_FIG = "reports/figures/model_radar_chart.png"

def generate_radar_chart():
    if not os.path.exists(REPORT_CSV):
        print(f"[ERROR] {REPORT_CSV} not found. Run src/19_global_performance_table.py first.")
        return

    # 1. Load the data generated in the previous step
    df = pd.read_csv(REPORT_CSV)
    
    # 2. Prepare Metrics (Higher is better for Radar Charts)
    # We invert Latency, Memory, and Params so that 'outer' points mean 'better/more efficient'
    
    # Speed = 1 / Latency
    df['Speed_Score'] = 1 / df['Lat(ms/sample)'].astype(float)
    
    # Memory Efficiency = 1 / Inc_RAM
    df['Mem_Score'] = 1 / df['Inc_RAM(MB)'].astype(float)
    
    # Compactness = 1 / Params
    # We treat Ridge (N/A) as 1 parameter (perfectly compact) for the scale
    df['Param_Score'] = df['Params'].replace('N/A', '1').str.replace(',', '').astype(float)
    df['Param_Score'] = 1 / df['Param_Score']

    # Accuracy Score: We normalize RMSE (Lower is better) to a 0-1 scale where 1 is best
    # Note: We use 1/RMSE to ensure 'outer' means 'more accurate'
    df['Acc_Score'] = 1 / df['RMSE']

    # 3. Normalize all scores to 0-100 range for the chart axes
    cols_to_norm = ['Acc_Score', 'Speed_Score', 'Mem_Score', 'Param_Score']
    for col in cols_to_norm:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * 80 + 20

    # 4. Radar Chart Setup
    categories = ['Accuracy (1/RMSE)', 'Inference Speed', 'Memory Efficiency', 'Compactness']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # Close the circle

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    
    # Model colors
    colors = {'Ridge (Baseline)': '#e74c3c', 
              'dCeNN-ELM (Proposed)': '#2ecc71', 
              'LSTM (Benchmark)': '#3498db'}

    for _, row in df.iterrows():
        values = [row['Acc_Score'], row['Speed_Score'], row['Mem_Score'], row['Param_Score']]
        values += values[:1] # Close the circle
        
        name = row['Model']
        color = colors.get(name, 'grey')
        
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.1)

    # 5. Styling
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color='black', size=11)
    
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)
    plt.ylim(0, 110)

    plt.title("System Efficiency vs. Performance Radar\n(Austrian Grid Anomaly Detection)", size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True)

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, bbox_inches='tight')
    print(f"[INFO] Radar chart saved to: {OUT_FIG}")

if __name__ == "__main__":
    generate_radar_chart()