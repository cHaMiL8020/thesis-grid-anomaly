# src/run_benchmarks.py
#!/usr/bin/env python3
"""
Master Benchmark Runner: dCeNN-ELM vs. LSTM vs. Ridge.
Consolidates Accuracy, Speed, and Size metrics into professional reports.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

# --- Configuration ---
BENCHMARK_CSV = "reports/tables/benchmark_comparison.csv"
BENCHMARK_PLOT = "reports/figures/benchmark_comparison.png"

def _run_script(script_path):
    """Executes a benchmark script and returns True if successful."""
    print(f"[RUNNING] {script_path}...")
    result = subprocess.run(["python3", script_path], capture_output=False)
    return result.returncode == 0

def collect_results():
    """Executes all benchmarks and merges results into a single DataFrame."""
    # 1. Execute Benchmarks
    if not _run_script("src/12_benchmark_dcenn.py"): return None
    if not _run_script("src/13_benchmark_lstm.py"): return None
    if not _run_script("src/14_benchmark_ridge.py"): return None
    
    # 2. Load Results from generated CSVs
    try:
        dcenn_df = pd.read_csv("reports/tables/benchmark_dcenn.csv")
        lstm_df = pd.read_csv("reports/tables/benchmark_lstm.csv")
        ridge_df = pd.read_csv("reports/tables/benchmark_ridge.csv")
        
        df = pd.concat([dcenn_df, lstm_df, ridge_df], ignore_index=True)
        
        os.makedirs("reports/tables", exist_ok=True)
        df.to_csv(BENCHMARK_CSV, index=False)
        return df
    except Exception as e:
        print(f"[ERROR] Result collection failed: {e}")
        return None

def plot_comparison(df):
    """Generates four-panel comparison plots to handle different unit scales."""
    if df is None: return

    # Academic Color Palette
    colors = ['#2ecc71', '#3498db', '#e67e22'] 
    
    # Updated to a 2x2 grid to better organize the different units (MW and €)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    models = df['model']
    
    # --- Panel 1: Load Accuracy (MW) ---
    axes[0, 0].bar(models, df['rmse_load'], color=colors)
    axes[0, 0].set_title("Load Prediction Error (RMSE MW)", fontweight='bold')
    axes[0, 0].set_ylabel("MW")
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Panel 2: Price Accuracy (€/MWh) ---
    axes[0, 1].bar(models, df['rmse_price'], color=colors)
    axes[0, 1].set_title("Price Prediction Error (RMSE €/MWh)", fontweight='bold')
    axes[0, 1].set_ylabel("Euro/MWh")
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Panel 3: Training Efficiency (Wall-Clock) ---
    axes[1, 0].bar(models, df['train_time_sec'], color=colors)
    axes[1, 0].set_title("Training Time (Efficiency)", fontweight='bold')
    axes[1, 0].set_ylabel("Seconds (log)")
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Panel 4: Edge Latency (Inference) ---
    axes[1, 1].bar(models, df['inf_latency_ms'], color=colors)
    axes[1, 1].set_title("Edge Inference Latency", fontweight='bold')
    axes[1, 1].set_ylabel("ms (log)")
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(pad=4.0)
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig(BENCHMARK_PLOT, dpi=300)
    print(f"[INFO] Comparison visualization saved to {BENCHMARK_PLOT}")

if __name__ == "__main__":
    print("--- Initializing Master Benchmark Pipeline ---")
    df = collect_results()
    if df is not None:
        print("\n[SUMMARY TABLE]")
        print(df[["model", "rmse_load", "rmse_price", "train_time_sec", "inf_latency_ms"]])
        plot_comparison(df)
        print("\n--- Benchmarking Complete ---")