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
    # Ensures a clean environment for each benchmark's timing logic
    result = subprocess.run(["python3", script_path], capture_output=False)
    return result.returncode == 0

def collect_results():
    """Executes all benchmarks and merges results into a single DataFrame."""
    # 1. Execute Benchmarks
    # We run them fresh to ensure consistent hardware state (CPU temperature/load)
    if not _run_script("src/12_benchmark_dcenn.py"): return None
    if not _run_script("src/13_benchmark_lstm.py"): return None
    if not _run_script("src/14_benchmark_ridge.py"): return None
    
    # 2. Load Results from generated CSVs
    try:
        dcenn_df = pd.read_csv("reports/tables/benchmark_dcenn.csv")
        lstm_df = pd.read_csv("reports/tables/benchmark_lstm.csv")
        ridge_df = pd.read_csv("reports/tables/benchmark_ridge.csv")
        
        # Merge all baselines with the proposed model
        df = pd.concat([dcenn_df, lstm_df, ridge_df], ignore_index=True)
        
        # Save Consolidated Table for Thesis Appendix
        os.makedirs("reports/tables", exist_ok=True)
        df.to_csv(BENCHMARK_CSV, index=False)
        return df
    except Exception as e:
        print(f"[ERROR] Result collection failed: {e}")
        return None

def plot_comparison(df):
    """Generates three-panel comparison plots for the Thesis Results section."""
    if df is None: return

    # Standard colors for your thesis defense
    # Green for proposed, Blue/Orange for baselines
    colors = ['#2ecc71', '#3498db', '#e67e22'] 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    models = df['model']
    
    # Panel 1: Accuracy (RMSE Price) - Lower is Better
    axes[0].bar(models, df['rmse_price'], color=colors)
    axes[0].set_title("Prediction Error (Price RMSE)")
    axes[0].set_ylabel("Euro/MWh")
    axes[0].tick_params(axis='x', rotation=15)

    # Panel 2: Training Efficiency (Log Scale)
    # Essential for showing dCeNN-ELM vs iterative LSTM
    axes[1].bar(models, df['train_time_sec'], color=colors)
    axes[1].set_title("Training Wall-Clock Time")
    axes[1].set_ylabel("Seconds (log)")
    axes[1].set_yscale('log')
    axes[1].tick_params(axis='x', rotation=15)

    # Panel 3: Edge Inference Latency (Log Scale)
    # Validates 'Edge Readiness' goal
    axes[2].bar(models, df['inf_latency_ms'], color=colors)
    axes[2].set_title("Per-Sample Inference Latency")
    axes[2].set_ylabel("ms (log)")
    axes[2].set_yscale('log')
    axes[2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig(BENCHMARK_PLOT, dpi=300)
    print(f"[INFO] Comparison visualization saved to {BENCHMARK_PLOT}")

if __name__ == "__main__":
    print("--- Initializing Master Benchmark Pipeline ---")
    df = collect_results()
    if df is not None:
        print("\n[SUMMARY TABLE]")
        print(df[["model", "rmse_price", "train_time_sec", "inf_latency_ms", "parameters"]])
        plot_comparison(df)
        print("\n--- Benchmarking Complete ---")