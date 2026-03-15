# src/benchmark_ridge.py
#!/usr/bin/env python3
"""
Module 3: Benchmark Model (Ridge Regression)
Purpose: A simple linear baseline to prove the value of the dCeNN encoder.
"""

import numpy as np
import time
import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def train_benchmark():
    # 1. Load Data
    data_path = "artifacts/preprocessed_datasets_dcen_elm_h1.npz"
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at {data_path}. Run 'make split' first.")
        return

    data = np.load(data_path, allow_pickle=True)
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_test, Y_test = data["X_test"], data["Y_test"]
    target_names = [str(t) for t in data["target_names"]]

    # Sanity-check expected target ordering/labels
    expected_targets = ["CF_Solar", "CF_Wind", "Load_MW", "Price"]
    target_lower = {t.lower(): i for i, t in enumerate(target_names)}
    missing = [t for t in expected_targets if t.lower() not in target_lower]
    if missing:
        raise RuntimeError(
            "Unexpected target_names in NPZ: missing "
            f"{missing}. Got: {target_names}"
        )

    # 2. Setup Model (Standard Ridge Regression)
    model = Ridge(alpha=1.0)

    # 3. Benchmark Training Time
    print("[INFO] Starting Ridge Regression Training...")
    start_time = time.time()
    model.fit(X_train, Y_train)
    training_duration = time.time() - start_time
    print(f"[RESULT] Ridge Training Time: {training_duration:.4f} seconds")

    # 4. Benchmark Inference Latency
    latencies = []
    for i in range(min(1000, len(X_test))):
        sample = X_test[i:i+1]
        start_inf = time.perf_counter()
        _ = model.predict(sample)
        latencies.append(time.perf_counter() - start_inf)
    
    avg_inf_latency_ms = np.mean(latencies) * 1000
    print(f"[RESULT] Ridge Avg Inference Latency: {avg_inf_latency_ms:.4f} ms")

    # 5. Calculate Accuracy (RMSE)
    # Note: Y_test is already in real units (Euro/MWh, MW, etc.), so no re-scaling is required.
    preds = model.predict(X_test)

    # Calculate RMSE for each target (same order as target_names)
    raw_rmse = np.sqrt(mean_squared_error(Y_test, preds, multioutput='raw_values'))

    errors = {name.lower(): raw_rmse[i] for i, name in enumerate(target_names)}

    # Final print to verify
    print("\n[RESULT] Ridge RMSE (Real Units):")
    for n, err in errors.items():
        print(f"  {n}: {err:.4f}")

    # 6. Save results - Robust Mapping
    # We look for the keys regardless of where they are in the list
    results = {
        "model": "Ridge (Linear Baseline)",
        "train_time_sec": training_duration,
        "inf_latency_ms": avg_inf_latency_ms,
        "rmse_solar": errors.get("cf_solar", 0),
        "rmse_wind":  errors.get("cf_wind", 0),
        "rmse_load":  errors.get("actual_load_mw", 0) or errors.get("load_mw", 0),
        "rmse_price": errors.get("price", 0) or errors.get("price_eur_mwh", 0),
        "parameters": X_train.shape[1] * Y_train.shape[1] 
    }
    
    os.makedirs("reports/tables", exist_ok=True)
    pd.DataFrame([results]).to_csv("reports/tables/benchmark_ridge.csv", index=False)

if __name__ == "__main__":
    train_benchmark()