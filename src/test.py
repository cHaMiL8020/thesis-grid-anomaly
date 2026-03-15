# src/99_verify_indices.py
import numpy as np
import os

data_path = "artifacts/preprocessed_datasets_dcen_elm_h1.npz"
scaler_path = "artifacts/scaler_y.npz"

print("--- THESIS DATA INTEGRITY CHECK ---")
if os.path.exists(data_path):
    data = np.load(data_path, allow_pickle=True)
    names = data["target_names"]
    print(f"\n[1] Target Column Order (Index 0-3):")
    for i, name in enumerate(names):
        print(f"    Index {i}: {name}")
else:
    print("[ERROR] Dataset .npz not found.")

if os.path.exists(scaler_path):
    sy = np.load(scaler_path)
    print(f"\n[2] Scaling Values (Index 0-3):")
    for i, s in enumerate(sy["scale"]):
        print(f"    Index {i} Scale: {s:.4f}")