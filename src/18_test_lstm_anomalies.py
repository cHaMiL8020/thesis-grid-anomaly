import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import importlib.util

# --- Academic Plotting Configuration ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 200})

DATA_NPZ = "artifacts/preprocessed_datasets_dcen_elm_h1.npz"
LSTM_WEIGHTS = "artifacts/lstm_model.pt"
OUT_FIG = "reports/figures/lstm_anomaly_test.png"

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_diagnostic():
    # 1. Load Data
    data = np.load(DATA_NPZ, allow_pickle=True)
    X_tr, Y_tr = data["X_train"], data["Y_train"]
    X_te, Y_te = data["X_test"], data["Y_test"]
    target_names = list(data["target_names"])

    # 2. Load LSTM Model
    lstm_mod = load_module("lstm_mod", "src/13_benchmark_lstm.py")
    lookback = lstm_mod.LOOKBACK
    model = lstm_mod.LSTMModel(in_dim=X_te.shape[1], hidden_dim=64, 
                                out_dim=Y_te.shape[1], num_layers=2, dropout=0.2)
    
    if not os.path.exists(LSTM_WEIGHTS):
        print(f"[ERROR] {LSTM_WEIGHTS} not found. Run benchmark_lstm first.")
        return

    model.load_state_dict(torch.load(LSTM_WEIGHTS))
    model.eval()

    # 3. Warm Start Prediction (using end of training for Jan 2022 test set start)
    X_context = np.vstack([X_tr[-lookback:], X_te])
    X_seq = np.array([X_context[i : i + lookback] for i in range(len(X_te))])
    
    with torch.no_grad():
        preds = model(torch.tensor(X_seq, dtype=torch.float32)).numpy()

    # 4. Anomaly Check Logic
    # We define an anomaly where error > 3 * std_dev of the error
    residuals = np.abs(Y_te - preds)
    thresholds = np.mean(residuals, axis=0) + 3 * np.std(residuals, axis=0)

    # 5. Plotting (Solar, Wind, Price)
    target_indices = [0, 1, 3] # CF_Solar, CF_Wind, Price
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Zoom into a volatile 10-day window in early 2022
    window = slice(0, 240) 

    for ax, idx in zip(axes, target_indices):
        name = target_names[idx]
        y_true = Y_te[window, idx]
        y_pred = preds[window, idx]
        res = residuals[window, idx]
        thr = thresholds[idx]
        
        # Flags where the model is 'surprised'
        anom_mask = res > thr

        ax.plot(y_true, label="Ground Truth", color='black', alpha=0.6)
        ax.plot(y_pred, label="LSTM Prediction", color='blue', linestyle='--')
        
        # Highlight Anomalies
        ax.fill_between(range(len(y_true)), y_true, y_pred, where=anom_mask, 
                        color='red', alpha=0.3, label="LSTM Anomaly Detection")

        ax.set_title(f"LSTM Diagnostic: {name}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.xlabel("Hours (Start of 2022)")
    plt.tight_layout()
    plt.savefig(OUT_FIG)
    print(f"[INFO] Diagnostic plot saved to {OUT_FIG}")

if __name__ == "__main__":
    run_diagnostic()