import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import yaml
import importlib.util
from sklearn.linear_model import Ridge

# --- Academic Plotting Configuration ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'lines.linewidth': 1.2})

# Paths
DATA_NPZ = "artifacts/preprocessed_datasets_dcen_elm_h1.npz"
BASE_CFG = "configs/base.yaml"
DCENN_CFG = "configs/dcenn.yaml"
OUT_DIR = "reports/figures/validation_plots"
SCALER_Y_PATH = "artifacts/scaler_y.npz"

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_validation_predictions():
    """Generates predictions for all models with correct 72-hour lag alignment and Y-scaling."""
    # 1. Load Data
    data = np.load(DATA_NPZ, allow_pickle=True)
    X_tr, Y_tr = data["X_train"], data["Y_train"]
    X_va, Y_va = data["X_val"], data["Y_val"]
    target_names = data["target_names"]

    # 2. Ridge Baseline
    ridge = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    preds_ridge = ridge.predict(X_va)

    # 3. Proposed dCeNN-ELM
    train_mod = load_module("train_mod", "src/03_train_dcenn_elm.py")
    with open(DCENN_CFG, "r") as f:
        dcfg = yaml.safe_load(f)
    
    enc = train_mod.TinyDCeNN(X_va.shape[1], enc_dim=dcfg["enc_dim"], 
                              steps=dcfg["steps"], block=dcfg["block"])
    enc.load_state_dict(torch.load("artifacts/dcenn_encoder.pt"))
    enc.eval()
    
    with torch.no_grad():
        H_va = enc(torch.tensor(X_va, dtype=torch.float32)).numpy()
    
    elm_weights = np.load("artifacts/elm_heads.npz")["W"]
    preds_dcenn = H_va @ elm_weights

    # 4. LSTM (Benchmark) with Target Unscaling
    # Load scaling parameters saved during LSTM training
    if not os.path.exists(SCALER_Y_PATH):
        raise FileNotFoundError("LSTM Target Scaler missing. Retrain LSTM with scaling enabled.")
    
    sy = np.load(SCALER_Y_PATH)
    y_mean, y_scale = sy["mean"], sy["scale"]

    lstm_mod = load_module("lstm_mod", "src/13_benchmark_lstm.py")
    lookback = lstm_mod.LOOKBACK 
    
    lstm_model = lstm_mod.LSTMModel(in_dim=X_va.shape[1], hidden_dim=64, 
                                    out_dim=Y_va.shape[1], num_layers=2, dropout=0.2)
    lstm_model.load_state_dict(torch.load("artifacts/lstm_model.pt"))
    lstm_model.eval()
    
    # Context alignment for sliding window
    X_context = np.vstack([X_tr[-lookback:], X_va])
    X_va_seq = []
    for i in range(len(X_va)):
        seq = X_context[i : i + lookback]
        X_va_seq.append(seq)
    
    with torch.no_grad():
        # Step A: Get raw scaled predictions from LSTM
        preds_s = lstm_model(torch.tensor(np.array(X_va_seq), dtype=torch.float32)).numpy()
        # Step B: Invert scaling to match original MW/Price units
        preds_lstm = (preds_s * y_scale) + y_mean

    return Y_va, preds_dcenn, preds_lstm, preds_ridge, target_names

def plot_2_month_comparison(Y_true, Y_dcenn, Y_lstm, Y_ridge, target_names):
    """Plots a 4-panel comparison for a 2-month window."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(BASE_CFG, 'r') as f:
        cfg = yaml.safe_load(f)
    
    df_full = pd.read_csv(cfg["engineered_csv"], parse_dates=["Time (UTC)"], index_col="Time (UTC)")
    val_time_full = df_full.loc[cfg["split"]["val_start"]:cfg["split"]["val_end"]].index
    
    max_lag = 72
    val_time = val_time_full[max_lag:]

    window_hours = 24 * 60
    t_win = val_time[:window_hours]
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    colors = {'true': 'black', 'dcenn': '#2ecc71', 'lstm': '#3498db', 'ridge': '#e74c3c'}

    for i, (ax, target) in enumerate(zip(axes, target_names)):
        ax.plot(t_win, Y_true[:window_hours, i], label="Actual", color=colors['true'], alpha=0.3)
        ax.plot(t_win, Y_dcenn[:window_hours, i], label="dCeNN-ELM (Proposed)", color=colors['dcenn'])
        ax.plot(t_win, Y_lstm[:window_hours, i], label="LSTM (Benchmark)", color=colors['lstm'], linestyle='--')
        ax.plot(t_win, Y_ridge[:window_hours, i], label="Ridge (Linear)", color=colors['ridge'], linestyle=':')

        ax.set_ylabel(target.replace("_", " "))
        ax.grid(True, alpha=0.15)
        if i == 0:
            ax.legend(loc="upper right", ncol=4, frameon=True)

    plt.xlabel("Validation Timeline (Starting Jan 4, 2021)")
    plt.suptitle("Validation Set Performance Comparison (Target-Aligned)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = f"{OUT_DIR}/val_comparison_2months.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Comparison plot saved to: {save_path}")

if __name__ == "__main__":
    try:
        y_val, p_dcenn, p_lstm, p_ridge, names = load_validation_predictions()
        plot_2_month_comparison(y_val, p_dcenn, p_lstm, p_ridge, names)
    except Exception as e:
        print(f"[ERROR] {e}")