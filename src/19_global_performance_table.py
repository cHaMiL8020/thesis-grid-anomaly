import pandas as pd
import numpy as np
import time
import torch
import os
import psutil
import platform
import yaml
import importlib.util
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

# Paths
DATA_NPZ = "artifacts/preprocessed_datasets_dcen_elm_h1.npz"
DCENN_CFG = "configs/dcenn.yaml"
REPORT_DIR = "reports/tables"
REPORT_CSV = f"{REPORT_DIR}/global_performance.csv"
REPORT_TXT = f"{REPORT_DIR}/system_hardware_report.txt"

def get_full_sys_info():
    """Captures comprehensive hardware and environment specifications."""
    return {
        "Architecture": platform.machine(),
        "OS": f"{platform.system()} {platform.release()}",
        "Processor": platform.processor() or "x86_64",
        "Physical Cores": psutil.cpu_count(logical=False),
        "Total Logical Cores": psutil.cpu_count(logical=True),
        "Total RAM": f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB",
        "Python Version": platform.python_version(),
        "PyTorch Version": torch.__version__
    }

def measure_usage(func, *args):
    """Measures incremental RAM and Average CPU during function execution."""
    process = psutil.Process(os.getpid())
    base_mem = process.memory_info().rss / 1024**2
    
    # Initialize CPU measurement (requires an interval or two calls)
    psutil.cpu_percent(interval=None) 
    start_time = time.perf_counter()
    
    result = func(*args)
    
    end_time = time.perf_counter()
    # Calculate average CPU usage over the duration
    avg_cpu = psutil.cpu_percent(interval=None)
    incremental_mem = (process.memory_info().rss / 1024**2) - base_mem
    duration_ms = (end_time - start_time) * 1000
    
    return result, incremental_mem, avg_cpu, duration_ms

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def evaluate_all_models():
    os.makedirs(REPORT_DIR, exist_ok=True)
    sys_info = get_full_sys_info()
    
    data = np.load(DATA_NPZ, allow_pickle=True)
    X_tr, Y_tr = data["X_train"], data["Y_train"]
    X_te, Y_te = data["X_test"], data["Y_test"]
    sy = np.load("artifacts/scaler_y.npz")
    results = []

    # --- MODEL A: Ridge (Linear Baseline) ---
    ridge = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    preds_ridge, mem_r, cpu_r, dur_r = measure_usage(ridge.predict, X_te)
    results.append({
        "Model": "Ridge (Baseline)", 
        "RMSE": np.sqrt(mean_squared_error(Y_te, preds_ridge)),
        "Lat(ms/sample)": f"{dur_r/len(X_te):.5f}", 
        "Avg_CPU(%)": f"{cpu_r:.1f}",
        "Inc_RAM(MB)": f"{max(0.01, mem_r):.2f}",
        "Params": "N/A"
    })

    # --- MODEL B: dCeNN-ELM (Proposed) ---
    train_mod = load_module("train_mod", "src/03_train_dcenn_elm.py")
    with open(DCENN_CFG, "r") as f: dcfg = yaml.safe_load(f)
    enc = train_mod.TinyDCeNN(X_te.shape[1], dcfg["enc_dim"], dcfg["steps"], dcfg["block"])
    enc.load_state_dict(torch.load("artifacts/dcenn_encoder.pt"))
    enc.eval()
    elm_weights = np.load("artifacts/elm_heads.npz")["W"]

    def dcenn_inf(x):
        with torch.no_grad():
            return (enc(torch.tensor(x, dtype=torch.float32)).numpy()) @ elm_weights

    preds_dcenn, mem_d, cpu_d, dur_d = measure_usage(dcenn_inf, X_te)
    p_count = sum(p.numel() for p in enc.parameters()) + elm_weights.size
    results.append({
        "Model": "dCeNN-ELM (Proposed)", 
        "RMSE": np.sqrt(mean_squared_error(Y_te, preds_dcenn)),
        "Lat(ms/sample)": f"{dur_d/len(X_te):.5f}", 
        "Avg_CPU(%)": f"{cpu_d:.1f}",
        "Inc_RAM(MB)": f"{max(0.01, mem_d):.2f}",
        "Params": f"{p_count:,}"
    })

    # --- MODEL C: LSTM (Benchmark) ---
    lstm_mod = load_module("lstm_mod", "src/13_benchmark_lstm.py")
    lstm_model = lstm_mod.LSTMModel(X_te.shape[1], 64, Y_te.shape[1])
    lstm_model.load_state_dict(torch.load("artifacts/lstm_model.pt"))
    lstm_model.eval()

    def lstm_inf_batched(x):
        batch_size = 500
        preds = []
        X_p = np.vstack([X_tr[-24:], x])
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                b_seq = np.array([X_p[j:j+24] for j in range(i, min(i+batch_size, len(x)))])
                p_s = lstm_model(torch.tensor(b_seq, dtype=torch.float32)).numpy()
                preds.append((p_s * sy["scale"]) + sy["mean"])
        return np.vstack(preds)

    preds_lstm, mem_l, cpu_l, dur_l = measure_usage(lstm_inf_batched, X_te)
    results.append({
        "Model": "LSTM (Benchmark)", 
        "RMSE": np.sqrt(mean_squared_error(Y_te, preds_lstm)),
        "Lat(ms/sample)": f"{dur_l/len(X_te):.5f}", 
        "Avg_CPU(%)": f"{cpu_l:.1f}",
        "Inc_RAM(MB)": f"{mem_l:.2f}",
        "Params": f"{sum(p.numel() for p in lstm_model.parameters()):,}"
    })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(REPORT_CSV, index=False)

    # Save Full Hardware Report
    with open(REPORT_TXT, "w") as f:
        f.write("=== THESIS HARDWARE & SYSTEM REPORT ===\n")
        for k, v in sys_info.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== PERFORMANCE SUMMARY ===\n")
        f.write(df.to_string(index=False))

    return df, sys_info

if __name__ == "__main__":
    df, sys = evaluate_all_models()
    print("\n" + "="*80)
    print("GLOBAL PERFORMANCE & HARDWARE REPORT")
    print("="*80)
    print(f"Hardware: {sys['Processor']} | {sys['Total RAM']} | {sys['Physical Cores']} Cores")
    print("-"*80)
    print(df.to_string(index=False))
    print("="*80)
    print(f"Full report saved to: {REPORT_TXT}")