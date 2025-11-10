# src/05_detect_anomalies.py
import os, json, yaml, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
from torch import nn

# ----------------- Configs -----------------
base = yaml.safe_load(open("configs/base.yaml"))
f_cfg = yaml.safe_load(open("configs/features.yaml"))
t_cfg = yaml.safe_load(open("configs/thresholds.yaml"))

ENGINEERED = base["engineered_csv"]
SPLIT = base["split"]
HORIZON = int(base.get("horizon", 1))

LAGS  = f_cfg["lags"]
ROLLS = f_cfg["rolls"]

BUCKET_BY = t_cfg.get("bucket_by", "none")

# ----------------- Model (dCeNN) -----------------
class TinyDCeNN(nn.Module):
    def __init__(self, in_dim, enc_dim=48, steps=2, block=8):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, enc_dim, bias=True)
        self.cell = nn.Linear(enc_dim, enc_dim, bias=True)
        self.steps = steps
        mask = torch.zeros(enc_dim, enc_dim)
        for i in range(0, enc_dim, block):
            mask[i:i+block, i:i+block] = 1.0
        self.register_buffer("mask", mask)
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.8)
        nn.init.xavier_uniform_(self.cell.weight,    gain=0.2)
        nn.init.zeros_(self.in_proj.bias); nn.init.zeros_(self.cell.bias)
    def forward(self, x):
        h = torch.tanh(self.in_proj(x))
        for _ in range(self.steps):
            W = self.cell.weight * self.mask
            h = torch.tanh(torch.addmm(self.cell.bias, h, W.T) + 0.3*h)
        return h

def load_encoder():
    dcfg = yaml.safe_load(open("configs/dcenn.yaml"))
    return int(dcfg.get("enc_dim", 48)), int(dcfg.get("steps", 2)), int(dcfg.get("block", 8))

# ----------------- Feature building -----------------
FEATS_BASE = [
  "Actual_Load_MW","Solar_MW","Wind_MW","Price_EUR_MWh",
  "temperature_2m (°C)","relative_humidity_2m (%)","wind_speed_10m (m/s)",
  "surface_pressure (hPa)","precipitation (mm)","shortwave_radiation (W/m²)",
  "air_density_kgm3","wind_speed_100m (m/s)","wind_power_proxy","pv_proxy",
  "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
  "is_public_holiday","is_weekend","is_special_day"
]

def winsorize_inplace(df, cols, lo=0.01, hi=0.99):
    for c in cols:
        a,b = df[c].quantile([lo,hi]); df[c] = df[c].clip(a,b)

def make_supervised(df, horizon=1, lags=(1,2,3,6,12,24,48,72), rolls=(3,6,12,24,48)):
    X = df[FEATS_BASE].copy()
    core = ["Actual_Load_MW","Solar_MW","Wind_MW","Price_EUR_MWh",
            "temperature_2m (°C)","wind_speed_10m (m/s)","shortwave_radiation (W/m²)"]
    for c in core:
        for L in lags:  X[f"{c}_lag{L}"] = df[c].shift(L)
        for R in rolls: X[f"{c}_rmean{R}"] = df[c].rolling(R, min_periods=max(2,int(0.6*R))).mean()
    Y = pd.DataFrame({
        "CF_Solar": df["CF_Solar"].shift(-horizon),
        "CF_Wind":  df["CF_Wind"].shift(-horizon),
        "Load_MW":  df["Actual_Load_MW"].shift(-horizon),
        "Price":    df["Price_EUR_MWh"].shift(-horizon),
    }, index=df.index)
    XY = X.join(Y).dropna()
    return XY.index, XY.iloc[:, :X.shape[1]], XY.iloc[:, X.shape[1]:]

def sub(df, s, e): return df.loc[s:e]

# ----------------- Rebuild sets with timestamps -----------------
df = pd.read_csv(ENGINEERED, parse_dates=["Time (UTC)"]).set_index("Time (UTC)").sort_index()
winsor_cols = f_cfg["winsorize_cols"]; winsorize_inplace(df, winsor_cols)

tr = sub(df, SPLIT["train_start"], SPLIT["train_end"])
te = sub(df, SPLIT["test_start"],  SPLIT["test_end"])

t_tr, Xtr, Ytr = make_supervised(tr, HORIZON, tuple(LAGS), tuple(ROLLS))
t_te, Xte, Yte = make_supervised(te, HORIZON, tuple(LAGS), tuple(ROLLS))

# Standardize using Train only
scaler = StandardScaler().fit(Xtr.values)
Xtr_s, Xte_s = scaler.transform(Xtr.values), scaler.transform(Xte.values)

# ----------------- Load model + predict on Test -----------------
enc_dim, steps, block = load_encoder()
encoder = TinyDCeNN(Xtr_s.shape[1], enc_dim=enc_dim, steps=steps, block=block)
state = torch.load("artifacts/dcenn_encoder.pt", map_location="cpu")
encoder.load_state_dict(state["state_dict"]); encoder.eval()

with torch.no_grad():
    Hte = encoder(torch.tensor(Xte_s, dtype=torch.float32)).numpy()

Wobj = np.load("artifacts/elm_heads.npz", allow_pickle=True)
W = Wobj["W"]; target_names = [t for t in Wobj["target_names"]]
Yte_pred = Hte @ W

# Residuals
resid = Yte.values - Yte_pred
abs_resid = np.abs(resid)

test_df = pd.DataFrame(index=t_te)
for i, name in enumerate(target_names):
    test_df[f"{name}_true"] = Yte.values[:, i]
    test_df[f"{name}_pred"] = Yte_pred[:, i]
    test_df[f"{name}_resid"] = resid[:, i]
    test_df[f"{name}_abs"]   = abs_resid[:, i]

# Buckets for thresholds
if BUCKET_BY == "hour":
    test_df["bucket"] = test_df.index.hour.astype(int)
elif BUCKET_BY == "holiday":
    test_df["bucket"] = df.loc[test_df.index, "is_public_holiday"].astype(int)
else:
    test_df["bucket"] = 0

# ----------------- Load thresholds and flag anomalies -----------------
thr = json.load(open("artifacts/thresholds.json"))
thr_targets = thr["targets"]

def get_threshold(tname, bucket_value):
    d = thr_targets[tname]
    if "default" in d:
        return float(d["default"])
    key = str(int(bucket_value))
    return float(d.get(key, list(d.values())[0]))  # fallback to first bucket if missing

for name in target_names:
    th = np.array([get_threshold(name, b) for b in test_df["bucket"].values], dtype=float)
    test_df[f"{name}_thr"] = th
    test_df[f"{name}_anom"] = (test_df[f"{name}_abs"].values > th).astype(int)

# Combined anomaly score (simple sum of z-ish ranks normalized by threshold)
score = 0
for name in target_names:
    # 0 if below threshold, else (abs/threshold - 1)
    score += np.maximum(0.0, test_df[f"{name}_abs"].values / (test_df[f"{name}_thr"].values + 1e-9) - 1.0)
test_df["combined_score"] = score

# ----------------- Save outputs -----------------
os.makedirs("reports/tables", exist_ok=True)
out_csv = "reports/tables/anomalies_2022.csv"
test_df.to_csv(out_csv)
print(f"Saved {out_csv}")

# Quick plot of combined score
os.makedirs("reports/figures", exist_ok=True)
plt.figure(figsize=(12,4))
plt.plot(test_df.index, test_df["combined_score"].values)
plt.title("Combined anomaly score — Test 2022")
plt.ylabel("score"); plt.xlabel("time")
plt.tight_layout()
plt.savefig("reports/figures/anomaly_combined_2022.png", dpi=140)
print("Saved reports/figures/anomaly_combined_2022.png")


