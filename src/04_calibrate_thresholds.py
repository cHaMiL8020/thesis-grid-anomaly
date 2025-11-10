# src/04_calibrate_thresholds.py
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
HOLIDAYS_CSV = base["holidays_csv"]
NPZ_PATH = base["npz_path"]  # not required, but kept for reference

SPLIT = base["split"]
HORIZON = int(base.get("horizon", 1))

LAGS = f_cfg["lags"]
ROLLS = f_cfg["rolls"]

METHOD = t_cfg.get("method", "conformal")           # conformal | rolling_mad
ALPHA = float(t_cfg.get("alpha", 0.90))             # e.g., 0.90
BUCKET_BY = t_cfg.get("bucket_by", "none")          # none | hour | holiday

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

def load_encoder_and_heads():
    # Load training configs for encoder dims
    dcfg = yaml.safe_load(open("configs/dcenn.yaml"))
    enc_dim = int(dcfg.get("enc_dim", 48))
    steps   = int(dcfg.get("steps", 2))
    block   = int(dcfg.get("block", 8))

    # We need input dimension -> get from rebuilt features later
    return enc_dim, steps, block

# ----------------- Feature building (same as split) -----------------
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
        a,b = df[c].quantile([lo,hi])
        df[c] = df[c].clip(a,b)

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
    # return timestamps to map back
    return XY.index, XY.iloc[:, :X.shape[1]], XY.iloc[:, X.shape[1]:]

def sub(df, s, e): return df.loc[s:e]

# ----------------- Load engineered hourly + rebuild sets -----------------
df = pd.read_csv(ENGINEERED, parse_dates=["Time (UTC)"]).set_index("Time (UTC)").sort_index()

# Winsorize like in split script
winsor_cols = f_cfg["winsorize_cols"]
winsorize_inplace(df, winsor_cols)

# Train/Val/Test windows
tr = sub(df, SPLIT["train_start"], SPLIT["train_end"])
va = sub(df, SPLIT["val_start"],   SPLIT["val_end"])
te = sub(df, SPLIT["test_start"],  SPLIT["test_end"])

# Build supervised + capture timestamps
t_tr, Xtr, Ytr = make_supervised(tr, HORIZON, tuple(LAGS), tuple(ROLLS))
t_va, Xva, Yva = make_supervised(va, HORIZON, tuple(LAGS), tuple(ROLLS))
t_te, Xte, Yte = make_supervised(te, HORIZON, tuple(LAGS), tuple(ROLLS))

# Standardize using Train only
scaler = StandardScaler().fit(Xtr.values)
Xtr_s, Xva_s = scaler.transform(Xtr.values), scaler.transform(Xva.values)

# ----------------- Load encoder + heads; predict on Val -----------------
enc_dim, steps, block = load_encoder_and_heads()
encoder = TinyDCeNN(Xtr_s.shape[1], enc_dim=enc_dim, steps=steps, block=block)
state = torch.load("artifacts/dcenn_encoder.pt", map_location="cpu")
encoder.load_state_dict(state["state_dict"])
encoder.eval()

with torch.no_grad():
    Hva = encoder(torch.tensor(Xva_s, dtype=torch.float32)).numpy()

Wobj = np.load("artifacts/elm_heads.npz", allow_pickle=True)
W = Wobj["W"]  # enc_dim x n_targets
target_names = [t for t in Wobj["target_names"]]

Yva_pred = Hva @ W
resid = Yva.values - Yva_pred
abs_resid = np.abs(resid)

# ----------------- Calibration -----------------
val_df = pd.DataFrame(index=t_va)
for i, name in enumerate(target_names):
    val_df[f"{name}_resid"] = resid[:, i]
    val_df[f"{name}_abs"]   = abs_resid[:, i]

# Add buckets
if BUCKET_BY == "hour":
    val_df["bucket"] = val_df.index.hour.astype(int)
elif BUCKET_BY == "holiday":
    # need holiday info from engineered df
    val_df["bucket"] = df.loc[val_df.index, "is_public_holiday"].astype(int)
else:
    val_df["bucket"] = 0

thresholds = {"meta": {
    "method": METHOD,
    "alpha": ALPHA,
    "bucket_by": BUCKET_BY
}, "targets": {}}

def conformal_threshold(series, alpha):
    # quantile of |residual| at alpha (e.g., 0.90)
    return float(np.nanquantile(series.dropna().values, alpha))

def rolling_mad_threshold(series, alpha):
    # median +/- k * MAD; choose k s.t. coverage ≈ alpha (rough)
    # For Laplace-ish residuals, P(|X-median| <= k*MAD) ~ tanh(k/2)
    # Solve alpha ≈ tanh(k/2)  => k ≈ 2 * arctanh(alpha)
    import math
    k = 2.0 * np.arctanh(alpha - 1e-6)  # clamp
    med = float(np.nanmedian(series))
    mad = float(np.nanmedian(np.abs(series - med)) + 1e-9)
    return float(k * mad)

for name in target_names:
    thr = {}
    if BUCKET_BY in ("hour", "holiday"):
        for b, g in val_df.groupby("bucket"):
            if METHOD == "conformal":
                tval = conformal_threshold(g[f"{name}_abs"], ALPHA)
            else:
                tval = rolling_mad_threshold(g[f"{name}_resid"], ALPHA)
            thr[str(b)] = tval
    else:
        if METHOD == "conformal":
            thr["default"] = conformal_threshold(val_df[f"{name}_abs"], ALPHA)
        else:
            thr["default"] = rolling_mad_threshold(val_df[f"{name}_resid"], ALPHA)
    thresholds["targets"][name] = thr

# ----------------- Save artifacts -----------------
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)

# Quick residual plot
os.makedirs("reports/figures", exist_ok=True)
plt.figure(figsize=(8,5))
for name in target_names:
    plt.hist(val_df[f"{name}_resid"].dropna().values, bins=80, alpha=0.4, label=name, density=True)
plt.title("Validation residual distribution")
plt.xlabel("residual"); plt.ylabel("density"); plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/residual_hist_val.png", dpi=140)

print("Saved artifacts/thresholds.json and reports/figures/residual_hist_val.png")


