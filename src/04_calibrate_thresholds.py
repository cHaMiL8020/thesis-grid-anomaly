# src/04_calibrate_thresholds.py

import os, json, yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
from torch import nn

# ================================================================
# 1. LOAD CONFIGS
# ================================================================
with open("configs/base.yaml") as f:
    base = yaml.safe_load(f)

with open("configs/features.yaml") as f:
    f_cfg = yaml.safe_load(f)

with open("configs/thresholds.yaml") as f:
    t_cfg = yaml.safe_load(f)

ENGINEERED   = base["engineered_csv"]
HOLIDAYS_CSV = base["holidays_csv"]
NPZ_PATH     = base["npz_path"]

SPLIT   = base["split"]
HORIZON = int(base.get("horizon", 1))

LAGS  = f_cfg["lags"]
ROLLS = f_cfg["rolls"]
WINS  = f_cfg["winsorize_cols"]

METHOD    = t_cfg.get("method", "conformal")     # conformal | rolling_mad
ALPHA     = float(t_cfg.get("alpha", 0.90))
BUCKET_BY = t_cfg.get("bucket_by", "none")       # none | hour | holiday

# ================================================================
# 2. Tiny DCeNN (same as Step 03)
# ================================================================
class TinyDCeNN(nn.Module):
    def __init__(self, in_dim, enc_dim=48, steps=2, block=8):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, enc_dim)
        self.cell    = nn.Linear(enc_dim, enc_dim)
        self.steps   = steps

        # Block diagonal connectivity mask
        mask = torch.zeros(enc_dim, enc_dim)
        for i in range(0, enc_dim, block):
            mask[i:i+block, i:i+block] = 1.0
        self.register_buffer("mask", mask)

        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.8)
        nn.init.xavier_uniform_(self.cell.weight,    gain=0.2)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.cell.bias)

    def forward(self, x):
        h = torch.tanh(self.in_proj(x))
        for _ in range(self.steps):
            W = self.cell.weight * self.mask
            h = torch.tanh(torch.addmm(self.cell.bias, h, W.T) + 0.3*h)
        return h


# Load encoder parameters
dcfg = yaml.safe_load(open("configs/dcenn.yaml"))
enc_dim = int(dcfg.get("enc_dim", 48))
steps   = int(dcfg.get("steps", 2))
block   = int(dcfg.get("block", 8))


# ================================================================
# 3. Feature List (must match Step 02 EXACTLY)
# ================================================================
FEATS_BASE = [
    "Actual_Load_MW","Solar_MW","Wind_MW","Price_EUR_MWh",
    "temperature_2m (°C)","relative_humidity_2m (%)",
    "wind_speed_10m (m/s)","surface_pressure (hPa)",
    "shortwave_radiation (W/m²)",
    "air_density_kgm3","wind_speed_100m (m/s)",
    "wind_power_proxy","pv_proxy",
    "hour_sin","hour_cos","dow_sin","dow_cos",
    "month_sin","month_cos",
    "is_public_holiday","is_weekend","is_special_day"
]

# ================================================================
# 4. Utility Functions
# ================================================================
def winsorize_inplace(df, cols, lo=0.01, hi=0.99):
    for c in cols:
        lo_q, hi_q = df[c].quantile([lo, hi])
        df[c] = df[c].clip(lo_q, hi_q)

def make_supervised(df, lags, rolls):
    """
    Same-hour targets (correct for anomaly detection)
    EXACT logics as Step 02.
    """
    X = df[FEATS_BASE].copy()
    core = [
        "Actual_Load_MW","Solar_MW","Wind_MW","Price_EUR_MWh",
        "temperature_2m (°C)","wind_speed_10m (m/s)",
        "shortwave_radiation (W/m²)"
    ]

    # Lags + rolling windows
    for c in core:
        for L in lags:
            X[f"{c}_lag{L}"] = df[c].shift(L)
        for R in rolls:
            X[f"{c}_rmean{R}"] = df[c].rolling(
                R, min_periods=int(0.7*R)
            ).mean()

    # Same-hour (NOT horizon-shifted) targets
    Y = pd.DataFrame({
        "CF_Solar": df["CF_Solar"],
        "CF_Wind":  df["CF_Wind"],
        "Load_MW":  df["Actual_Load_MW"],
        "Price":    df["Price_EUR_MWh"],
    }, index=df.index)

    XY = X.join(Y).dropna()
    return XY.index, XY[X.columns], Y.loc[XY.index]


# ================================================================
# 5. LOAD ENGINEERED DATA + REBUILD SUPERVISED SETS
# ================================================================
df = (pd.read_csv(ENGINEERED, parse_dates=["Time (UTC)"])
        .set_index("Time (UTC)")
        .sort_index())

winsorize_inplace(df, WINS)

# Split
tr = df.loc[SPLIT["train_start"]:SPLIT["train_end"]]
va = df.loc[SPLIT["val_start"]:SPLIT["val_end"]]

t_tr, Xtr, Ytr = make_supervised(tr, LAGS, ROLLS)
t_va, Xva, Yva = make_supervised(va, LAGS, ROLLS)

# ================================================================
# 6. LOAD SCALER from Step 02 (not refit!)
# ================================================================
scaler_obj = np.load("artifacts/scaler.npz", allow_pickle=True)
scaler = scaler_obj["scaler"][()]

Xtr_s = scaler.transform(Xtr)
Xva_s = scaler.transform(Xva)


# ================================================================
# 7. LOAD ENCODER + ELM HEAD
# ================================================================
encoder = TinyDCeNN(Xtr_s.shape[1], enc_dim=enc_dim, steps=steps, block=block)
state = torch.load("artifacts/dcenn_encoder.pt", map_location="cpu")
encoder.load_state_dict(state)
encoder.eval()

with torch.no_grad():
    Hva = encoder(torch.tensor(Xva_s, dtype=torch.float32)).numpy()

Wobj = np.load("artifacts/elm_heads.npz", allow_pickle=True)
W = Wobj["W"]
target_names = [t for t in Wobj["target_names"]]

# Predictions + residuals
Yva_pred = Hva @ W
resid     = Yva.values - Yva_pred
abs_resid = np.abs(resid)


# ================================================================
# 8. BUILD RESIDUAL DF w/ BUCKETING
# ================================================================
val_df = pd.DataFrame(index=t_va)
for i, name in enumerate(target_names):
    val_df[f"{name}_resid"] = resid[:, i]
    val_df[f"{name}_abs"]   = abs_resid[:, i]

# Bucketing
if BUCKET_BY == "hour":
    val_df["bucket"] = val_df.index.hour
elif BUCKET_BY == "holiday":
    val_df["bucket"] = df.loc[val_df.index, "is_public_holiday"].astype(int)
else:
    val_df["bucket"] = 0


# ================================================================
# 9. THRESHOLD CALIBRATION
# ================================================================
def conformal(series, alpha):
    return float(np.nanquantile(series, alpha))

def mad_threshold(series, alpha):
    med = np.nanmedian(series)
    mad = np.nanmedian(np.abs(series - med)) + 1e-9
    # Simple scaling: t = k * MAD where k approximates quantile
    k = np.sqrt(2) * (alpha / (1 - alpha))
    return float(k * mad)

thresholds = {
    "meta": {
        "method": METHOD,
        "alpha": ALPHA,
        "bucket_by": BUCKET_BY,
    },
    "targets": {}
}

for name in target_names:
    thr = {}
    if BUCKET_BY in ("hour", "holiday"):
        for b, g in val_df.groupby("bucket"):
            if METHOD == "conformal":
                thr[str(b)] = conformal(g[f"{name}_abs"], ALPHA)
            else:
                thr[str(b)] = mad_threshold(g[f"{name}_resid"], ALPHA)
    else:
        if METHOD == "conformal":
            thr["default"] = conformal(val_df[f"{name}_abs"], ALPHA)
        else:
            thr["default"] = mad_threshold(val_df[f"{name}_resid"], ALPHA)
    thresholds["targets"][name] = thr


# ================================================================
# 10. SAVE ARTIFACTS
# ================================================================
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)


# ================================================================
# 11. RESIDUAL DISTRIBUTION PLOT
# ================================================================
os.makedirs("reports/figures", exist_ok=True)

plt.figure(figsize=(10,6))
for name in target_names:
    arr = val_df[f"{name}_resid"].dropna().values
    if len(arr) == 0: 
        continue
    z = (arr - np.mean(arr)) / (np.std(arr) + 1e-9)
    hist, bins = np.histogram(z, bins=200, density=True)
    centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(centers, hist, label=name)

plt.title("Validation Residual Distribution (z-scored)")
plt.xlabel("z-score residual")
plt.ylabel("density")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/residual_hist_val.png", dpi=140)

print("Saved thresholds + residual histogram.")
