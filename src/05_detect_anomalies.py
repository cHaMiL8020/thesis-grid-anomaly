# src/05_detect_anomalies.py

import os, json, yaml, numpy as np, pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================================
# 1. Load configs
# ==========================================================
with open("configs/base.yaml") as f:
    base = yaml.safe_load(f)

with open("configs/features.yaml") as f:
    f_cfg = yaml.safe_load(f)

with open("configs/thresholds.yaml") as f:
    t_cfg = yaml.safe_load(f)

ENGINEERED = base["engineered_csv"]
SPLIT = base["split"]

LAGS  = f_cfg["lags"]
ROLLS = f_cfg["rolls"]

BUCKET_BY = t_cfg.get("bucket_by", "none")

# ==========================================================
# 2. Tiny DCeNN (same as Step 03)
# ==========================================================
class TinyDCeNN(nn.Module):
    def __init__(self, in_dim, enc_dim=48, steps=2, block=8):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, enc_dim)
        self.cell    = nn.Linear(enc_dim, enc_dim)
        self.steps   = steps

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

# Encoder config
dcfg = yaml.safe_load(open("configs/dcenn.yaml"))
enc_dim = int(dcfg.get("enc_dim", 48))
steps   = int(dcfg.get("steps", 2))
block   = int(dcfg.get("block", 8))

# ==========================================================
# 3. Feature List (consistent with Step 02)
# ==========================================================
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

# ==========================================================
# 4. Supervised builder (same-hour + correct rolling)
# ==========================================================
def make_supervised(df, lags, rolls):
    X = df[FEATS_BASE].copy()
    core = [
        "Actual_Load_MW","Solar_MW","Wind_MW","Price_EUR_MWh",
        "temperature_2m (°C)","wind_speed_10m (m/s)",
        "shortwave_radiation (W/m²)"
    ]

    for c in core:
        for L in lags:
            X[f"{c}_lag{L}"] = df[c].shift(L)
        for R in rolls:
            X[f"{c}_rmean{R}"] = df[c].rolling(
                R, min_periods=int(0.7 * R)
            ).mean()

    Y = pd.DataFrame({
        "CF_Solar": df["CF_Solar"],
        "CF_Wind":  df["CF_Wind"],
        "Load_MW":  df["Actual_Load_MW"],
        "Price":    df["Price_EUR_MWh"],
    }, index=df.index)

    XY = X.join(Y).dropna()
    return XY.index, XY[X.columns], Y.loc[XY.index]

# ==========================================================
# 5. Load engineered
# ==========================================================
df = (pd.read_csv(ENGINEERED, parse_dates=["Time (UTC)"])
        .set_index("Time (UTC)")
        .sort_index())

# Train + test split
tr = df.loc[SPLIT["train_start"]:SPLIT["train_end"]]
te = df.loc[SPLIT["test_start"]: SPLIT["test_end"]]

t_tr, Xtr, Ytr = make_supervised(tr, LAGS, ROLLS)
t_te, Xte, Yte = make_supervised(te, LAGS, ROLLS)

# ==========================================================
# 6. Load scaler (DO NOT REFIT!)
# ==========================================================
scaler_obj = np.load("artifacts/scaler.npz", allow_pickle=True)
scaler = scaler_obj["scaler"][()]
Xtr_s = scaler.transform(Xtr)
Xte_s = scaler.transform(Xte)

# ==========================================================
# 7. Load encoder + ELM head
# ==========================================================
encoder = TinyDCeNN(Xtr_s.shape[1], enc_dim=enc_dim, steps=steps, block=block)
encoder.load_state_dict(torch.load("artifacts/dcenn_encoder.pt"))
encoder.eval()

with torch.no_grad():
    Hte = encoder(torch.tensor(Xte_s, dtype=torch.float32)).numpy()

Wobj = np.load("artifacts/elm_heads.npz", allow_pickle=True)
W = Wobj["W"]
target_names = list(Wobj["target_names"])

Yte_pred = Hte @ W

# ==========================================================
# 8. Residuals
# ==========================================================
resid = Yte.values - Yte_pred
abs_resid = np.abs(resid)

test_df = pd.DataFrame(index=t_te)
for i, name in enumerate(target_names):
    test_df[f"{name}_true"] = Yte.values[:, i]
    test_df[f"{name}_pred"] = Yte_pred[:, i]
    test_df[f"{name}_resid"] = resid[:, i]
    test_df[f"{name}_abs"]   = abs_resid[:, i]

# Bucket assignment
if BUCKET_BY == "hour":
    test_df["bucket"] = test_df.index.hour
elif BUCKET_BY == "holiday":
    test_df["bucket"] = df.loc[test_df.index, "is_public_holiday"]
else:
    test_df["bucket"] = 0

# ==========================================================
# 9. Threshold application
# ==========================================================
thr = json.load(open("artifacts/thresholds.json"))
thr_targets = thr["targets"]

def get_thr(tname, b):
    d = thr_targets[tname]
    if "default" in d:
        return float(d["default"])
    b = str(int(b))
    return float(d.get(b, list(d.values())[0]))

for name in target_names:
    th = np.array([get_thr(name, b) for b in test_df["bucket"]])
    test_df[f"{name}_thr"] = th
    test_df[f"{name}_anom"] = (test_df[f"{name}_abs"] > th).astype(int)

# Combined anomaly score
score = 0
for name in target_names:
    score += np.maximum(
        0.0,
        test_df[f"{name}_abs"] / (test_df[f"{name}_thr"] + 1e-9) - 1.0,
    )
test_df["combined_score"] = score

# ==========================================================
# 10. Save
# ==========================================================
os.makedirs("reports/tables", exist_ok=True)
test_df.to_csv("reports/tables/anomalies_2022.csv")

# Plot
os.makedirs("reports/figures", exist_ok=True)
plt.figure(figsize=(12,4))
plt.plot(test_df.index, test_df["combined_score"])
plt.ylabel("score")
plt.title("Combined Anomaly Score (2022)")
plt.tight_layout()
plt.savefig("reports/figures/anomaly_combined_2022.png", dpi=140)

print("Saved anomaly CSV + figure.")
