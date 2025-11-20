# src/02_split_and_scale.py

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------
# Load configs
# --------------------------------------------------------
with open("configs/base.yaml") as f:
    base = yaml.safe_load(f)

with open("configs/features.yaml") as f:
    f_cfg = yaml.safe_load(f)

ENGINEERED = base["engineered_csv"]
NPZ_PATH   = base["npz_path"]
SPLIT      = base["split"]
HORIZON    = int(base.get("horizon", 1))  # currently unused (same-hour targets)

LAGS  = f_cfg["lags"]
ROLLS = f_cfg["rolls"]
WINS  = f_cfg["winsorize_cols"]

# --------------------------------------------------------
# Feature list (authoritative for pipeline)
# --------------------------------------------------------
FEATS_BASE = [
    "Actual_Load_MW", "Solar_MW", "Wind_MW", "Price_EUR_MWh",
    "temperature_2m (°C)", "relative_humidity_2m (%)",
    "wind_speed_10m (m/s)", "surface_pressure (hPa)",
    "shortwave_radiation (W/m²)",
    "air_density_kgm3", "wind_speed_100m (m/s)",
    "wind_power_proxy", "pv_proxy",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_public_holiday", "is_weekend", "is_special_day",
]

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------
def winsorize_inplace(df, cols, lo=0.01, hi=0.99):
    for c in cols:
        lo_q, hi_q = df[c].quantile([lo, hi])
        df[c] = df[c].clip(lo_q, hi_q)

def make_supervised(df, lags, rolls):
    """
    Build supervised dataset:
      - X: engineered + lags + rolling means
      - Y: same-hour targets (NO future shift)
    """
    X = df[FEATS_BASE].copy()

    core = [
        "Actual_Load_MW", "Solar_MW", "Wind_MW", "Price_EUR_MWh",
        "temperature_2m (°C)", "wind_speed_10m (m/s)",
        "shortwave_radiation (W/m²)",
    ]

    for c in core:
        for L in lags:
            X[f"{c}_lag{L}"] = df[c].shift(L)
        for R in rolls:
            X[f"{c}_rmean{R}"] = df[c].rolling(
                R, min_periods=int(0.7 * R)
            ).mean()

    # Same-hour targets (no shift)
    Y = pd.DataFrame({
        "CF_Solar": df["CF_Solar"],
        "CF_Wind":  df["CF_Wind"],
        "Load_MW":  df["Actual_Load_MW"],
        "Price":    df["Price_EUR_MWh"],
    }, index=df.index)

    XY = X.join(Y).dropna()
    return XY.index, XY[X.columns], Y.loc[XY.index]

# --------------------------------------------------------
# Load engineered data
# --------------------------------------------------------
df = (
    pd.read_csv(ENGINEERED, parse_dates=["Time (UTC)"])
      .set_index("Time (UTC)")
      .sort_index()
)

winsorize_inplace(df, WINS)

# Train/val/test splits
tr = df.loc[SPLIT["train_start"]:SPLIT["train_end"]]
va = df.loc[SPLIT["val_start"]:  SPLIT["val_end"]]
te = df.loc[SPLIT["test_start"]: SPLIT["test_end"]]

t_tr, Xtr, Ytr = make_supervised(tr, LAGS, ROLLS)
t_va, Xva, Yva = make_supervised(va, LAGS, ROLLS)
t_te, Xte, Yte = make_supervised(te, LAGS, ROLLS)

# --------------------------------------------------------
# Scale features (train only) and save scaler
# --------------------------------------------------------
scaler = StandardScaler().fit(Xtr.values)

Xtr_s = scaler.transform(Xtr.values)
Xva_s = scaler.transform(Xva.values)
Xte_s = scaler.transform(Xte.values)

os.makedirs("artifacts", exist_ok=True)
np.savez("artifacts/scaler.npz", scaler=scaler)

# --------------------------------------------------------
# Save datasets for training (Step 03 and onwards)
# --------------------------------------------------------
np.savez_compressed(
    NPZ_PATH,
    X_train=Xtr_s, Y_train=Ytr.values,
    X_val=Xva_s,   Y_val=Yva.values,
    X_test=Xte_s,  Y_test=Yte.values,
    feature_names=np.array(Xtr.columns),
    target_names=np.array(Ytr.columns),
)

print(f"Saved: {NPZ_PATH}")
print("Saved: artifacts/scaler.npz")
