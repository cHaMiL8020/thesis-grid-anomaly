import os, yaml, numpy as np, pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

base = yaml.safe_load(open("configs/base.yaml"))
f_cfg = yaml.safe_load(open("configs/features.yaml"))
dcfg  = yaml.safe_load(open("configs/dcenn.yaml"))

ENGINEERED = base["engineered_csv"]
SPLIT = base["split"]
HORIZON = int(base.get("horizon",1))
LAGS  = f_cfg["lags"]; ROLLS = f_cfg["rolls"]

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
    return XY.index, X.columns.tolist(), XY.iloc[:, :X.shape[1]], XY.iloc[:, X.shape[1]:]

# rebuild train for scaler/feature order
df = pd.read_csv(ENGINEERED, parse_dates=["Time (UTC)"]).set_index("Time (UTC)").sort_index()
winsor_cols = yaml.safe_load(open("configs/features.yaml"))["winsorize_cols"]
winsorize_inplace(df, winsor_cols)
train = df.loc[SPLIT["train_start"]:SPLIT["train_end"]]
_, feat_names, Xtr, _ = make_supervised(train, HORIZON, tuple(LAGS), tuple(ROLLS))
scaler = StandardScaler().fit(Xtr.values)

# load encoder + ELM
state = torch.load("artifacts/dcenn_encoder.pt", map_location="cpu")
Wobj  = np.load("artifacts/elm_heads.npz", allow_pickle=True)
W     = Wobj["W"]; target_names = [t for t in Wobj["target_names"]]

sd = state["state_dict"]
in_W = sd["in_proj.weight"].cpu().numpy().astype(np.float32)
in_b = sd["in_proj.bias"].cpu().numpy().astype(np.float32)
cell_W = sd["cell.weight"].cpu().numpy().astype(np.float32)
cell_b = sd["cell.bias"].cpu().numpy().astype(np.float32)

enc_dim = int(dcfg.get("enc_dim", 48))
steps   = int(dcfg.get("steps", 2))
block   = int(dcfg.get("block", 8))

mask = np.zeros((enc_dim, enc_dim), dtype=np.float32)
for i in range(0, enc_dim, block):
    mask[i:i+block, i:i+block] = 1.0

os.makedirs("edge", exist_ok=True)
np.savez_compressed(
  "edge/model_bundle.npz",
  in_W=in_W, in_b=in_b, cell_W=cell_W, cell_b=cell_b,
  mask=mask, steps=np.array([steps], dtype=np.int32),
  W=W.astype(np.float32),
  scaler_mean=scaler.mean_.astype(np.float32),
  scaler_scale=scaler.scale_.astype(np.float32),
  feature_names=np.array(feat_names),
  target_names=np.array(target_names)
)

with open("edge/runtime_infer.py","w") as f:
    f.write("""\
import numpy as np
def tanh(x): return np.tanh(x)
def dcenn_forward(x, b):
    h = tanh(x @ b["in_W"].T + b["in_b"])
    Wc = b["cell_W"] * b["mask"]
    for _ in range(int(b["steps"][0])):
        h = tanh(h @ Wc.T + b["cell_b"] + 0.3*h)
    return h
def predict(features_dict, b):
    feat_names = b["feature_names"].tolist()
    x = np.array([features_dict.get(k, 0.0) for k in feat_names], dtype=np.float32)
    x = (x - b["scaler_mean"]) / (b["scaler_scale"] + 1e-9)
    h = dcenn_forward(x, b)
    return h @ b["W"]
""")
print("Saved edge/model_bundle.npz and edge/runtime_infer.py")
