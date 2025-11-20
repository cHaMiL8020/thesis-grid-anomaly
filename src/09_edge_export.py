# src/09_export_edge_bundle.py  (or keep as 09_edge_export.py)

import os
import yaml
import numpy as np
import torch

# --------------------------------------------------------
# Load configs
# --------------------------------------------------------
with open("configs/base.yaml") as f:
    base = yaml.safe_load(f)

with open("configs/dcenn.yaml") as f:
    dcfg = yaml.safe_load(f)

NPZ_PATH = base["npz_path"]

enc_dim = int(dcfg.get("enc_dim", 48))
steps   = int(dcfg.get("steps", 2))
block   = int(dcfg.get("block", 8))

# --------------------------------------------------------
# Load training artifacts
# --------------------------------------------------------
# 1) Feature & target names from Step 02 npz
data = np.load(NPZ_PATH, allow_pickle=True)
feature_names = data["feature_names"]
target_names  = data["target_names"]

# 2) Scaler from artifacts/scaler.npz (saved in Step 02)
scaler_obj = np.load("artifacts/scaler.npz", allow_pickle=True)
scaler = scaler_obj["scaler"][()]
scaler_mean  = scaler.mean_.astype(np.float32)
scaler_scale = scaler.scale_.astype(np.float32)

# 3) Encoder weights from Step 03
sd = torch.load("artifacts/dcenn_encoder.pt", map_location="cpu")
in_W   = sd["in_proj.weight"].cpu().numpy().astype(np.float32)
in_b   = sd["in_proj.bias"].cpu().numpy().astype(np.float32)
cell_W = sd["cell.weight"].cpu().numpy().astype(np.float32)
cell_b = sd["cell.bias"].cpu().numpy().astype(np.float32)

# 4) ELM head from Step 03
Wobj = np.load("artifacts/elm_heads.npz", allow_pickle=True)
W    = Wobj["W"].astype(np.float32)
# target_names from Wobj should match data["target_names"], but we use data's for consistency

# --------------------------------------------------------
# Build connectivity mask (same as TinyDCeNN)
# --------------------------------------------------------
mask = np.zeros((enc_dim, enc_dim), dtype=np.float32)
for i in range(0, enc_dim, block):
    mask[i:i+block, i:i+block] = 1.0

# --------------------------------------------------------
# Save edge bundle
# --------------------------------------------------------
os.makedirs("edge", exist_ok=True)

np.savez_compressed(
    "edge/model_bundle.npz",
    in_W=in_W,
    in_b=in_b,
    cell_W=cell_W,
    cell_b=cell_b,
    mask=mask,
    steps=np.array([steps], dtype=np.int32),
    W=W,
    scaler_mean=scaler_mean,
    scaler_scale=scaler_scale,
    feature_names=np.array(feature_names),
    target_names=np.array(target_names),
)

# --------------------------------------------------------
# Write runtime inference stub
# --------------------------------------------------------
with open("edge/runtime_infer.py", "w") as f:
    f.write("""\
import numpy as np

def tanh(x):
    return np.tanh(x)

def dcenn_forward(x, b):
    # x: (F,)
    h = tanh(x @ b["in_W"].T + b["in_b"])
    Wc = b["cell_W"] * b["mask"]
    for _ in range(int(b["steps"][0])):
        h = tanh(h @ Wc.T + b["cell_b"] + 0.3 * h)
    return h

def predict(features_dict, b):
    # features_dict: {feature_name: value}
    feat_names = b["feature_names"].tolist()
    x = np.array([features_dict.get(k, 0.0) for k in feat_names], dtype=np.float32)
    x = (x - b["scaler_mean"]) / (b["scaler_scale"] + 1e-9)
    h = dcenn_forward(x, b)
    y = h @ b["W"]
    # returns vector [CF_Solar, CF_Wind, Load_MW, Price]
    return y
""")

print("Saved edge/model_bundle.npz and edge/runtime_infer.py")
