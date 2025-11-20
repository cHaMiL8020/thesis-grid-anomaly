# src/03_train_dcenn_elm.py

import os
import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------
# Load configs
# --------------------------------------------------
with open("configs/base.yaml") as f:
    base = yaml.safe_load(f)
with open("configs/dcenn.yaml") as f:
    dcfg = yaml.safe_load(f)
with open("configs/elm.yaml") as f:
    ecfg = yaml.safe_load(f)

enc_dim = int(dcfg.get("enc_dim", 48))
steps   = int(dcfg.get("steps", 2))
block   = int(dcfg.get("block", 8))
ae_epochs = int(dcfg.get("ae_epochs", 15))
batch_size = int(dcfg.get("batch_size", 256))
lr = float(dcfg.get("lr", 1e-3))
weight_decay = float(dcfg.get("weight_decay", 1e-4))
l2 = float(ecfg.get("l2", 1e-2))
seed = int(dcfg.get("seed", 42))

# --------------------------------------------------
# Load data
# --------------------------------------------------
data = np.load(base["npz_path"], allow_pickle=True)
Xtr = torch.tensor(data["X_train"], dtype=torch.float32)
Ytr = torch.tensor(data["Y_train"], dtype=torch.float32)
Xva = torch.tensor(data["X_val"], dtype=torch.float32)
Yva = torch.tensor(data["Y_val"], dtype=torch.float32)
Xte = torch.tensor(data["X_test"], dtype=torch.float32)
Yte = torch.tensor(data["Y_test"], dtype=torch.float32)

feature_names = [f for f in data["feature_names"]]
target_names  = [t for t in data["target_names"]]

torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Define Tiny DCeNN encoder
# --------------------------------------------------
class TinyDCeNN(nn.Module):
    def __init__(self, in_dim, enc_dim=48, steps=2, block=8):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, enc_dim)
        self.cell = nn.Linear(enc_dim, enc_dim)
        self.steps = steps

        # Block diagonal mask
        mask = torch.zeros(enc_dim, enc_dim)
        for i in range(0, enc_dim, block):
            mask[i:i+block, i:i+block] = 1.0
        self.register_buffer("mask", mask)

        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.8)
        nn.init.xavier_uniform_(self.cell.weight, gain=0.2)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.cell.bias)

    def forward(self, x):
        h = torch.tanh(self.in_proj(x))
        for _ in range(self.steps):
            W = self.cell.weight * self.mask
            h = torch.tanh(torch.addmm(self.cell.bias, h, W.T) + 0.3 * h)
        return h

# --------------------------------------------------
# Autoencoder
# --------------------------------------------------
enc = TinyDCeNN(Xtr.shape[1], enc_dim, steps, block).to(device)
dec = nn.Linear(enc_dim, Xtr.shape[1]).to(device)

opt = torch.optim.Adam(
    list(enc.parameters()) + list(dec.parameters()),
    lr=lr, weight_decay=weight_decay,
)
loss_fn = nn.MSELoss()

dl = DataLoader(TensorDataset(Xtr), batch_size=batch_size, shuffle=True)

print("Training autoencoder...")
for ep in range(ae_epochs):
    enc.train()
    dec.train()
    losses = []

    for (xb,) in dl:
        xb = xb.to(device)
        opt.zero_grad()
        h = enc(xb)
        xhat = dec(h)
        loss = loss_fn(xhat, xb)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"[AE] Epoch {ep+1}/{ae_epochs}  loss={np.mean(losses):.5f}")

# --------------------------------------------------
# Encode full dataset
# --------------------------------------------------
enc.eval()
with torch.no_grad():
    Htr = enc(Xtr.to(device)).cpu().numpy()
    Hva = enc(Xva.to(device)).cpu().numpy()
    Hte = enc(Xte.to(device)).cpu().numpy()

# --------------------------------------------------
# Fit ELM ridge regression
# --------------------------------------------------
def fit_ridge(H, Y, l2):
    HtH = H.T @ H + l2 * np.eye(H.shape[1])
    HtY = H.T @ Y
    return np.linalg.solve(HtH, HtY)

W = fit_ridge(Htr, Ytr.numpy(), l2)

def predict(H, W):
    return H @ W

Yva_pred = predict(Hva, W)
Yte_pred = predict(Hte, W)

# RMSE
def rmse(A, B):
    return np.sqrt(np.mean((A - B)**2, axis=0))

print("\nValidation RMSE:")
print(dict(zip(target_names, rmse(Yva_pred, Yva.numpy()))))

print("\nTest RMSE:")
print(dict(zip(target_names, rmse(Yte_pred, Yte.numpy()))))

# --------------------------------------------------
# Save artifacts
# --------------------------------------------------
os.makedirs("artifacts", exist_ok=True)
torch.save(enc.state_dict(), "artifacts/dcenn_encoder.pt")
np.savez_compressed(
    "artifacts/elm_heads.npz",
    W=W,
    target_names=np.array(target_names),
)

print("\nSaved encoder + ELM heads.")
