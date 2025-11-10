import numpy as np, torch, yaml, os
from torch import nn

base = yaml.safe_load(open("configs/base.yaml"))
dcfg = yaml.safe_load(open("configs/dcenn.yaml"))
ecfg = yaml.safe_load(open("configs/elm.yaml"))

# ---- Type-safe pulls from YAML (handles quoted numbers) ----
enc_dim         = int(dcfg.get("enc_dim", 48))
steps           = int(dcfg.get("steps", 2))
block           = int(dcfg.get("block", 8))
ae_epochs       = int(dcfg.get("ae_epochs", 3))
subsample_train = int(dcfg.get("subsample_train", 20000))
seed            = int(dcfg.get("seed", 42))
lr              = float(dcfg.get("lr", 1e-3))
weight_decay    = float(dcfg.get("weight_decay", 1e-4))
l2              = float(ecfg.get("l2", 1e-2))

data = np.load(base["npz_path"], allow_pickle=True)
Xtr = torch.tensor(data["X_train"], dtype=torch.float32)
Ytr = torch.tensor(data["Y_train"], dtype=torch.float32)
Xva = torch.tensor(data["X_val"],   dtype=torch.float32)
Yva = torch.tensor(data["Y_val"],   dtype=torch.float32)
Xte = torch.tensor(data["X_test"],  dtype=torch.float32)
Yte = torch.tensor(data["Y_test"],  dtype=torch.float32)
t_names = [t for t in data["target_names"]]

g = torch.Generator().manual_seed(seed)

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

enc = TinyDCeNN(Xtr.shape[1], enc_dim=enc_dim, steps=steps, block=block)
dec = nn.Linear(enc_dim, Xtr.shape[1], bias=False)
opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),
                       lr=lr, weight_decay=weight_decay)
mse = nn.MSELoss()

# AE prefit (subsample for speed)
N = Xtr.shape[0]
subN = min(N, subsample_train)
idx = torch.randperm(N, generator=g)[:subN]
for ep in range(ae_epochs):
    opt.zero_grad()
    H = enc(Xtr[idx]); Xhat = dec(H)
    loss = mse(Xhat, Xtr[idx]); loss.backward(); opt.step()
    print(f"[AE] {ep+1}/{ae_epochs} loss={loss.item():.5f}")

enc.eval()
with torch.no_grad():
    Htr, Hva, Hte = enc(Xtr).numpy(), enc(Xva).numpy(), enc(Xte).numpy()

def fit_ridge(H, Y, l2=1e-2):
    HtH = H.T @ H + l2*np.eye(H.shape[1], dtype=np.float32)
    HtY = H.T @ Y
    return np.linalg.solve(HtH, HtY)
def predict(H,W): return H @ W
def rmse(A,B): return np.sqrt(np.mean((A-B)**2, axis=0))

W = fit_ridge(Htr, Ytr.numpy(), l2=l2)
Yva_pred, Yte_pred = predict(Hva,W), predict(Hte,W)

val_rmse, test_rmse = rmse(Yva_pred, Yva.numpy()), rmse(Yte_pred, Yte.numpy())
print("\nValidation RMSE:");  print(dict(zip(t_names, map(float,val_rmse))))
print("\nTest RMSE:");        print(dict(zip(t_names, map(float,test_rmse))))

os.makedirs("artifacts", exist_ok=True)
torch.save({"state_dict": enc.state_dict()}, "artifacts/dcenn_encoder.pt")
np.savez_compressed("artifacts/elm_heads.npz", W=W, target_names=np.array(t_names))
