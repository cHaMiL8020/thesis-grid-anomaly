# src/03_train_dcenn_elm.py

#!/usr/bin/env python3
"""
Train Tiny dCeNN encoder + ELM ridge regressor.

Pipeline:
1. Load configs (base.yaml, dcenn.yaml, elm.yaml)
2. Load NPZ dataset from step 02 (X/Y train/val/test, feature/target names)
3. Train TinyDCeNN autoencoder on X_train
4. Encode X_train/X_val/X_test to latent H_train/H_val/H_test
5. Fit ELM ridge regression in latent space (H -> Y) using closed-form
6. Report RMSE on val and test
7. Save artifacts:
    - artifacts/dcenn_encoder.pt  (encoder weights)
    - artifacts/elm_heads.npz     (ridge weights W, target_names)

This version keeps the original behaviour but adds:
- Input validation and clear error messages
- Config parsing via a main() entrypoint
- Basic sanity checks on shapes and config values
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml


DEFAULT_BASE_CONFIG = "configs/base.yaml"
DEFAULT_DCENN_CONFIG = "configs/dcenn.yaml"
DEFAULT_ELM_CONFIG = "configs/elm.yaml"


# ------------------------- helpers -------------------------


def _fail(msg: str) -> None:
    """Print an error message and exit with non-zero status."""
    import sys
    sys.stderr.write(f"[ERROR] {msg}\n")
    sys.exit(1)


def _warn(msg: str) -> None:
    """Print a warning message to stderr."""
    import sys
    sys.stderr.write(f"[WARN] {msg}\n")


def _load_yaml(path: str) -> Dict:
    """Load YAML file and ensure it is a dict."""
    if not os.path.exists(path):
        _fail(f"Config file not found: {path}")
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read config YAML '{path}': {exc}")
    if not isinstance(cfg, dict):
        _fail(f"Config file '{path}' did not contain a YAML mapping (dict).")
    return cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Tiny dCeNN encoder + ELM ridge regressor."
    )
    parser.add_argument(
        "--base-config",
        default=DEFAULT_BASE_CONFIG,
        help=f"Path to base config YAML (default: {DEFAULT_BASE_CONFIG}).",
    )
    parser.add_argument(
        "--dcenn-config",
        default=DEFAULT_DCENN_CONFIG,
        help=f"Path to dCeNN config YAML (default: {DEFAULT_DCENN_CONFIG}).",
    )
    parser.add_argument(
        "--elm-config",
        default=DEFAULT_ELM_CONFIG,
        help=f"Path to ELM config YAML (default: {DEFAULT_ELM_CONFIG}).",
    )
    return parser.parse_args()


def _load_npz_dataset(npz_path: str) -> Tuple[torch.Tensor, ...]:
    """Load X/Y train/val/test + feature_names + target_names from NPZ."""
    if not os.path.exists(npz_path):
        _fail(f"NPZ dataset file not found: {npz_path}")

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to load NPZ dataset '{npz_path}': {exc}")

    required_keys = [
        "X_train", "Y_train",
        "X_val", "Y_val",
        "X_test", "Y_test",
        "feature_names", "target_names",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        _fail(
            f"NPZ dataset is missing required key(s): {missing}. "
            f"Available keys: {list(data.keys())}"
        )

    Xtr = torch.tensor(data["X_train"], dtype=torch.float32)
    Ytr = torch.tensor(data["Y_train"], dtype=torch.float32)
    Xva = torch.tensor(data["X_val"], dtype=torch.float32)
    Yva = torch.tensor(data["Y_val"], dtype=torch.float32)
    Xte = torch.tensor(data["X_test"], dtype=torch.float32)
    Yte = torch.tensor(data["Y_test"], dtype=torch.float32)

    feature_names = [str(f) for f in data["feature_names"]]
    target_names = [str(t) for t in data["target_names"]]

    # Basic shape checks
    if Xtr.ndim != 2 or Xva.ndim != 2 or Xte.ndim != 2:
        _fail("X_train/X_val/X_test must be 2D arrays (N x D).")
    if Ytr.ndim != 2 or Yva.ndim != 2 or Yte.ndim != 2:
        _fail("Y_train/Y_val/Y_test must be 2D arrays (N x T).")

    if Xtr.shape[0] != Ytr.shape[0]:
        _fail("X_train and Y_train must have the same number of samples.")
    if Xva.shape[0] != Yva.shape[0]:
        _fail("X_val and Y_val must have the same number of samples.")
    if Xte.shape[0] != Yte.shape[0]:
        _fail("X_test and Y_test must have the same number of samples.")

    if Xtr.shape[1] != Xva.shape[1] or Xtr.shape[1] != Xte.shape[1]:
        _fail("Feature dimension must be consistent across train/val/test.")

    if Ytr.shape[1] != Yva.shape[1] or Ytr.shape[1] != Yte.shape[1]:
        _fail("Target dimension must be consistent across train/val/test.")

    if Xtr.shape[1] != len(feature_names):
        _warn(
            f"Feature dimension {Xtr.shape[1]} does not match len(feature_names) "
            f"{len(feature_names)}. Continuing, but check your pipeline."
        )

    if Ytr.shape[1] != len(target_names):
        _warn(
            f"Target dimension {Ytr.shape[1]} does not match len(target_names) "
            f"{len(target_names)}. Continuing, but check your pipeline."
        )

    return Xtr, Ytr, Xva, Yva, Xte, Yte, feature_names, target_names


# ------------------------- model definitions -------------------------


class TinyDCeNN(nn.Module):
    """
    Tiny discrete Cellular Neural Network encoder.

    - Input: Dense vector of dimension in_dim
    - Projects to enc_dim
    - Applies a masked linear recurrent cell steps times
    - Mask is block-diagonal with blocks of size 'block'
    """

    def __init__(self, in_dim: int, enc_dim: int = 48, steps: int = 2, block: int = 8) -> None:
        super().__init__()
        if enc_dim <= 0 or steps <= 0 or block <= 0:
            _fail(f"TinyDCeNN: enc_dim, steps, and block must be positive. "
                  f"Got enc_dim={enc_dim}, steps={steps}, block={block}")
        if enc_dim % block != 0:
            _warn(
                f"TinyDCeNN: enc_dim ({enc_dim}) is not a multiple of block ({block}). "
                "Some neurons will never connect across blocks."
            )

        self.in_proj = nn.Linear(in_dim, enc_dim)
        self.cell = nn.Linear(enc_dim, enc_dim)
        self.steps = steps

        # Block diagonal mask
        mask = torch.zeros(enc_dim, enc_dim)
        for i in range(0, enc_dim, block):
            mask[i:i + block, i:i + block] = 1.0
        self.register_buffer("mask", mask)

        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.8)
        nn.init.xavier_uniform_(self.cell.weight, gain=0.2)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.cell.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.in_proj(x))
        for _ in range(self.steps):
            W = self.cell.weight * self.mask
            h = torch.tanh(torch.addmm(self.cell.bias, h, W.T) + 0.3 * h)
        return h


# ------------------------- training & ELM -------------------------


def train_autoencoder(
    enc: nn.Module,
    dec: nn.Module,
    Xtr: torch.Tensor,
    device: torch.device,
    ae_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> None:
    """Train the dCeNN autoencoder on X_train."""
    if Xtr.numel() == 0:
        _fail("X_train is empty; cannot train autoencoder.")

    opt = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()

    dl = DataLoader(
        TensorDataset(Xtr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    print("[INFO] Training autoencoder...")
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
            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[AE] Epoch {ep + 1}/{ae_epochs}  loss={mean_loss:.6f}")


def encode_dataset(
    enc: nn.Module,
    X: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Encode a full dataset X using the trained encoder."""
    enc.eval()
    with torch.no_grad():
        H = enc(X.to(device)).cpu().numpy()
    return H


def fit_ridge(H: np.ndarray, Y: np.ndarray, l2: float) -> np.ndarray:
    """
    Closed-form ridge regression:
        W = (H^T H + l2 I)^-1 H^T Y

    Returns:
        W: weight matrix of shape (latent_dim, n_targets)
    """
    if H.shape[0] == 0:
        _fail("fit_ridge: H has zero rows. Check training data.")
    if H.shape[0] != Y.shape[0]:
        _fail("fit_ridge: H and Y must have the same number of rows.")

    HtH = H.T @ H + l2 * np.eye(H.shape[1])
    HtY = H.T @ Y
    try:
        W = np.linalg.solve(HtH, HtY)
    except np.linalg.LinAlgError as exc:
        _warn(f"np.linalg.solve failed ({exc}); falling back to np.linalg.lstsq.")
        W, *_ = np.linalg.lstsq(HtH, HtY, rcond=None)
    return W


def predict(H: np.ndarray, W: np.ndarray) -> np.ndarray:
    return H @ W


def rmse(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((A - B) ** 2, axis=0))


# ---------------------------- main ----------------------------


def main() -> None:
    args = _parse_args()

    base_cfg = _load_yaml(args.base_config)
    dcenn_cfg = _load_yaml(args.dcenn_config)
    elm_cfg = _load_yaml(args.elm_config)

    npz_path = base_cfg.get("npz_path")
    if not isinstance(npz_path, str):
        _fail("base['npz_path'] must be present and a string in base.yaml.")

    # dCeNN hyperparams
    enc_dim = int(dcenn_cfg.get("enc_dim", 48))
    steps = int(dcenn_cfg.get("steps", 2))
    block = int(dcenn_cfg.get("block", 8))
    ae_epochs = int(dcenn_cfg.get("ae_epochs", 15))
    batch_size = int(dcenn_cfg.get("batch_size", 256))
    lr = float(dcenn_cfg.get("lr", 1e-3))
    weight_decay = float(dcenn_cfg.get("weight_decay", 1e-4))
    seed = int(dcenn_cfg.get("seed", 42))

    # ELM hyperparams
    l2 = float(elm_cfg.get("l2", 1e-2))

    # Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load dataset
    (
        Xtr,
        Ytr,
        Xva,
        Yva,
        Xte,
        Yte,
        feature_names,
        target_names,
    ) = _load_npz_dataset(npz_path)

    in_dim = Xtr.shape[1]
    n_targets = Ytr.shape[1]
    print(
        f"[INFO] Loaded dataset from '{npz_path}': "
        f"in_dim={in_dim}, n_targets={n_targets}, "
        f"X_train={tuple(Xtr.shape)}, X_val={tuple(Xva.shape)}, X_test={tuple(Xte.shape)}"
    )

    # Define encoder and decoder
    enc = TinyDCeNN(in_dim, enc_dim=enc_dim, steps=steps, block=block).to(device)
    dec = nn.Linear(enc_dim, in_dim).to(device)

    # Train AE
    train_autoencoder(
        enc=enc,
        dec=dec,
        Xtr=Xtr,
        device=device,
        ae_epochs=ae_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
    )

    # Encode all splits
    Htr = encode_dataset(enc, Xtr, device=device)
    Hva = encode_dataset(enc, Xva, device=device)
    Hte = encode_dataset(enc, Xte, device=device)

    # Fit ELM
    W = fit_ridge(Htr, Ytr.numpy(), l2=l2)

    # Predictions
    Yva_pred = predict(Hva, W)
    Yte_pred = predict(Hte, W)

    # RMSE
    val_rmse = rmse(Yva_pred, Yva.numpy())
    test_rmse = rmse(Yte_pred, Yte.numpy())

    print("\n[INFO] Validation RMSE per target:")
    for name, r in zip(target_names, val_rmse):
        print(f"  {name:>10s}: {r:.5f}")

    print("\n[INFO] Test RMSE per target:")
    for name, r in zip(target_names, test_rmse):
        print(f"  {name:>10s}: {r:.5f}")

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)

    try:
        torch.save(enc.state_dict(), "artifacts/dcenn_encoder.pt")
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to save encoder state_dict to 'artifacts/dcenn_encoder.pt': {exc}")

    try:
        np.savez_compressed(
            "artifacts/elm_heads.npz",
            W=W,
            target_names=np.array(target_names),
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to save ELM heads to 'artifacts/elm_heads.npz': {exc}")

    print("\n[INFO] Saved encoder + ELM heads to 'artifacts/'.")
    print("[INFO] Training step (03_train_dcenn_elm.py) completed successfully.")


if __name__ == "__main__":
    main()
