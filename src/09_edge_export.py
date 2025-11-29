# src/09_edge_export.py

#!/usr/bin/env python3
"""
Export dCeNN+ELM + scaler + feature metadata into a compact NPZ bundle
for edge deployment, plus a small pure-NumPy runtime stub.

Reads:
  - configs/base.yaml       (npz_path)
  - configs/dcenn.yaml      (enc_dim, steps, block)
  - {npz_path}              (X/Y splits, feature_names, target_names)
  - artifacts/scaler.npz    (StandardScaler from step 02)
  - artifacts/dcenn_encoder.pt   (TinyDCeNN state_dict from step 03)
  - artifacts/elm_heads.npz      (W, target_names from step 03)

Writes:
  - edge/model_bundle.npz
  - edge/runtime_infer.py
"""

import argparse
import os
from typing import Dict, Any

import numpy as np
import torch
import yaml

DEFAULT_BASE_CONFIG = "configs/base.yaml"
DEFAULT_DCENN_CONFIG = "configs/dcenn.yaml"
DEFAULT_SCALER_PATH = "artifacts/scaler.npz"
DEFAULT_ENCODER_PATH = "artifacts/dcenn_encoder.pt"
DEFAULT_ELM_HEADS_PATH = "artifacts/elm_heads.npz"
DEFAULT_EDGE_DIR = "edge"


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------


def _fail(msg: str) -> None:
    import sys
    sys.stderr.write(f"[ERROR] {msg}\n")
    sys.exit(1)


def _warn(msg: str) -> None:
    import sys
    sys.stderr.write(f"[WARN] {msg}\n")


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        _fail(f"Config file not found: {path}")
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read config YAML '{path}': {exc}")
    if not isinstance(cfg, dict):
        _fail(f"Config '{path}' did not contain a YAML mapping (dict).")
    return cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export dCeNN+ELM model bundle for edge deployment."
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
        "--scaler-path",
        default=DEFAULT_SCALER_PATH,
        help=f"Path to scaler NPZ (default: {DEFAULT_SCALER_PATH}).",
    )
    parser.add_argument(
        "--encoder-path",
        default=DEFAULT_ENCODER_PATH,
        help=f"Path to encoder state_dict (default: {DEFAULT_ENCODER_PATH}).",
    )
    parser.add_argument(
        "--elm-heads-path",
        default=DEFAULT_ELM_HEADS_PATH,
        help=f"Path to ELM heads NPZ (default: {DEFAULT_ELM_HEADS_PATH}).",
    )
    parser.add_argument(
        "--edge-dir",
        default=DEFAULT_EDGE_DIR,
        help=f"Output directory for edge bundle and runtime (default: {DEFAULT_EDGE_DIR}).",
    )
    return parser.parse_args()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------------
# Main
# --------------------------------------------------------


def main() -> None:
    args = _parse_args()

    base_cfg = _load_yaml(args.base_config)
    dcenn_cfg = _load_yaml(args.dcenn_config)

    if "npz_path" not in base_cfg:
        _fail("base.yaml must contain 'npz_path' pointing to the training NPZ file.")

    npz_path = base_cfg["npz_path"]

    if not os.path.exists(npz_path):
        _fail(f"Training NPZ file not found: {npz_path}")

    enc_dim = int(dcenn_cfg.get("enc_dim", 48))
    steps = int(dcenn_cfg.get("steps", 2))
    block = int(dcenn_cfg.get("block", 8))

    print(f"[INFO] Using npz_path={npz_path}")
    print(f"[INFO] dCeNN config: enc_dim={enc_dim}, steps={steps}, block={block}")

    # ----------------------------------------------------
    # 1) Load dataset NPZ from step 02 (for feature/target names)
    # ----------------------------------------------------
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to load NPZ '{npz_path}': {exc}")

    if "feature_names" not in data or "target_names" not in data:
        _fail(
            f"NPZ '{npz_path}' must contain 'feature_names' and 'target_names'. "
            f"Keys available: {list(data.keys())}"
        )

    feature_names = data["feature_names"]
    target_names = data["target_names"]

    in_dim = int(len(feature_names))
    print(f"[INFO] Feature dimension in_dim={in_dim}, targets={list(target_names)}")

    # ----------------------------------------------------
    # 2) Load scaler from artifacts/scaler.npz (step 02)
    # ----------------------------------------------------
    scaler_path = args.scaler_path
    if not os.path.exists(scaler_path):
        _fail(f"Scaler NPZ not found: {scaler_path}")

    try:
        scaler_obj = np.load(scaler_path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to load scaler NPZ '{scaler_path}': {exc}")

    if "scaler" not in scaler_obj:
        _fail(
            f"'scaler' key not found in '{scaler_path}'. "
            "Ensure step 02 saved np.savez('artifacts/scaler.npz', scaler=scaler)."
        )

    scaler = scaler_obj["scaler"][()]
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        _fail(
            "Loaded scaler object is missing 'mean_' or 'scale_' attributes. "
            "Expected sklearn.preprocessing.StandardScaler."
        )

    scaler_mean = np.asarray(scaler.mean_, dtype=np.float32)
    scaler_scale = np.asarray(scaler.scale_, dtype=np.float32)

    if scaler_mean.shape[0] != in_dim or scaler_scale.shape[0] != in_dim:
        _warn(
            f"Scaler dimension mismatch: scaler_mean.shape={scaler_mean.shape}, "
            f"in_dim={in_dim}. Check that scaler and feature_names are from the same run."
        )

    # ----------------------------------------------------
    # 3) Load encoder weights (TinyDCeNN) from step 03
    # ----------------------------------------------------
    encoder_path = args.encoder_path
    if not os.path.exists(encoder_path):
        _fail(f"Encoder state_dict not found: {encoder_path}")

    try:
        sd = torch.load(encoder_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to load encoder state_dict '{encoder_path}': {exc}")

    required_keys = ["in_proj.weight", "in_proj.bias", "cell.weight", "cell.bias"]
    missing_keys = [k for k in required_keys if k not in sd]
    if missing_keys:
        _fail(
            f"Encoder state_dict '{encoder_path}' is missing keys: {missing_keys}. "
            "Ensure it was trained with TinyDCeNN from step 03."
        )

    in_W = sd["in_proj.weight"].cpu().numpy().astype(np.float32)
    in_b = sd["in_proj.bias"].cpu().numpy().astype(np.float32)
    cell_W = sd["cell.weight"].cpu().numpy().astype(np.float32)
    cell_b = sd["cell.bias"].cpu().numpy().astype(np.float32)

    if in_W.shape[1] != in_dim:
        _warn(
            f"in_proj.weight.shape={in_W.shape} but in_dim={in_dim}. "
            "Feature ordering mismatch may exist."
        )
    if in_W.shape[0] != enc_dim or cell_W.shape != (enc_dim, enc_dim):
        _warn(
            f"Encoder dimensions do not match enc_dim={enc_dim}: "
            f"in_W.shape={in_W.shape}, cell_W.shape={cell_W.shape}"
        )

    # ----------------------------------------------------
    # 4) Load ELM head from step 03
    # ----------------------------------------------------
    elm_path = args.elm_heads_path
    if not os.path.exists(elm_path):
        _fail(f"ELM heads NPZ not found: {elm_path}")

    try:
        Wobj = np.load(elm_path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to load ELM heads NPZ '{elm_path}': {exc}")

    if "W" not in Wobj:
        _fail(
            f"'W' key not found in '{elm_path}'. "
            "Ensure step 03 saved np.savez_compressed('elm_heads.npz', W=W, target_names=...)."
        )
    W = Wobj["W"].astype(np.float32)

    if W.ndim != 2 or W.shape[0] != enc_dim:
        _warn(
            f"ELM weight matrix W has shape {W.shape} but enc_dim={enc_dim}. "
            "Check that encoder and ELM heads come from the same training run."
        )

    # --------------------------------------------------------
    # 5) Build connectivity mask (same as TinyDCeNN)
    # --------------------------------------------------------
    mask = np.zeros((enc_dim, enc_dim), dtype=np.float32)
    for i in range(0, enc_dim, block):
        mask[i:i + block, i:i + block] = 1.0

    # --------------------------------------------------------
    # 6) Save edge bundle
    # --------------------------------------------------------
    edge_dir = args.edge_dir
    _ensure_dir(edge_dir)

    bundle_path = os.path.join(edge_dir, "model_bundle.npz")
    np.savez_compressed(
        bundle_path,
        in_W=in_W,
        in_b=in_b,
        cell_W=cell_W,
        cell_b=cell_b,
        mask=mask,
        steps=np.array([steps], dtype=np.int32),
        enc_dim=np.array([enc_dim], dtype=np.int32),
        block=np.array([block], dtype=np.int32),
        in_dim=np.array([in_dim], dtype=np.int32),
        W=W,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_names=np.array(feature_names),
        target_names=np.array(target_names),
    )

    print(f"[INFO] Saved edge bundle to '{bundle_path}'")

    # --------------------------------------------------------
    # 7) Write runtime inference stub (pure NumPy)
    # --------------------------------------------------------
    runtime_path = os.path.join(edge_dir, "runtime_infer.py")
    with open(runtime_path, "w") as f:
        f.write(
            """\"\"\"Minimal runtime for dCeNN+ELM bundle on edge devices.

Usage:
    import numpy as np
    from runtime_infer import load_bundle, predict, predict_dict

    bundle = load_bundle("model_bundle.npz")
    features = {
        "Actual_Load_MW":  5000.0,
        "Solar_MW":        200.0,
        "Wind_MW":         800.0,
        # ...
    }

    y = predict(features, bundle)       # np.ndarray aligned with target_names
    y_dict = predict_dict(features, bundle)

\"\"\"


import numpy as np


def load_bundle(path: str = "model_bundle.npz"):
    \"\"\"Load the edge model bundle npz as a dict-like object.\"\"\"
    return np.load(path, allow_pickle=True)


def tanh(x):
    return np.tanh(x)


def dcenn_forward(x, b):
    \"\"\"Forward pass of Tiny dCeNN encoder.

    Args:
        x: np.ndarray of shape (F,)
        b: np.load bundle with keys:
           - in_W, in_b
           - cell_W, cell_b
           - mask
           - steps

    Returns:
        h: np.ndarray of shape (enc_dim,)
    \"\"\"
    in_W = b["in_W"]
    in_b = b["in_b"]
    cell_W = b["cell_W"]
    cell_b = b["cell_b"]
    mask = b["mask"]
    steps = int(b["steps"][0])

    h = tanh(x @ in_W.T + in_b)
    Wc = cell_W * mask
    for _ in range(steps):
        h = tanh(h @ Wc.T + cell_b + 0.3 * h)
    return h


def _build_feature_vector(features_dict, b):
    \"\"\"Build standardized feature vector from dict and bundle metadata.\"\"\"
    feat_names = b["feature_names"].tolist()
    x = np.array([features_dict.get(k, 0.0) for k in feat_names], dtype=np.float32)

    scaler_mean = b["scaler_mean"]
    scaler_scale = b["scaler_scale"]
    x = (x - scaler_mean) / (scaler_scale + 1e-9)
    return x


def predict(features_dict, b):
    \"\"\"Predict targets given a feature dict and loaded bundle.

    Args:
        features_dict: mapping {feature_name: value}
        b: loaded bundle from np.load("model_bundle.npz", allow_pickle=True)

    Returns:
        np.ndarray of shape (num_targets,) aligned with b["target_names"].
    \"\"\"
    x = _build_feature_vector(features_dict, b)
    h = dcenn_forward(x, b)
    W = b["W"]
    y = h @ W
    return y


def predict_dict(features_dict, b):
    \"\"\"Predict and return a dict {target_name: value}.\"\"\"
    y = predict(features_dict, b)
    target_names = b["target_names"].tolist()
    return {name: float(val) for name, val in zip(target_names, y)}
"""
        )

    print(f"[INFO] Wrote runtime stub to '{runtime_path}'")
    print("[INFO] Edge export (09_edge_export.py) completed.")


if __name__ == "__main__":
    main()

