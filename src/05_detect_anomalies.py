# src/05_detect_anomalies.py

#!/usr/bin/env python3
"""
Detect anomalies on the test set using the trained dCeNN+ELM model and
calibrated thresholds.

Pipeline:
1. Load configs (base.yaml, features.yaml, thresholds.yaml, dcenn.yaml)
2. Rebuild supervised X/Y for train & test from engineered CSV
3. Load scaler (from step 02), encoder (from step 03), ELM heads, thresholds
4. Compute predictions, residuals and |residuals| on test set
5. Apply bucketed thresholds (none | hour | holiday)
6. Compute:
     - per-target anomalies
     - combined anomaly score
     - global flags (any_anomaly_flag, n_targets_anom)
7. Save anomaly table + combined score plot for 2022
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

# Non-interactive backend for headless environments
plt.switch_backend("Agg")

DEFAULT_BASE_CONFIG = "configs/base.yaml"
DEFAULT_FEATURES_CONFIG = "configs/features.yaml"
DEFAULT_THRESHOLDS_CONFIG = "configs/thresholds.yaml"
DEFAULT_DCENN_CONFIG = "configs/dcenn.yaml"

SCALER_PATH_DEFAULT = "artifacts/scaler.npz"
ENCODER_PATH_DEFAULT = "artifacts/dcenn_encoder.pt"
ELM_HEADS_PATH_DEFAULT = "artifacts/elm_heads.npz"
THRESHOLDS_JSON_DEFAULT = "artifacts/thresholds.json"


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
    """Load a YAML file and ensure it is a dict."""
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


def _ensure_parent_dir(path: str) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect anomalies on the test set using dCeNN+ELM + thresholds."
    )
    parser.add_argument(
        "--base-config",
        default=DEFAULT_BASE_CONFIG,
        help=f"Path to base config YAML (default: {DEFAULT_BASE_CONFIG}).",
    )
    parser.add_argument(
        "--features-config",
        default=DEFAULT_FEATURES_CONFIG,
        help=f"Path to features config YAML (default: {DEFAULT_FEATURES_CONFIG}).",
    )
    parser.add_argument(
        "--thresholds-config",
        default=DEFAULT_THRESHOLDS_CONFIG,
        help=f"Path to thresholds config YAML (default: {DEFAULT_THRESHOLDS_CONFIG}).",
    )
    parser.add_argument(
        "--dcenn-config",
        default=DEFAULT_DCENN_CONFIG,
        help=f"Path to dCeNN config YAML (default: {DEFAULT_DCENN_CONFIG}).",
    )
    parser.add_argument(
        "--scaler-path",
        default=SCALER_PATH_DEFAULT,
        help=f"Path to scaler NPZ from step 02 (default: {SCALER_PATH_DEFAULT}).",
    )
    parser.add_argument(
        "--encoder-path",
        default=ENCODER_PATH_DEFAULT,
        help=f"Path to trained encoder state_dict (default: {ENCODER_PATH_DEFAULT}).",
    )
    parser.add_argument(
        "--elm-heads-path",
        default=ELM_HEADS_PATH_DEFAULT,
        help=f"Path to NPZ with ELM heads (default: {ELM_HEADS_PATH_DEFAULT}).",
    )
    parser.add_argument(
        "--thresholds-json",
        default=THRESHOLDS_JSON_DEFAULT,
        help=f"Path to thresholds JSON (default: {THRESHOLDS_JSON_DEFAULT}).",
    )
    return parser.parse_args()


# --------------------------------------------------------
# Feature list (must match Step 02 EXACTLY)
# --------------------------------------------------------

FEATS_BASE: List[str] = [
    "Actual_Load_MW",
    "Solar_MW",
    "Wind_MW",
    "Price_EUR_MWh",
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "wind_speed_10m (m/s)",
    "surface_pressure (hPa)",
    "shortwave_radiation (W/m²)",
    "air_density_kgm3",
    "wind_speed_100m (m/s)",
    "wind_power_proxy",
    "pv_proxy",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_public_holiday",
    "is_weekend",
    "is_special_day",
]


# ------------------------- models -------------------------


class TinyDCeNN(nn.Module):
    """
    Same Tiny dCeNN encoder as in step 03.
    """

    def __init__(self, in_dim: int, enc_dim: int = 48, steps: int = 2, block: int = 8) -> None:
        super().__init__()
        if enc_dim <= 0 or steps <= 0 or block <= 0:
            _fail(
                f"TinyDCeNN: enc_dim, steps, and block must be positive. "
                f"Got enc_dim={enc_dim}, steps={steps}, block={block}"
            )

        self.in_proj = nn.Linear(in_dim, enc_dim)
        self.cell = nn.Linear(enc_dim, enc_dim)
        self.steps = steps

        mask = torch.zeros(enc_dim, enc_dim)
        for i in range(0, enc_dim, block):
            mask[i : i + block, i : i + block] = 1.0
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


# ------------------------- core utils -------------------------


def winsorize_inplace(df: pd.DataFrame, cols: List[str], lo: float = 0.01, hi: float = 0.99) -> None:
    """
    Winsorize columns in-place to [lo, hi] quantiles (consistent with steps 02 & 04).
    """
    for c in cols:
        if c not in df.columns:
            _warn(
                f"winsorize_inplace: column '{c}' not found in DataFrame; skipping."
            )
            continue
        series = df[c].dropna()
        if series.empty:
            _warn(
                f"winsorize_inplace: column '{c}' is empty/NaN-only; skipping."
            )
            continue
        lo_q, hi_q = series.quantile([lo, hi])
        df[c] = df[c].clip(lo_q, hi_q)


def make_supervised(
    df: pd.DataFrame,
    lags: List[int],
    rolls: List[int],
) -> Tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    """
    Same-hour supervised dataset (NO horizon shift).
    Must match Step 02 logic.
    """
    missing_feats = [c for c in FEATS_BASE if c not in df.columns]
    if missing_feats:
        _fail(
            f"Missing required base feature(s) in engineered DataFrame: {missing_feats}. "
            f"Available columns: {list(df.columns)}"
        )

    X = df[FEATS_BASE].copy()
    core = [
        "Actual_Load_MW",
        "Solar_MW",
        "Wind_MW",
        "Price_EUR_MWh",
        "temperature_2m (°C)",
        "wind_speed_10m (m/s)",
        "shortwave_radiation (W/m²)",
    ]

    for c in core:
        if c not in df.columns:
            _fail(
                f"Core feature '{c}' missing from engineered DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        for L in lags:
            X[f"{c}_lag{L}"] = df[c].shift(L)
        for R in rolls:
            X[f"{c}_rmean{R}"] = df[c].rolling(
                R, min_periods=int(0.7 * R)
            ).mean()

    target_cols = ["CF_Solar", "CF_Wind", "Actual_Load_MW", "Price_EUR_MWh"]
    for tc in target_cols:
        if tc not in df.columns:
            _fail(
                f"Target column '{tc}' missing from engineered DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    Y = pd.DataFrame(
        {
            "CF_Solar": df["CF_Solar"],
            "CF_Wind": df["CF_Wind"],
            "Load_MW": df["Actual_Load_MW"],
            "Price": df["Price_EUR_MWh"],
        },
        index=df.index,
    )

    XY = X.join(Y).dropna()
    if XY.empty:
        _fail(
            "make_supervised: resulting supervised DataFrame is empty after "
            "adding lags/rolls and dropping NaNs. Check lags/rolls or date range."
        )

    return XY.index, XY[X.columns], Y.loc[XY.index]


def _load_engineered(path: str) -> pd.DataFrame:
    """Load engineered CSV and ensure 'Time (UTC)' index exists."""
    if not os.path.exists(path):
        _fail(f"Engineered CSV not found: {path}")
    try:
        df = pd.read_csv(path, parse_dates=["Time (UTC)"])
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read engineered CSV '{path}': {exc}")

    if "Time (UTC)" not in df.columns:
        _fail(
            "Engineered CSV must contain 'Time (UTC)' column. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.set_index("Time (UTC)").sort_index()
    if df.empty:
        _fail(f"Engineered DataFrame from '{path}' is empty.")

    if df.index.has_duplicates:
        _warn(
            "Engineered DataFrame index has duplicate timestamps. "
            "Keeping first occurrence per timestamp."
        )
        df = df[~df.index.duplicated(keep="first")]

    return df


def _split_train_test(df: pd.DataFrame, split_cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/test using base['split'].
    """
    required_keys = ["train_start", "train_end", "test_start", "test_end"]
    missing = [k for k in required_keys if k not in split_cfg]
    if missing:
        _fail(
            f"Missing split keys in base['split']: {missing}. "
            "Expected at least train_start, train_end, test_start, test_end."
        )

    tr = df.loc[split_cfg["train_start"] : split_cfg["train_end"]]
    te = df.loc[split_cfg["test_start"] : split_cfg["test_end"]]

    if tr.empty or te.empty:
        _fail(
            "Train or test split is empty. "
            f"Train range: {split_cfg['train_start']}–{split_cfg['train_end']} "
            f"({len(tr)} rows), "
            f"Test range: {split_cfg['test_start']}–{split_cfg['test_end']} "
            f"({len(te)} rows). "
            "Check split ranges in base.yaml."
        )

    return tr, te


def _load_scaler(path: str):
    if not os.path.exists(path):
        _fail(f"Scaler NPZ not found: {path}")
    try:
        obj = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to load scaler NPZ '{path}': {exc}")

    if "scaler" not in obj:
        _fail(
            f"'scaler' key not found in '{path}'. "
            "Ensure step 02 saved np.savez('artifacts/scaler.npz', scaler=scaler)."
        )
    return obj["scaler"][()]


def _load_encoder(
    path: str,
    in_dim: int,
    enc_dim: int,
    steps: int,
    block: int,
) -> TinyDCeNN:
    if not os.path.exists(path):
        _fail(f"Encoder state_dict not found: {path}")
    enc = TinyDCeNN(in_dim, enc_dim=enc_dim, steps=steps, block=block)
    try:
        state = torch.load(path, map_location="cpu")
        enc.load_state_dict(state)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to load encoder state_dict from '{path}': {exc}")
    enc.eval()
    return enc


def _load_elm_heads(path: str) -> Tuple[np.ndarray, List[str]]:
    if not os.path.exists(path):
        _fail(f"ELM heads NPZ not found: {path}")
    try:
        obj = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to load ELM heads NPZ '{path}': {exc}")

    if "W" not in obj or "target_names" not in obj:
        _fail(
            f"'W' or 'target_names' not found in '{path}'. "
            "Ensure step 03 saved np.savez_compressed('elm_heads.npz', W=W, target_names=...)."
        )
    W = obj["W"]
    target_names = [str(t) for t in obj["target_names"]]
    return W, target_names


def _load_thresholds(path: str) -> Dict:
    if not os.path.exists(path):
        _fail(f"Thresholds JSON not found: {path}")
    try:
        with open(path, "r") as f:
            thr = json.load(f)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read thresholds JSON '{path}': {exc}")

    if "targets" not in thr:
        _fail("Thresholds JSON must contain a 'targets' key.")
    return thr


def _get_thr(thr_targets: Dict, tname: str, bucket_value) -> float:
    """
    Retrieve threshold for a target + bucket.
    If 'default' exists, use it.
    Else use the bucket id if present, or fall back to first value.
    """
    if tname not in thr_targets:
        _fail(f"Target '{tname}' not found in thresholds JSON.")

    d = thr_targets[tname]
    if "default" in d:
        return float(d["default"])

    b_str = str(int(bucket_value))
    if b_str in d:
        return float(d[b_str])

    # Fallback: first threshold value (and warn)
    _warn(
        f"Bucket '{b_str}' not found for target '{tname}' in thresholds; "
        "falling back to first available threshold."
    )
    # values() ordering is deterministic in Python 3.7+
    return float(next(iter(d.values())))


# ---------------------------- main ----------------------------


def main() -> None:
    args = _parse_args()

    base_cfg = _load_yaml(args.base_config)
    feat_cfg = _load_yaml(args.features_config)
    thr_cfg = _load_yaml(args.thresholds_config)
    dcenn_cfg = _load_yaml(args.dcenn_config)

    engineered_path = base_cfg.get("engineered_csv")
    split_cfg = base_cfg.get("split")
    if not isinstance(engineered_path, str):
        _fail("base['engineered_csv'] must be present and a string in base.yaml.")
    if not isinstance(split_cfg, dict):
        _fail("base['split'] must be present and a mapping in base.yaml.")

    # Features config
    try:
        lags = list(feat_cfg["lags"])
        rolls = list(feat_cfg["rolls"])
        wins_cols = list(feat_cfg["winsorize_cols"])
    except KeyError as exc:
        _fail(f"Missing key in features config: {exc}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"Invalid features config structure: {exc}")

    # Threshold config
    bucket_by = thr_cfg.get("bucket_by", "none")  # none | hour | holiday
    if bucket_by not in ("none", "hour", "holiday"):
        _fail(
            f"Unknown bucket_by '{bucket_by}'. Use 'none', 'hour', or 'holiday'."
        )

    # dCeNN config
    enc_dim = int(dcenn_cfg.get("enc_dim", 48))
    steps = int(dcenn_cfg.get("steps", 2))
    block = int(dcenn_cfg.get("block", 8))

    print(f"[INFO] Engineered CSV: {engineered_path}")
    print(f"[INFO] bucket_by: {bucket_by}")
    print(f"[INFO] dCeNN config: enc_dim={enc_dim}, steps={steps}, block={block}")
    print(f"[INFO] Lags: {lags}, Rolls: {rolls}")
    print(f"[INFO] Winsorize columns: {wins_cols}")

    # Load engineered data and winsorize (consistent with steps 02 & 04)
    df = _load_engineered(engineered_path)
    winsorize_inplace(df, wins_cols)

    # Train + test split
    tr, te = _split_train_test(df, split_cfg)

    # Rebuild supervised sets
    t_tr, Xtr, Ytr = make_supervised(tr, lags, rolls)
    t_te, Xte, Yte = make_supervised(te, lags, rolls)

    print(
        f"[INFO] Supervised shapes: "
        f"X_train={Xtr.shape}, Y_train={Ytr.shape}, "
        f"X_test={Xte.shape},  Y_test={Yte.shape}"
    )

    # Load scaler (DO NOT REFIT)
    scaler = _load_scaler(args.scaler_path)
    Xtr_s = scaler.transform(Xtr.values)
    Xte_s = scaler.transform(Xte.values)

    in_dim = Xtr_s.shape[1]
    print(f"[INFO] Scaled feature dimension: in_dim={in_dim}")

    # Load encoder + ELM heads
    encoder = _load_encoder(
        args.encoder_path,
        in_dim=in_dim,
        enc_dim=enc_dim,
        steps=steps,
        block=block,
    )

    with torch.no_grad():
        Hte = encoder(torch.tensor(Xte_s, dtype=torch.float32)).numpy()
    latent_dim = Hte.shape[1]
    print(f"[INFO] Encoded test latent shape: {Hte.shape}")

    W, target_names = _load_elm_heads(args.elm_heads_path)
    if W.shape[0] != latent_dim:
        _fail(
            f"ELM weight matrix W has shape {W.shape} but latent_dim={latent_dim}. "
            "Check that encoder/ELM heads come from the same training run."
        )

    # Predictions & residuals
    Yte_pred = Hte @ W
    resid = Yte.values - Yte_pred
    abs_resid = np.abs(resid)

    # Build test_df with true/pred/resid/abs per target
    test_df = pd.DataFrame(index=t_te)
    for i, name in enumerate(target_names):
        test_df[f"{name}_true"] = Yte.values[:, i]
        test_df[f"{name}_pred"] = Yte_pred[:, i]
        test_df[f"{name}_resid"] = resid[:, i]
        test_df[f"{name}_abs"] = abs_resid[:, i]

    # Bucket assignment
    if bucket_by == "hour":
        test_df["bucket"] = test_df.index.hour
    elif bucket_by == "holiday":
        if "is_public_holiday" not in df.columns:
            _fail(
                "bucket_by='holiday' but 'is_public_holiday' is missing in engineered "
                "DataFrame. Ensure holidays & flags were added in step 01."
            )
        test_df["bucket"] = df.loc[test_df.index, "is_public_holiday"].astype(int)
    else:
        test_df["bucket"] = 0

    # Load thresholds JSON
    thr = _load_thresholds(args.thresholds_json)
    thr_targets = thr["targets"]

    # Apply thresholds per target
    for name in target_names:
        thr_arr = np.array(
            [_get_thr(thr_targets, name, b) for b in test_df["bucket"]],
            dtype=float,
        )
        test_df[f"{name}_thr"] = thr_arr
        # Flag anomalies where |resid| > thr (ignore NaN thr => flag=0)
        mask_valid = ~np.isnan(thr_arr)
        anom = np.zeros_like(thr_arr, dtype=int)
        anom[mask_valid] = (
            test_df.loc[mask_valid, f"{name}_abs"].values > thr_arr[mask_valid]
        ).astype(int)
        test_df[f"{name}_anom"] = anom

    # Combined anomaly score (same as original logic)
    score = np.zeros(len(test_df), dtype=float)
    for name in target_names:
        thr_col = test_df[f"{name}_thr"].values
        abs_col = test_df[f"{name}_abs"].values
        # Avoid division by zero; treat zero/NaN threshold as no contribution
        valid = thr_col > 0
        contrib = np.zeros_like(score)
        contrib[valid] = np.maximum(
            0.0,
            abs_col[valid] / (thr_col[valid] + 1e-9) - 1.0,
        )
        score += contrib

    test_df["combined_score"] = score
    # Alias: anomaly_score (for downstream scripts / readability)
    test_df["anomaly_score"] = score

    # Global flags
    anom_cols = [f"{name}_anom" for name in target_names]
    test_df["n_targets_anom"] = test_df[anom_cols].sum(axis=1)
    test_df["any_anomaly_flag"] = (test_df["n_targets_anom"] > 0).astype(int)

    # Save anomaly table
    os.makedirs("reports/tables", exist_ok=True)
    anomalies_csv = "reports/tables/anomalies_2022.csv"
    test_df.to_csv(anomalies_csv)
    print(f"[INFO] Saved anomaly CSV to '{anomalies_csv}'")

    # Plot combined anomaly score
    os.makedirs("reports/figures", exist_ok=True)
    fig_path = "reports/figures/anomaly_combined_2022.png"

    plt.figure(figsize=(12, 4))
    plt.plot(test_df.index, test_df["combined_score"])
    plt.ylabel("score")
    plt.title("Combined Anomaly Score (2022)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()

    print(f"[INFO] Saved combined anomaly plot to '{fig_path}'")
    print("[INFO] Anomaly detection (05_detect_anomalies.py) completed.")


if __name__ == "__main__":
    main()
