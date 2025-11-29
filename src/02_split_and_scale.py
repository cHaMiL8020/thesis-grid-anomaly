# src/02_split_and_scale.py

#!/usr/bin/env python3
"""
Split engineered dataset into train/val/test, build supervised matrices,
scale features, and save to NPZ for downstream training.

Inputs:
- configs/base.yaml      (paths, split ranges, horizon)
- configs/features.yaml  (lags, rolls, winsorize columns)
- ENGINEERED CSV from step 01 (cfg["engineered_csv"])

Outputs:
- NPZ file at base["npz_path"] with:
    X_train, Y_train, X_val, Y_val, X_test, Y_test,
    feature_names, target_names
- artifacts/scaler.npz with a fitted sklearn StandardScaler

Defaults are fully compatible with the original script; only error handling
and robustness are improved.
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


DEFAULT_BASE_CONFIG = "configs/base.yaml"
DEFAULT_FEATURES_CONFIG = "configs/features.yaml"


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split engineered data, build supervised features, and scale."
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
    return parser.parse_args()


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
    """Ensure the parent directory for a file path exists."""
    out_dir = os.path.dirname(path) or "."
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to create directory '{out_dir}': {exc}")


# --------------------------------------------------------
# Feature list (authoritative for pipeline)
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


# ------------------------- core logic -------------------------


def winsorize_inplace(df: pd.DataFrame, cols: List[str], lo: float = 0.01, hi: float = 0.99) -> None:
    """
    Winsorize columns in-place to [lo, hi] quantiles.

    Missing columns are skipped with a warning instead of raising.
    """
    for c in cols:
        if c not in df.columns:
            _warn(
                f"winsorize_inplace: column '{c}' not found in DataFrame; "
                "skipping."
            )
            continue
        series = df[c].dropna()
        if series.empty:
            _warn(
                f"winsorize_inplace: column '{c}' is empty/NaN-only; "
                "skipping."
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
    Build supervised dataset:
      - X: base engineered features + lags + rolling means
      - Y: same-hour targets (NO future shift)

    Returns:
      index: timestamps after dropping NaNs
      X: feature matrix DataFrame
      Y: target matrix DataFrame aligned with X
    """
    # Ensure base features are present
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

    # Same-hour targets (no shift)
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

    # X must keep only its own columns; Y aligned to XY index
    return XY.index, XY[X.columns], Y.loc[XY.index]


def _load_engineered(path: str) -> pd.DataFrame:
    """Load engineered CSV and ensure it has 'Time (UTC)' index."""
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
            "Keeping the first occurrence per timestamp."
        )
        df = df[~df.index.duplicated(keep="first")]

    return df


def _split_by_time(df: pd.DataFrame, split_cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/val/test using string ranges from base['split']:

    Expected keys in split_cfg:
      - train_start, train_end
      - val_start,   val_end
      - test_start,  test_end
    """
    required_keys = [
        "train_start", "train_end",
        "val_start", "val_end",
        "test_start", "test_end",
    ]
    missing = [k for k in required_keys if k not in split_cfg]
    if missing:
        _fail(
            f"Missing split keys in base['split']: {missing}. "
            "Expected keys: " + ", ".join(required_keys)
        )

    tr = df.loc[split_cfg["train_start"]:split_cfg["train_end"]]
    va = df.loc[split_cfg["val_start"]:split_cfg["val_end"]]
    te = df.loc[split_cfg["test_start"]:split_cfg["test_end"]]

    if tr.empty or va.empty or te.empty:
        _fail(
            "One or more of the splits (train/val/test) is empty. "
            f"Train range: {split_cfg['train_start']}–{split_cfg['train_end']} "
            f"({len(tr)} rows), "
            f"Val range: {split_cfg['val_start']}–{split_cfg['val_end']} "
            f"({len(va)} rows), "
            f"Test range: {split_cfg['test_start']}–{split_cfg['test_end']} "
            f"({len(te)} rows). "
            "Check split ranges in base.yaml."
        )

    return tr, va, te


# ---------------------------- main ----------------------------


def main() -> None:
    args = _parse_args()

    base_cfg = _load_yaml(args.base_config)
    feat_cfg = _load_yaml(args.features_config)

    # Resolve essential base config fields
    engineered_path = base_cfg.get("engineered_csv")
    npz_path = base_cfg.get("npz_path")
    split_cfg = base_cfg.get("split")
    horizon = int(base_cfg.get("horizon", 1))  # currently unused (same-hour targets)

    if not isinstance(engineered_path, str):
        _fail("base['engineered_csv'] must be present and a string in base.yaml.")
    if not isinstance(npz_path, str):
        _fail("base['npz_path'] must be present and a string in base.yaml.")
    if not isinstance(split_cfg, dict):
        _fail("base['split'] must be present and a mapping in base.yaml.")

    # Resolve features config
    try:
        lags = list(feat_cfg["lags"])
        rolls = list(feat_cfg["rolls"])
        wins_cols = list(feat_cfg["winsorize_cols"])
    except KeyError as exc:
        _fail(f"Missing key in features config (configs/features.yaml): {exc}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"Invalid features config structure: {exc}")

    print(f"[INFO] Using engineered CSV: {engineered_path}")
    print(f"[INFO] Target NPZ path: {npz_path}")
    print(f"[INFO] Horizon (currently unused): {horizon}")
    print(f"[INFO] Lags: {lags}")
    print(f"[INFO] Rolls: {rolls}")
    print(f"[INFO] Winsorize columns: {wins_cols}")

    # Load engineered data
    df = _load_engineered(engineered_path)

    # Apply winsorization
    winsorize_inplace(df, wins_cols)

    # Split into train/val/test
    tr, va, te = _split_by_time(df, split_cfg)

    print(
        f"[INFO] Split sizes: train={len(tr)}, val={len(va)}, test={len(te)} "
        f"(total={len(df)})"
    )

    # Make supervised datasets
    t_tr, Xtr, Ytr = make_supervised(tr, lags, rolls)
    t_va, Xva, Yva = make_supervised(va, lags, rolls)
    t_te, Xte, Yte = make_supervised(te, lags, rolls)

    print(
        f"[INFO] Supervised shapes: "
        f"X_train={Xtr.shape}, Y_train={Ytr.shape}, "
        f"X_val={Xva.shape},   Y_val={Yva.shape}, "
        f"X_test={Xte.shape},  Y_test={Yte.shape}"
    )

    # Scale features (train-only fit)
    scaler = StandardScaler().fit(Xtr.values)

    Xtr_s = scaler.transform(Xtr.values)
    Xva_s = scaler.transform(Xva.values)
    Xte_s = scaler.transform(Xte.values)

    # Save scaler
    os.makedirs("artifacts", exist_ok=True)
    try:
        # This will store scaler as an object array; load with allow_pickle=True.
        np.savez("artifacts/scaler.npz", scaler=scaler)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to save scaler to 'artifacts/scaler.npz': {exc}")

    # Save NPZ dataset
    _ensure_parent_dir(npz_path)
    try:
        np.savez_compressed(
            npz_path,
            X_train=Xtr_s,
            Y_train=Ytr.values,
            X_val=Xva_s,
            Y_val=Yva.values,
            X_test=Xte_s,
            Y_test=Yte.values,
            feature_names=np.array(Xtr.columns),
            target_names=np.array(Ytr.columns),
            train_timestamps=t_tr.values,
            val_timestamps=t_va.values,
            test_timestamps=t_te.values,
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to save NPZ dataset to '{npz_path}': {exc}")

    print(f"[INFO] Saved dataset NPZ to '{npz_path}'")
    print("[INFO] Saved scaler to 'artifacts/scaler.npz'")


if __name__ == "__main__":
    main()
