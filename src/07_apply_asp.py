# src/07_apply_asp.py

#!/usr/bin/env python3
"""
Apply ASP rules (rules/07_asp_rules.lp) to refine anomalies.

Inputs:
  - configs/base.yaml (engineered_csv, holidays_csv)
  - reports/tables/anomalies_2022.csv (or latest anomalies_*.csv)
  - rules/07_asp_rules.lp

Facts generated per hour T (hour_index = floor(UnixTime / 3600)):
  - hour(T).
  - pred(solar_elevation, E, T).      # synthetic: >0 day, <0 night (from radiation)
  - pred(cf_solar,        V, T).
  - pred(cf_wind,         C, T).
  - pred(load_mw,         L, T).
  - pred(load_ref,        R, T).      # climatological baseline load
  - pred(dload_pred,      D, T).
  - pred(wind_speed_100m, Vw, T).

  - anom(Name, T).                    # from *_anom columns in anomalies CSV
  - bucket(holiday, H, T).            # H in {0,1} from engineered is_public_holiday

Output:
  - reports/tables/anomalies_refined.csv with columns:
      target         (e.g. "CF_Solar")
      asp_name       (e.g. "cf_solar")
      hour_index
      Time (UTC)
      anomaly_score  (from anomalies CSV: anomaly_score or combined_score)
      final_flag     (=1 for all rows)
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import clingo
import numpy as np
import pandas as pd
import yaml


DEFAULT_BASE_CONFIG = "configs/base.yaml"
DEFAULT_RULES_PATH = "rules/07_asp_rules.lp"
DEFAULT_ANOM_CSV = "reports/tables/anomalies_2022.csv"
DEFAULT_REFINED_CSV = "reports/tables/anomalies_refined.csv"


# ------------------------- helpers -------------------------


def _fail(msg: str) -> None:
    import sys
    sys.stderr.write(f"[ERROR] {msg}\n")
    sys.exit(1)


def _warn(msg: str) -> None:
    import sys
    sys.stderr.write(f"[WARN] {msg}\n")


def _load_yaml(path: str) -> Dict:
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
        description="Apply ASP rules to refine anomalies (final(Name,T))."
    )
    parser.add_argument(
        "--base-config",
        default=DEFAULT_BASE_CONFIG,
        help=f"Path to base config YAML (default: {DEFAULT_BASE_CONFIG}).",
    )
    parser.add_argument(
        "--anoms-csv",
        default=DEFAULT_ANOM_CSV,
        help=(
            "Path to anomalies CSV. If this path does not exist, "
            "will fall back to latest 'reports/tables/anomalies_*.csv'. "
            f"(default: {DEFAULT_ANOM_CSV})"
        ),
    )
    parser.add_argument(
        "--rules-path",
        default=DEFAULT_RULES_PATH,
        help=f"Path to ASP rules file (default: {DEFAULT_RULES_PATH}).",
    )
    parser.add_argument(
        "--out-csv",
        default=DEFAULT_REFINED_CSV,
        help=f"Path to refined anomalies CSV (default: {DEFAULT_REFINED_CSV}).",
    )
    return parser.parse_args()


def _ensure_parent_dir(path: str) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)


def to_hour_index(ts: pd.Timestamp) -> int:
    """Map a UTC timestamp → integer hour index since Unix epoch."""
    ts = pd.to_datetime(ts, utc=True)
    return int(ts.timestamp() // 3600)


def _detect_anoms_csv(path: str) -> str:
    """Use provided anomalies CSV if it exists; else fall back to latest anomalies_*.csv."""
    if os.path.exists(path):
        return path

    _warn(
        f"Anomalies CSV '{path}' not found. Looking for "
        "'reports/tables/anomalies_*.csv' instead."
    )
    cands = sorted(glob.glob("reports/tables/anomalies_*.csv"))
    if not cands:
        _fail(
            "No anomalies CSV found. Run:\n"
            "  make train\n"
            "  make thresholds\n"
            "  make detect\n"
        )
    chosen = cands[-1]
    _warn(f"Using latest anomalies CSV: {chosen}")
    return chosen


def _load_engineered(path: str) -> pd.DataFrame:
    """Load engineered CSV and set UTC index."""
    if not os.path.exists(path):
        _fail(f"Engineered CSV not found: {path}")
    try:
        df = pd.read_csv(path, parse_dates=["Time (UTC)"])
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read engineered CSV '{path}': {exc}")

    if "Time (UTC)" not in df.columns:
        _fail(
            f"Engineered CSV '{path}' must contain 'Time (UTC)' column. "
            f"Columns={list(df.columns)}"
        )

    df["Time (UTC)"] = pd.to_datetime(df["Time (UTC)"], utc=True)
    df = df.set_index("Time (UTC)").sort_index()
    if df.empty:
        _fail(f"Engineered DataFrame from '{path}' is empty.")
    return df


def _load_anomalies(anoms_csv: str) -> pd.DataFrame:
    """Load anomalies CSV and ensure 'Time (UTC)' index."""
    try:
        df = pd.read_csv(anoms_csv, parse_dates=["Time (UTC)"])
    except ValueError as exc:
        _fail(
            f"Failed to read anomalies CSV '{anoms_csv}' with 'Time (UTC)' "
            f"as parse_dates: {exc}."
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read anomalies CSV '{anoms_csv}': {exc}")

    if "Time (UTC)" not in df.columns:
        _fail(
            f"Anomalies CSV '{anoms_csv}' must contain 'Time (UTC)' column. "
            f"Columns={list(df.columns)}"
        )

    df["Time (UTC)"] = pd.to_datetime(df["Time (UTC)"], utc=True)
    df = df.set_index("Time (UTC)").sort_index()
    if df.empty:
        _fail(f"Anomalies DataFrame from '{anoms_csv}' is empty.")
    return df


def _extract_targets(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Extract target names from '*_anom' columns.

    Returns:
      targets_disp:  display names, e.g. ["CF_Solar", "CF_Wind", "Load_MW", "Price"]
      asp_from_disp: mapping display -> asp name, e.g. "CF_Solar" -> "cf_solar"
      disp_from_asp: reverse mapping asp -> display
    """
    anom_cols = [c for c in df.columns if c.endswith("_anom")]
    if not anom_cols:
        _fail(
            "No '*_anom' columns found in anomalies CSV. Expected per-target "
            "anomaly flags like 'CF_Solar_anom', 'CF_Wind_anom', etc."
        )

    targets_disp = sorted({c[:-5] for c in anom_cols})  # strip '_anom'
    asp_from_disp = {d: d.lower() for d in targets_disp}
    disp_from_asp = {asp: disp for disp, asp in asp_from_disp.items()}
    return targets_disp, asp_from_disp, disp_from_asp


def _build_load_reference(df_engineered: pd.DataFrame) -> pd.Series:
    """
    Build a simple climatological load reference:
      load_ref(t) = median Actual_Load_MW for that hour-of-week across all years.

    Returns:
      ref_series indexed like df_engineered.index
    """
    if "Actual_Load_MW" not in df_engineered.columns:
        _fail(
            "Engineered data must contain 'Actual_Load_MW' to build load_ref. "
            f"Columns={list(df_engineered.columns)}"
        )

    idx = df_engineered.index
    how = idx.dayofweek * 24 + idx.hour  # hour-of-week 0..167
    clim = (
        pd.DataFrame({"how": how, "load": df_engineered["Actual_Load_MW"]})
        .groupby("how")["load"]
        .median()
    )

    # Map back
    ref_vals = clim.reindex(how).to_numpy()
    # Fallback: if any NaNs (shouldn't happen), use actual load
    mask_nan = np.isnan(ref_vals)
    if mask_nan.any():
        ref_vals[mask_nan] = df_engineered["Actual_Load_MW"].to_numpy()[mask_nan]

    ref_series = pd.Series(ref_vals, index=idx, name="load_ref")
    return ref_series


def _run_clingo(facts_program: str, rules_path: str) -> str:
    """Run Clingo on given facts + rules and return text of all models."""
    if not os.path.exists(rules_path):
        _fail(f"ASP rules file not found: {rules_path}")

    print("[INFO] Running Clingo reasoning via Python API ...")

    import tempfile

    with tempfile.NamedTemporaryFile("w", delete=False) as tmp_f:
        tmp_f.write(facts_program)
        facts_path = tmp_f.name

    ctl = clingo.Control()
    ctl.load(facts_path)
    ctl.load(rules_path)
    ctl.ground([("base", [])])

    models: List[str] = []

    def on_model(m: clingo.Model) -> None:
        models.append(str(m))

    ctl.solve(on_model=on_model)
    os.remove(facts_path)

    return "\n".join(models)


def _parse_final(clingo_output: str) -> List[Tuple[str, int]]:
    """
    Parse final(Name, T) atoms from Clingo output.

    Returns:
      List of (asp_name, hour_index)
    """
    pa

