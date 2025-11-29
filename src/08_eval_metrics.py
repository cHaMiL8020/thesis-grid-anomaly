# src/08_eval_metrics.py

#!/usr/bin/env python3
"""
Compute prediction & anomaly metrics from anomalies_2022.csv.

Default inputs/outputs:
  IN_CSV  = reports/tables/anomalies_2022.csv
  OUT_SUM = reports/tables/metrics_summary.csv
  OUT_MON = reports/tables/metrics_monthly.csv

Metrics:
  - Per-target global:
      * n_points
      * RMSE
      * MAE
      * MAPE (%)
      * R2
      * anomaly_count
  - Per-target monthly:
      * year, month
      * n_points
      * RMSE, MAE, MAPE, R2
      * anomaly_count
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

IN_CSV_DEFAULT = "reports/tables/anomalies_2022.csv"
OUT_SUM_DEFAULT = "reports/tables/metrics_summary.csv"
OUT_MON_DEFAULT = "reports/tables/metrics_monthly.csv"


# ------------------------- helpers -------------------------


def _fail(msg: str) -> None:
    import sys
    sys.stderr.write(f"[ERROR] {msg}\n")
    sys.exit(1)


def _warn(msg: str) -> None:
    import sys
    sys.stderr.write(f"[WARN] {msg}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute prediction & anomaly metrics from anomalies CSV."
    )
    parser.add_argument(
        "--in-csv",
        default=IN_CSV_DEFAULT,
        help=f"Path to anomalies CSV (default: {IN_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--out-summary",
        default=OUT_SUM_DEFAULT,
        help=f"Path to global metrics CSV (default: {OUT_SUM_DEFAULT}).",
    )
    parser.add_argument(
        "--out-monthly",
        default=OUT_MON_DEFAULT,
        help=f"Path to monthly metrics CSV (default: {OUT_MON_DEFAULT}).",
    )
    return parser.parse_args()


def _ensure_parent_dir(path: str) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)


def rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def mape(a, b) -> float:
    """
    Mean Absolute Percentage Error in %, ignoring zero true values.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = a != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100.0)


def r2_score(a, b) -> float:
    """
    Coefficient of determination R².
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return float("nan")
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _detect_targets(df: pd.DataFrame) -> List[str]:
    """
    Detect targets from '*_true' columns.
    E.g. CF_Solar_true -> 'CF_Solar'.
    """
    true_cols = [c for c in df.columns if c.endswith("_true")]
    if not true_cols:
        _fail(
            "No '*_true' columns found in anomalies CSV. "
            "Expected columns like 'CF_Solar_true', 'Load_MW_true', etc."
        )
    targets = sorted({c[:-5] for c in true_cols})  # strip '_true'
    return targets


def _extract_true_pred(
    df: pd.DataFrame, target: str
) -> Tuple[pd.Series, pd.Series, int]:
    """
    For a given target name (e.g. 'CF_Solar'), return (true, pred, n_points).
    Warn and return empty if missing.
    """
    t_true = f"{target}_true"
    t_pred = f"{target}_pred"

    if t_true not in df.columns or t_pred not in df.columns:
        _warn(
            f"Skipping target '{target}': missing '{t_true}' or '{t_pred}' "
            f"in anomalies CSV."
        )
        return pd.Series(dtype=float), pd.Series(dtype=float), 0

    true = df[t_true].astype(float)
    pred = df[t_pred].astype(float)
    # Drop rows where either is NaN
    mask = ~(true.isna() | pred.isna())
    true = true[mask]
    pred = pred[mask]
    return true, pred, len(true)


# ---------------------------- main ----------------------------


def main() -> None:
    args = _parse_args()

    in_csv = args.in_csv
    out_sum = args.out_summary
    out_mon = args.out_monthly

    if not os.path.exists(in_csv):
        _fail(
            f"Input CSV '{in_csv}' not found. "
            "Run `make detect` (Step 05) first to create anomalies CSV."
        )

    df = (
        pd.read_csv(in_csv, parse_dates=["Time (UTC)"])
        .set_index("Time (UTC)")
        .sort_index()
    )

    # Detect available targets automatically
    targets = _detect_targets(df)
    print(f"[INFO] Detected targets from '*_true' columns: {targets}")

    # ---------- Global metrics ----------
    rows_global: List[Dict] = []

    for t in targets:
        true, pred, n_points = _extract_true_pred(df, t)
        if n_points == 0:
            continue

        anom_col = f"{t}_anom"
        if anom_col in df.columns:
            anom_count = int(df[anom_col].fillna(0).astype(int).sum())
        else:
            _warn(
                f"Anomaly flag column '{anom_col}' missing; anomaly_count=0 for '{t}'."
            )
            anom_count = 0

        rows_global.append(
            {
                "target": t,
                "n_points": n_points,
                "RMSE": rmse(true, pred),
                "MAE": mae(true, pred),
                "MAPE_percent": mape(true, pred),
                "R2": r2_score(true, pred),
                "anomaly_count": anom_count,
            }
        )

    if not rows_global:
        _fail("No valid targets with true/pred data to evaluate.")

    summ = pd.DataFrame(rows_global)
    _ensure_parent_dir(out_sum)
    summ.to_csv(out_sum, index=False)

    # ---------- Monthly metrics ----------
    rows_monthly: List[Dict] = []

    # Guard: ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        _fail("DataFrame index must be a DatetimeIndex after parsing 'Time (UTC)'.")

    grouped = df.groupby([df.index.year, df.index.month])

    for (y, m), g in grouped:
        for t in targets:
            true, pred, n_points = _extract_true_pred(g, t)
            if n_points == 0:
                continue

            anom_col = f"{t}_anom"
            if anom_col in g.columns:
                anom_count = int(g[anom_col].fillna(0).astype(int).sum())
            else:
                anom_count = 0

            rows_monthly.append(
                {
                    "year": int(y),
                    "month": int(m),
                    "target": t,
                    "n_points": n_points,
                    "RMSE": rmse(true, pred),
                    "MAE": mae(true, pred),
                    "MAPE_percent": mape(true, pred),
                    "R2": r2_score(true, pred),
                    "anomaly_count": anom_count,
                }
            )

    monthly_df = pd.DataFrame(rows_monthly)
    _ensure_parent_dir(out_mon)
    monthly_df.to_csv(out_mon, index=False)

    print(f"[INFO] Saved global metrics to '{out_sum}'")
    print(f"[INFO] Saved monthly metrics to '{out_mon}'")


if __name__ == "__main__":
    main()
