#!/usr/bin/env python3
"""
Generate additional thesis report tables (CSV + Markdown) from existing artifacts.

Inputs:
  - reports/tables/anomalies_2022.csv
  - reports/tables/anomalies_refined.csv
  - reports/tables/benchmark_comparison.csv

Outputs (new files only):
  - reports/tables/veto_reason_summary.csv
  - reports/tables/veto_reason_summary.md
  - reports/tables/seasonal_performance.csv
  - reports/tables/seasonal_performance.md
  - reports/tables/edge_efficiency_snapshot.csv
  - reports/tables/edge_efficiency_snapshot.md
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_ANOM_RAW = "reports/tables/anomalies_2022.csv"
DEFAULT_ANOM_REF = "reports/tables/anomalies_refined.csv"
DEFAULT_BENCH = "reports/tables/benchmark_comparison.csv"
DEFAULT_OUT_DIR = "reports/tables"

TARGET_COLS = {
    "cf_solar": "CF_Solar_anom",
    "cf_wind": "CF_Wind_anom",
    "load_mw": "Load_MW_anom",
    "price": "Price_anom",
}

TARGET_DISPLAY = {
    "cf_solar": "Solar",
    "cf_wind": "Wind",
    "load_mw": "Load",
    "price": "Price",
}

SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]


def _fail(msg: str) -> None:
    import sys

    sys.stderr.write(f"[ERROR] {msg}\n")
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate additional thesis report tables.")
    parser.add_argument("--anoms-raw", default=DEFAULT_ANOM_RAW)
    parser.add_argument("--anoms-refined", default=DEFAULT_ANOM_REF)
    parser.add_argument("--benchmark-csv", default=DEFAULT_BENCH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _load_csv(path: str, parse_dates: List[str] | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        _fail(f"Input file not found: {path}")
    try:
        return pd.read_csv(path, parse_dates=parse_dates or [], skipinitialspace=True)
    except Exception as exc:
        _fail(f"Failed to read '{path}': {exc}")


def _normalize_target(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().str.replace(" ", "", regex=False)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) else float("nan")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _to_markdown_table(df: pd.DataFrame, title: str) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    lines = [f"# {title}", "", header, sep]

    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.6f}".rstrip("0").rstrip("."))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    return "\n".join(lines)


def _build_veto_reason_summary(df_raw: pd.DataFrame, df_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Build a top-5 ASP veto reason summary using artifact-only proxy rules.

    Because the refined table does not include explicit veto_reason atoms, this function
    computes transparent, reproducible proxy categories from timestamp/target context.
    """
    if "Time (UTC)" not in df_raw.columns or "Time (UTC)" not in df_ref.columns:
        _fail("Both anomalies tables must include 'Time (UTC)'.")

    veto = df_ref.copy()
    veto["target"] = _normalize_target(veto["target"])
    veto = veto[(veto["is_vetoed"].astype(int) == 1) & (veto["target"].isin(TARGET_COLS.keys()))].copy()

    if veto.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "reason",
                "veto_count",
                "share_of_vetoed_false_positives_percent",
            ]
        )

    raw = df_raw.copy().set_index("Time (UTC)").sort_index()
    veto = veto.sort_values("Time (UTC)")

    reason_counts: Dict[str, int] = {
        "PV-at-night": 0,
        "Wind-ramp-guard (proxy)": 0,
        "Holiday/weekend-exemption (proxy)": 0,
        "Missing co-anomaly confirmation": 0,
        "Transient non-persistent spike": 0,
    }

    for row in veto.itertuples(index=False):
        ts = getattr(row, "_1") if False else row[1]
        target = getattr(row, "target")

        if ts not in raw.index:
            reason_counts["Transient non-persistent spike"] += 1
            continue

        rec = raw.loc[ts]
        hour = int(ts.hour)

        assigned = False

        if target == "cf_solar":
            if hour <= 5 or hour >= 20:
                reason_counts["PV-at-night"] += 1
                assigned = True
            else:
                reason_counts["Transient non-persistent spike"] += 1
                assigned = True

        elif target == "cf_wind":
            # Proxy for wind-ramp plausibility using absolute change in CF_Wind_true.
            if "CF_Wind_true" in raw.columns:
                prev_ts = ts - pd.Timedelta(hours=1)
                if prev_ts in raw.index:
                    ramp = abs(float(raw.loc[ts, "CF_Wind_true"]) - float(raw.loc[prev_ts, "CF_Wind_true"]))
                    if ramp > float(raw["CF_Wind_true"].diff().abs().quantile(0.90)):
                        reason_counts["Wind-ramp-guard (proxy)"] += 1
                        assigned = True
            if not assigned:
                reason_counts["Transient non-persistent spike"] += 1
                assigned = True

        elif target == "load_mw":
            # Proxy for holiday-type exception using weekend calendar context.
            if ts.dayofweek >= 5:
                reason_counts["Holiday/weekend-exemption (proxy)"] += 1
                assigned = True
            else:
                reason_counts["Transient non-persistent spike"] += 1
                assigned = True

        elif target == "price":
            # Proxy for missing price-load co-anomaly logic.
            load_flag = int(rec.get("Load_MW_anom", 0)) if isinstance(rec, pd.Series) else 0
            if load_flag == 0:
                reason_counts["Missing co-anomaly confirmation"] += 1
                assigned = True
            else:
                reason_counts["Transient non-persistent spike"] += 1
                assigned = True

        if not assigned:
            reason_counts["Transient non-persistent spike"] += 1

    total_veto = int(len(veto))
    rows = []
    for reason, cnt in reason_counts.items():
        if cnt <= 0:
            continue
        rows.append(
            {
                "reason": reason,
                "veto_count": int(cnt),
                "share_of_vetoed_false_positives_percent": round(100.0 * cnt / total_veto, 2),
            }
        )

    out = pd.DataFrame(rows).sort_values("veto_count", ascending=False).head(5).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def _build_seasonal_performance(df_raw: pd.DataFrame, df_ref: pd.DataFrame) -> pd.DataFrame:
    if "Time (UTC)" not in df_raw.columns:
        _fail("Raw anomalies table must include 'Time (UTC)'.")

    work = df_raw.copy().sort_values("Time (UTC)")
    work["month"] = work["Time (UTC)"].dt.month
    work["season"] = work["month"].map(SEASON_BY_MONTH)

    ref = df_ref.copy()
    ref["target"] = _normalize_target(ref["target"])
    ref = ref[ref["target"].isin(TARGET_COLS.keys())].copy()

    # Build fast lookup of confirmed anomaly timestamps by target.
    confirmed_map: Dict[str, set] = {}
    for t in TARGET_COLS.keys():
        confirmed_ts = set(
            ref[(ref["target"] == t) & (ref["final_flag"].astype(int) == 1)]["Time (UTC)"]
            .dropna()
            .tolist()
        )
        confirmed_map[t] = confirmed_ts

    rows: List[Dict] = []

    for target, anom_col in TARGET_COLS.items():
        pred_col = {
            "cf_solar": "CF_Solar_pred",
            "cf_wind": "CF_Wind_pred",
            "load_mw": "Load_MW_pred",
            "price": "Price_pred",
        }[target]
        true_col = {
            "cf_solar": "CF_Solar_true",
            "cf_wind": "CF_Wind_true",
            "load_mw": "Load_MW_true",
            "price": "Price_true",
        }[target]

        if pred_col not in work.columns or true_col not in work.columns or anom_col not in work.columns:
            continue

        for season in SEASON_ORDER:
            g = work[work["season"] == season]
            if g.empty:
                continue

            y_true_vals = g[true_col].astype(float).to_numpy()
            y_pred_vals = g[pred_col].astype(float).to_numpy()
            rmse_val = _rmse(y_true_vals, y_pred_vals)

            # F1 proxy definition:
            # - Predicted positives: neural anomaly flags from raw table.
            # - Proxy positives: ASP-confirmed anomalies from refined table.
            # This measures seasonal agreement between neural and symbolic layers.
            pred_mask = g[anom_col].fillna(0).astype(int).clip(0, 1).to_numpy()
            time_vals = g["Time (UTC)"].to_list()
            true_proxy = np.array([1 if ts in confirmed_map[target] else 0 for ts in time_vals], dtype=int)

            tp = int(np.sum((pred_mask == 1) & (true_proxy == 1)))
            fp = int(np.sum((pred_mask == 1) & (true_proxy == 0)))
            fn = int(np.sum((pred_mask == 0) & (true_proxy == 1)))

            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
                f1_proxy = float("nan")
            else:
                f1_proxy = float(2.0 * precision * recall / (precision + recall))

            rows.append(
                {
                    "season": season,
                    "target": TARGET_DISPLAY[target],
                    "n_points": int(len(g)),
                    "rmse": float(rmse_val),
                    "f1_proxy": float(f1_proxy),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["season"] = pd.Categorical(out["season"], categories=SEASON_ORDER, ordered=True)
    out = out.sort_values(["season", "target"]).reset_index(drop=True)

    # Add macro row per season to help thesis discussion.
    macro = (
        out.groupby("season", as_index=False, observed=False)
        .agg(
            n_points=("n_points", "sum"),
            rmse=("rmse", "mean"),
            f1_proxy=("f1_proxy", "mean"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
        )
    )
    macro["target"] = "Macro Avg"

    final = pd.concat([out, macro], ignore_index=True)
    final["season"] = pd.Categorical(final["season"], categories=SEASON_ORDER, ordered=True)
    final = final.sort_values(["season", "target"]).reset_index(drop=True)
    return final


def _build_edge_efficiency_snapshot(df_bench: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df_bench.columns:
        _fail("Benchmark table must include 'model' column.")

    work = df_bench.copy()

    needed = ["model", "inf_latency_ms", "parameters", "rmse_solar", "rmse_wind", "rmse_load", "rmse_price"]
    missing = [c for c in needed if c not in work.columns]
    if missing:
        _fail(f"Benchmark table missing required columns: {missing}")

    keep = work[work["model"].isin(["dCeNN-ELM (Proposed)", "LSTM (Baseline-Sequential)"])].copy()
    if keep.empty:
        _fail("Could not find dCeNN-ELM and LSTM rows in benchmark comparison table.")

    keep["parameters"] = pd.to_numeric(keep["parameters"], errors="coerce")
    keep["inf_latency_ms"] = pd.to_numeric(keep["inf_latency_ms"], errors="coerce")

    keep["latency_ms_per_1k_params"] = keep["inf_latency_ms"] / (keep["parameters"] / 1000.0)
    keep["avg_rmse"] = keep[["rmse_solar", "rmse_wind", "rmse_load", "rmse_price"]].astype(float).mean(axis=1)

    keep = keep[
        [
            "model",
            "parameters",
            "inf_latency_ms",
            "latency_ms_per_1k_params",
            "avg_rmse",
            "rmse_solar",
            "rmse_wind",
            "rmse_load",
            "rmse_price",
        ]
    ].sort_values("model")

    d_row = keep[keep["model"] == "dCeNN-ELM (Proposed)"]
    l_row = keep[keep["model"] == "LSTM (Baseline-Sequential)"]

    if not d_row.empty and not l_row.empty:
        d_lat = float(d_row.iloc[0]["latency_ms_per_1k_params"])
        l_lat = float(l_row.iloc[0]["latency_ms_per_1k_params"])
        d_abs = float(d_row.iloc[0]["inf_latency_ms"])
        l_abs = float(l_row.iloc[0]["inf_latency_ms"])

        summary = pd.DataFrame(
            [
                {
                    "model": "Edge Readiness Gain (dCeNN vs LSTM)",
                    "parameters": np.nan,
                    "inf_latency_ms": np.nan,
                    "latency_ms_per_1k_params": _safe_div(l_lat, d_lat),
                    "avg_rmse": np.nan,
                    "rmse_solar": np.nan,
                    "rmse_wind": np.nan,
                    "rmse_load": np.nan,
                    "rmse_price": np.nan,
                    "note": f"Absolute latency speedup: {l_abs / d_abs:.2f}x (lower is better)",
                }
            ]
        )
        keep["note"] = ""
        keep = pd.concat([keep, summary], ignore_index=True)
    else:
        keep["note"] = ""

    return keep.reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_raw = _load_csv(args.anoms_raw, parse_dates=["Time (UTC)"])
    df_ref = _load_csv(args.anoms_refined, parse_dates=["Time (UTC)"])
    df_bench = _load_csv(args.benchmark_csv)

    veto = _build_veto_reason_summary(df_raw, df_ref)
    seasonal = _build_seasonal_performance(df_raw, df_ref)
    edge = _build_edge_efficiency_snapshot(df_bench)

    veto_csv = os.path.join(args.out_dir, "veto_reason_summary.csv")
    veto_md = os.path.join(args.out_dir, "veto_reason_summary.md")
    seasonal_csv = os.path.join(args.out_dir, "seasonal_performance.csv")
    seasonal_md = os.path.join(args.out_dir, "seasonal_performance.md")
    edge_csv = os.path.join(args.out_dir, "edge_efficiency_snapshot.csv")
    edge_md = os.path.join(args.out_dir, "edge_efficiency_snapshot.md")

    veto.to_csv(veto_csv, index=False)
    seasonal.to_csv(seasonal_csv, index=False)
    edge.to_csv(edge_csv, index=False)

    with open(veto_md, "w", encoding="utf-8") as f:
        f.write(_to_markdown_table(veto, "Veto Reason Summary"))
        f.write("\n")
        f.write("Notes: Reasons are artifact-only proxy categories inferred from target and timestamp context because explicit veto_reason atoms are not stored in anomalies_refined.csv.\n")

    with open(seasonal_md, "w", encoding="utf-8") as f:
        f.write(_to_markdown_table(seasonal, "Seasonal Performance"))
        f.write("\n")
        f.write("Notes: f1_proxy is the seasonal agreement score between neural anomaly flags and ASP-confirmed anomalies.\n")

    with open(edge_md, "w", encoding="utf-8") as f:
        f.write(_to_markdown_table(edge, "Edge Efficiency Snapshot"))
        f.write("\n")
        f.write("Notes: latency_ms_per_1k_params = inf_latency_ms / (parameters / 1000). Lower is better.\n")

    print(f"[INFO] Saved: {veto_csv}")
    print(f"[INFO] Saved: {veto_md}")
    print(f"[INFO] Saved: {seasonal_csv}")
    print(f"[INFO] Saved: {seasonal_md}")
    print(f"[INFO] Saved: {edge_csv}")
    print(f"[INFO] Saved: {edge_md}")


if __name__ == "__main__":
    main()
