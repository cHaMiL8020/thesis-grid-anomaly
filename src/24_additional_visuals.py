#!/usr/bin/env python3
"""
Create additional thesis visualizations from existing report tables.

Inputs:
  - reports/tables/anomalies_2022.csv
  - reports/tables/anomalies_refined.csv

Outputs (new files only):
  - reports/figures/additional_visuals/asp_impact_heatmap.png
  - reports/figures/additional_visuals/threshold_uncertainty_by_hour.png
  - reports/figures/additional_visuals/joint_anomaly_upset.png
"""

import argparse
import os
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"font.size": 10, "figure.dpi": 200})

DEFAULT_ANOM_RAW = "reports/tables/anomalies_2022.csv"
DEFAULT_ANOM_REF = "reports/tables/anomalies_refined.csv"
DEFAULT_OUT_DIR = "reports/figures/additional_visuals"

TARGET_DISPLAY = {
    "cf_solar": "Solar",
    "cf_wind": "Wind",
    "load_mw": "Load",
    "price": "Price",
}

RAW_TARGETS = {
    "Solar": "CF_Solar",
    "Wind": "CF_Wind",
    "Load": "Load_MW",
    "Price": "Price",
}

MONTH_ORDER = list(range(1, 13))
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fail(msg: str) -> None:
    import sys

    sys.stderr.write(f"[ERROR] {msg}\n")
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate additional thesis visuals.")
    parser.add_argument("--anoms-raw", default=DEFAULT_ANOM_RAW)
    parser.add_argument("--anoms-refined", default=DEFAULT_ANOM_REF)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_csv(path: str, parse_dates: Sequence[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        _fail(f"Input file not found: {path}")
    try:
        return pd.read_csv(path, parse_dates=list(parse_dates), skipinitialspace=True)
    except Exception as exc:
        _fail(f"Failed to read '{path}': {exc}")


def _normalize_target(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
    )


def plot_asp_impact_heatmap(df_ref: pd.DataFrame, out_path: str) -> None:
    """
    Two heatmaps: monthly confirmed and monthly vetoed anomaly counts per target.
    """
    work = df_ref.copy()
    work = work[work["target"].notna()].copy()
    work["target"] = _normalize_target(work["target"])
    work = work[work["target"].isin(TARGET_DISPLAY.keys())].copy()

    if work.empty:
        _fail("No target rows found in refined anomalies table for heatmap.")

    work["month"] = work["Time (UTC)"].dt.month

    def _pivot_for(flag_value: int) -> pd.DataFrame:
        g = (
            work[work["is_vetoed"].astype(int) == flag_value]
            .groupby(["target", "month"], as_index=False)
            .size()
        )
        p = (
            g.pivot(index="target", columns="month", values="size")
            .reindex(index=list(TARGET_DISPLAY.keys()), columns=MONTH_ORDER)
            .fillna(0)
            .astype(int)
        )
        p.index = [TARGET_DISPLAY[t] for t in p.index]
        return p

    confirmed = _pivot_for(0)
    vetoed = _pivot_for(1)

    vmax = max(int(confirmed.to_numpy().max()), int(vetoed.to_numpy().max()), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, mat, title in [
        (axes[0], confirmed, "ASP Confirmed Anomalies"),
        (axes[1], vetoed, "ASP Vetoed Anomalies"),
    ]:
        im = ax.imshow(mat.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(MONTH_LABELS)))
        ax.set_xticklabels(MONTH_LABELS, rotation=0)
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_xlabel("Month of 2022")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = int(mat.iat[i, j])
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color="black")

    axes[0].set_ylabel("Target")

    # Reserve room on the right so the shared colorbar never overlaps the
    # vetoed heatmap panel.
    fig.subplots_adjust(right=0.90, wspace=0.12)
    cax = fig.add_axes([0.915, 0.14, 0.02, 0.72])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Count")
    fig.suptitle("Monthly ASP Impact: Confirmed vs Vetoed Anomalies", y=1.02)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_band_distribution(df_raw: pd.DataFrame, out_path: str) -> None:
    """
    Show hourly distribution of calibrated threshold tau by target.
    """
    work = df_raw.copy()
    work["hour"] = work["Time (UTC)"].dt.hour

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()

    colors = {
        "Solar": "#f4a261",
        "Wind": "#2a9d8f",
        "Load": "#264653",
        "Price": "#e76f51",
    }

    for idx, (label, raw_name) in enumerate(RAW_TARGETS.items()):
        thr_col = f"{raw_name}_thr"
        ax = axes[idx]

        if thr_col not in work.columns:
            ax.text(0.5, 0.5, f"Missing column\n{thr_col}", ha="center", va="center")
            ax.set_title(label)
            continue

        agg = (
            work.groupby("hour")[thr_col]
            .agg(
                tau_q10=lambda s: np.nanpercentile(s.dropna(), 10) if s.dropna().size else np.nan,
                tau_q50=lambda s: np.nanpercentile(s.dropna(), 50) if s.dropna().size else np.nan,
                tau_q90=lambda s: np.nanpercentile(s.dropna(), 90) if s.dropna().size else np.nan,
            )
            .reindex(range(24))
        )

        x = np.arange(24)
        ax.fill_between(x, agg["tau_q10"], agg["tau_q90"], color=colors[label], alpha=0.25, label="10-90% band")
        ax.plot(x, agg["tau_q50"], color=colors[label], linewidth=1.8, label="Median $\\tau$")
        ax.set_title(label)
        ax.set_xlim(0, 23)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        units = {
            "Solar": "%",
            "Wind": "%",
            "Load": "MW",
            "Price": "EUR/MWh",
        }
        ax.set_ylabel(f"Threshold $\\tau$ ({units[label]})")

    for ax in axes[2:4]:
        ax.set_xlabel("Hour of day (UTC)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Hourly Threshold Uncertainty Bands (Calibrated $\\tau$)", y=1.03)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _build_combo_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "Solar": "CF_Solar_anom",
        "Wind": "CF_Wind_anom",
        "Load": "Load_MW_anom",
        "Price": "Price_anom",
    }
    missing = [c for c in cols.values() if c not in df_raw.columns]
    if missing:
        _fail(f"Missing anomaly-flag columns for joint analysis: {missing}")

    flags = pd.DataFrame({k: df_raw[v].fillna(0).astype(int).clip(0, 1) for k, v in cols.items()})

    combo_counts: Dict[Tuple[int, int, int, int], int] = {}
    for row in flags.itertuples(index=False):
        key = tuple(int(x) for x in row)
        if sum(key) == 0:
            continue
        combo_counts[key] = combo_counts.get(key, 0) + 1

    rows = []
    ordered_labels = ["Solar", "Wind", "Load", "Price"]
    for key, cnt in combo_counts.items():
        active = [ordered_labels[i] for i, bit in enumerate(key) if bit == 1]
        rows.append(
            {
                "signature": key,
                "active_targets": " + ".join(active),
                "k": len(active),
                "count": int(cnt),
            }
        )

    if not rows:
        _fail("No anomaly intersections found for joint analysis plot.")

    out = pd.DataFrame(rows).sort_values(["count", "k"], ascending=[False, False]).reset_index(drop=True)
    return out


def _pairwise_overlap_matrix(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "Solar": "CF_Solar_anom",
        "Wind": "CF_Wind_anom",
        "Load": "Load_MW_anom",
        "Price": "Price_anom",
    }
    flags = {k: df_raw[v].fillna(0).astype(int).to_numpy() for k, v in cols.items()}
    labels = list(cols.keys())

    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            xa = flags[a] == 1
            xb = flags[b] == 1
            inter = float(np.sum(xa & xb))
            union = float(np.sum(xa | xb))
            mat[i, j] = inter / union if union > 0 else 0.0

    return pd.DataFrame(mat, index=labels, columns=labels)


def plot_joint_anomaly_upset(df_raw: pd.DataFrame, out_path: str) -> None:
    """
    UpSet-style figure made with matplotlib only:
      - Top panel: counts of most frequent anomaly intersections.
      - Bottom panel: pairwise Jaccard overlap matrix.
    """
    combos = _build_combo_table(df_raw).head(12)
    overlap = _pairwise_overlap_matrix(df_raw)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.4)

    ax_top = fig.add_subplot(gs[0, 0])
    x = np.arange(len(combos))
    ax_top.bar(x, combos["count"].values, color="#457b9d", alpha=0.9)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(combos["active_targets"].values, rotation=35, ha="right")
    ax_top.set_ylabel("Count")
    ax_top.set_title("UpSet-style Joint Anomaly Intersections (Top 12)")
    ax_top.grid(axis="y", linestyle="--", alpha=0.35)

    for xi, cnt in enumerate(combos["count"].values):
        ax_top.text(xi, cnt + max(1, 0.01 * combos["count"].max()), str(int(cnt)), ha="center", fontsize=8)

    ax_bottom = fig.add_subplot(gs[1, 0])
    im = ax_bottom.imshow(overlap.values, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax_bottom.set_xticks(np.arange(len(overlap.columns)))
    ax_bottom.set_xticklabels(overlap.columns)
    ax_bottom.set_yticks(np.arange(len(overlap.index)))
    ax_bottom.set_yticklabels(overlap.index)
    ax_bottom.set_title("Pairwise Overlap (Jaccard Index)")

    for i in range(overlap.shape[0]):
        for j in range(overlap.shape[1]):
            ax_bottom.text(j, i, f"{overlap.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax_bottom, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard")

    fig.suptitle("Joint Anomaly Analysis Across Price, Load, Solar, and Wind", y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    _ensure_dir(args.out_dir)

    df_raw = _load_csv(args.anoms_raw, parse_dates=["Time (UTC)"])
    df_ref = _load_csv(args.anoms_refined, parse_dates=["Time (UTC)"])

    if "Time (UTC)" not in df_raw.columns or "Time (UTC)" not in df_ref.columns:
        _fail("Both input tables must contain 'Time (UTC)'.")

    if "target" not in df_ref.columns or "is_vetoed" not in df_ref.columns:
        _fail("Refined anomalies table must contain 'target' and 'is_vetoed'.")

    p1 = os.path.join(args.out_dir, "asp_impact_heatmap.png")
    p2 = os.path.join(args.out_dir, "threshold_uncertainty_by_hour.png")
    p3 = os.path.join(args.out_dir, "joint_anomaly_upset.png")

    plot_asp_impact_heatmap(df_ref, p1)
    plot_uncertainty_band_distribution(df_raw, p2)
    plot_joint_anomaly_upset(df_raw, p3)

    print(f"[INFO] Saved: {p1}")
    print(f"[INFO] Saved: {p2}")
    print(f"[INFO] Saved: {p3}")


if __name__ == "__main__":
    main()
