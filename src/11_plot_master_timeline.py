#src/11_plot_master_timeline.py

#!/usr/bin/env python3
"""
Plot a master timeline figure for a single signal with a shared time axis.

Panels:
1) Actual vs predicted
2) Residuals
3) Anomaly score + flags (per-target overshoot)
4) Finance PnL (baseline vs strategy)

ASP-refined events are shaded as vertical bands across all panels,
coloured by severity.

This script is tailored to this repo's CSVs:

    - anomalies: reports/tables/anomalies_2022.csv
      columns like: CF_Solar_true, CF_Solar_pred, CF_Solar_resid,
                    CF_Solar_abs, CF_Solar_thr, CF_Solar_anom, ...

    - events: reports/tables/anomaly_events_2022.csv
      columns: event_id, signal_id, start_ts, end_ts, duration_hours,
               n_points, peak_score, mean_score, severity_label

    - finance: reports/tables/finance_backtest_2022.csv
      columns: Price_true, Price_pred, dPrice_true, dPrice_pred,
               Load_MW_true, Load_MW_pred,
               CF_Solar_pred, CF_Wind_pred,
               combined_score, action,
               utility, utility_baseline, utility_diff

Usage example:

    python src/11_plot_master_timeline.py \
        --signal Price \
        --start 2022-02-01 \
        --end 2022-03-01 \
        --output reports/figures/master_timeline_Price_2022-02.png
"""

import argparse
import sys
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------ utilities ------------------------


def _fail(msg: str) -> None:
    sys.stderr.write(f"[ERROR] {msg}\n")
    sys.exit(1)


def _warn(msg: str) -> None:
    sys.stderr.write(f"[WARN] {msg}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a shared-x master timeline for one signal (repo-specific)."
    )

    parser.add_argument(
        "--signal",
        default="Price",
        choices=["CF_Solar", "CF_Wind", "Load_MW", "Price"],
        help=(
            "Target to visualize (one of: CF_Solar, CF_Wind, Load_MW, Price). "
            "Default: Price."
        ),
    )

    parser.add_argument(
        "--anomalies",
        default="reports/tables/anomalies_2022.csv",
        help="Point-level anomalies CSV (default: reports/tables/anomalies_2022.csv).",
    )
    parser.add_argument(
        "--events",
        default="reports/tables/anomaly_events_2022.csv",
        help="Event-level anomalies CSV (default: reports/tables/anomaly_events_2022.csv).",
    )
    parser.add_argument(
        "--finance",
        default="reports/tables/finance_backtest_2022.csv",
        help="Finance backtest CSV (default: reports/tables/finance_backtest_2022.csv).",
    )

    parser.add_argument(
        "--start",
        default=None,
        help="Start date for plotting (inclusive), e.g. 2022-02-01 (optional).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date for plotting (inclusive), e.g. 2022-03-01 (optional).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PNG path. If not provided, a default based on signal name and "
            "date range will be used in reports/figures/."
        ),
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=1.0,
        help="Horizontal threshold for anomaly score plot (default: 1.0).",
    )

    return parser.parse_args()


def _parse_time_window(
    s: Optional[str], e: Optional[str]
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    start_ts = pd.to_datetime(s) if s is not None else None
    end_ts = pd.to_datetime(e) if e is not None else None
    return start_ts, end_ts


def _apply_time_window(
    df: pd.DataFrame, ts_col: str, start_ts: Optional[pd.Timestamp], end_ts: Optional[pd.Timestamp]
) -> pd.DataFrame:
    if start_ts is not None:
        df = df[df[ts_col] >= start_ts]
    if end_ts is not None:
        df = df[df[ts_col] <= end_ts]
    return df


# ------------------------ data loaders ------------------------


def load_anomalies_repo(path: str, signal: str) -> pd.DataFrame:
    """
    Load anomalies_2022.csv (wide format) and extract columns for one signal.

    Expected columns:
        f"{signal}_true", f"{signal}_pred", f"{signal}_resid",
        f"{signal}_abs",  f"{signal}_thr",  f"{signal}_anom",
        plus 'combined_score'.

    Returns a DataFrame with index Time (UTC) and columns:
        ['actual', 'pred', 'resid', 'score', 'flag', 'combined_score'].
    """
    try:
        df = (
            pd.read_csv(path, parse_dates=["Time (UTC)"])
            .set_index("Time (UTC)")
            .sort_index()
        )
    except FileNotFoundError:
        _fail(f"Anomalies file not found: {path}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read anomalies CSV '{path}': {exc}")

    cols_needed = [
        f"{signal}_true",
        f"{signal}_pred",
        f"{signal}_resid",
        f"{signal}_abs",
        f"{signal}_thr",
        f"{signal}_anom",
    ]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        _fail(
            f"Missing required columns for signal '{signal}' in anomalies CSV: {missing}. "
            f"Available columns include: {list(df.columns)[:10]} ..."
        )

    # per-target score = max(0, abs / thr - 1)
    abs_vals = df[f"{signal}_abs"].astype(float)
    thr_vals = df[f"{signal}_thr"].astype(float)
    score = np.maximum(0.0, abs_vals / (thr_vals + 1e-9) - 1.0)

    # flag from *_anom
    flag = df[f"{signal}_anom"].astype(int)

    out = pd.DataFrame(
        {
            "actual": df[f"{signal}_true"].astype(float),
            "pred": df[f"{signal}_pred"].astype(float),
            "resid": df[f"{signal}_resid"].astype(float),
            "score": score,
            "flag": flag,
        },
        index=df.index,
    )

    if "combined_score" in df.columns:
        out["combined_score"] = df["combined_score"].astype(float)
    else:
        out["combined_score"] = np.nan

    out.index.name = "Time (UTC)"
    return out


def load_events_repo(path: str, signal: str) -> pd.DataFrame:
    """
    Load anomaly_events_2022.csv and filter events for one signal.
    """
    try:
        ev = pd.read_csv(path, parse_dates=["start_ts", "end_ts"])
    except FileNotFoundError:
        _fail(f"Events file not found: {path}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read events CSV '{path}': {exc}")

    cols = ["signal_id", "start_ts", "end_ts", "severity_label"]
    missing = [c for c in cols if c not in ev.columns]
    if missing:
        _fail(
            f"Missing required columns in events CSV: {missing}. "
            f"Available columns: {list(ev.columns)}"
        )

    ev = ev[ev["signal_id"] == signal].copy()
    ev = ev.sort_values("start_ts").reset_index(drop=True)
    return ev


def load_finance_repo(path: str) -> pd.DataFrame:
    """
    Load finance_backtest_2022.csv and construct cumulative PnL series.
    """
    try:
        df = (
            pd.read_csv(path, parse_dates=["Time (UTC)"])
            .set_index("Time (UTC)")
            .sort_index()
        )
    except FileNotFoundError:
        _fail(f"Finance file not found: {path}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read finance CSV '{path}': {exc}")

    cols = ["utility", "utility_baseline"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        _fail(
            f"Missing required columns in finance CSV: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame(index=df.index)
    out["cum_baseline"] = df["utility_baseline"].cumsum()
    out["cum_strategy"] = df["utility"].cumsum()
    out.index.name = "Time (UTC)"
    return out


# ------------------------ plotting ------------------------


def plot_master_timeline(
    anomalies: pd.DataFrame,
    events: pd.DataFrame,
    finance: pd.DataFrame,
    signal: str,
    score_threshold: float,
    output_path: str,
) -> None:
    if anomalies.empty:
        _fail("Anomalies DataFrame is empty after filtering. Nothing to plot.")

    # x-axis for anomalies
    t_anom = anomalies.index

    # figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    # Panel 1: actual vs predicted
    ax1.plot(t_anom, anomalies["actual"], label=f"Actual {signal}")
    ax1.plot(t_anom, anomalies["pred"], linestyle="--", label=f"Predicted {signal}")
    ax1.set_ylabel(signal)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: residuals
    ax2.plot(t_anom, anomalies["resid"], label="Residual")
    ax2.axhline(0.0, linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Residual")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Panel 3: anomaly score + flags
    ax3.plot(t_anom, anomalies["score"], label="Per-target score")
    ax3.axhline(score_threshold, linestyle="--", label=f"Threshold={score_threshold}")

    flagged = anomalies[anomalies["flag"] == 1]
    if not flagged.empty:
        ax3.scatter(
            flagged.index,
            flagged["score"],
            marker="o",
            s=20,
            label="Flagged",
        )

    ax3.set_ylabel("Score")
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle=":", alpha=0.5)

    # Panel 4: finance PnL
    if not finance.empty:
        ax4.plot(finance.index, finance["cum_baseline"], label="Baseline PnL")
        ax4.plot(finance.index, finance["cum_strategy"], label="Strategy PnL")
        ax4.legend(loc="upper left")
    else:
        ax4.text(
            0.5,
            0.5,
            "No finance data for this window",
            transform=ax4.transAxes,
            ha="center",
            va="center",
        )
    ax4.set_ylabel("Cumulative PnL")
    ax4.set_xlabel("Time")
    ax4.grid(True, linestyle=":", alpha=0.5)

    # Shade ASP-refined events, coloured by severity
    if not events.empty:
        color_map = {
            "LOW": "#cdeccd",      # pale green
            "MEDIUM": "#ffe0b3",   # pale orange
            "HIGH": "#ffcccc",     # pale red
            "UNKNOWN": "#dddddd",  # grey
        }
        for _, ev in events.iterrows():
            start_ts = ev["start_ts"]
            end_ts = ev["end_ts"]
            sev = ev.get("severity_label", "UNKNOWN")
            color = color_map.get(sev, "#dddddd")
            for ax in axes:
                ax.axvspan(start_ts, end_ts, color=color, alpha=0.25)

    fig.tight_layout()

    try:
        fig.savefig(output_path, dpi=200)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to save figure to '{output_path}': {exc}")
    finally:
        plt.close(fig)

    print(f"[INFO] Saved master timeline figure to '{output_path}'")


# ------------------------ main ------------------------


def main() -> None:
    args = _parse_args()
    start_ts, end_ts = _parse_time_window(args.start, args.end)

    anomalies = load_anomalies_repo(args.anomalies, args.signal)
    events = load_events_repo(args.events, args.signal)
    finance = load_finance_repo(args.finance)

    # convert indices to columns for windowing
    anomalies = anomalies.reset_index().rename(columns={"Time (UTC)": "ts"})
    events = events.copy()
    finance = finance.reset_index().rename(columns={"Time (UTC)": "ts"})

    anomalies = _apply_time_window(anomalies, "ts", start_ts, end_ts)
    events = _apply_time_window(events, "start_ts", start_ts, end_ts)
    finance = _apply_time_window(finance, "ts", start_ts, end_ts)

    # set index back to ts
    anomalies = anomalies.set_index("ts").sort_index()
    finance = finance.set_index("ts").sort_index()

    # Default output path if not specified
    if args.output is None:
        date_suffix = ""
        if start_ts is not None:
            date_suffix += f"_{start_ts.date()}"
        if end_ts is not None:
            date_suffix += f"_{end_ts.date()}"
        out_name = f"master_timeline_{args.signal}{date_suffix}.png"
        output_path = f"reports/figures/{out_name}"
    else:
        output_path = args.output

    plot_master_timeline(
        anomalies=anomalies,
        events=events,
        finance=finance,
        signal=args.signal,
        score_threshold=args.score_threshold,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
