#!/usr/bin/env python3
"""
Build event-level anomaly table from point-level anomaly CSV.

Default for this repo (no arguments):
    Input : reports/tables/anomalies_refined.csv
    Output: reports/tables/anomaly_events_2022.csv
    Timestamp column: "Time (UTC)"
    Signal column   : target
    Index column    : hour_index
    Flag column     : final_flag
    Score column    : anomaly_score (falls back to combined_score)

How severity is computed
------------------------
1. If the input CSV already has a numeric score column (anomaly_score or
   combined_score), that is used directly.
2. Otherwise, this script tries to load:
       reports/tables/anomalies_2022.csv
   and for each (Time, target) pair, computes a per-target score:

       score = max(0, target_abs / target_thr - 1)

   where target_abs = e.g. 'Price_abs', target_thr = 'Price_thr'.
   If those columns are missing, it falls back to 'combined_score'.

That score is then aggregated per event (peak and mean) and mapped to:

    peak_score < 1.0   -> LOW
    1.0 ≤ peak_score < 1.5 -> MEDIUM
    peak_score ≥ 1.5   -> HIGH
"""

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd


# -------------------------------------------------------------------
# Basic helpers
# -------------------------------------------------------------------


def _fail(msg: str) -> None:
    sys.stderr.write(f"[ERROR] {msg}\n")
    sys.exit(1)


def _warn(msg: str) -> None:
    sys.stderr.write(f"[WARN] {msg}\n")


def _validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        _fail(
            f"Missing required column(s): {missing}. "
            f"Available columns are: {list(df.columns)}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build event-level anomaly table from point-level anomalies."
    )
    parser.add_argument(
        "--input",
        default="reports/tables/anomalies_refined.csv",
        help=(
            "Input CSV with point-level anomalies "
            "(default: 'reports/tables/anomalies_refined.csv')."
        ),
    )
    parser.add_argument(
        "--output",
        default="reports/tables/anomaly_events_2022.csv",
        help=(
            "Output CSV for event-level anomalies "
            "(default: 'reports/tables/anomaly_events_2022.csv')."
        ),
    )
    parser.add_argument(
        "--timestamp-col",
        default="Time (UTC)",
        help='Name of the timestamp column (default: "Time (UTC)").',
    )
    parser.add_argument(
        "--flag-col",
        default="final_flag",
        help=(
            "Name of the binary anomaly flag column (default: 'final_flag'). "
            "If this column is missing, all rows are treated as anomalies."
        ),
    )
    parser.add_argument(
        "--score-col",
        default="anomaly_score",
        help=(
            "Name of the anomaly score column (default: 'anomaly_score'). "
            "If missing, the script will try 'anomaly_score' or 'combined_score' "
            "automatically; if none found, it will try to attach per-target "
            "scores from 'reports/tables/anomalies_2022.csv'."
        ),
    )
    parser.add_argument(
        "--signal-col",
        default="target",
        help=(
            "Name of the signal/target identifier column "
            '(default: "target"). '
            "If not present, all rows are treated as one global series."
        ),
    )
    parser.add_argument(
        "--index-col",
        default="hour_index",
        help=(
            "Name of an integer index column (default: 'hour_index') used to "
            "detect contiguous anomaly runs. In FLAG mode, a new event starts "
            "when the flag goes from 0→1 or when index jumps by more than 1."
        ),
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=1,
        help=(
            "Minimum number of points in a contiguous block to keep as an event "
            "(default: 1)."
        ),
    )
    return parser.parse_args()


def _ensure_datetime(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Convert timestamp column to pandas datetime and drop any timezone."""
    try:
        ser = pd.to_datetime(df[ts_col])
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to parse '{ts_col}' as datetime: {exc}")

    # If tz-aware, remove timezone to get naive UTC timestamps
    if getattr(ser.dt, "tz", None) is not None:
        ser = ser.dt.tz_localize(None)

    df[ts_col] = ser
    return df



def _compute_severity(peak_score: Optional[float]) -> str:
    if peak_score is None or pd.isna(peak_score):
        return "UNKNOWN"
    if peak_score < 1.0:
        return "LOW"
    if peak_score < 1.5:
        return "MEDIUM"
    return "HIGH"


# -------------------------------------------------------------------
# Score enrichment from anomalies_2022.csv
# -------------------------------------------------------------------


def _attach_scores_from_base(
    df: pd.DataFrame,
    ts_col: str,
    signal_col: Optional[str],
    base_path: str = "reports/tables/anomalies_2022.csv",
) -> pd.DataFrame:
    """
    If df has no useful score columns, try to attach per-target scores
    from anomalies_2022.csv based on (timestamp, target) pairs.

    Score per target/time = max(0, abs / thr - 1) if abs & thr exist,
    otherwise fall back to combined_score.
    """
    if signal_col is None or signal_col not in df.columns:
        _warn(
            "Cannot attach scores from base: signal column not present. "
            "Severity will remain UNKNOWN."
        )
        return df

    if not os.path.exists(base_path):
        _warn(
            f"Base anomalies CSV '{base_path}' not found; cannot derive scores. "
            "Severity will remain UNKNOWN."
        )
        return df

    try:
        base = pd.read_csv(base_path)
    except Exception as exc:  # noqa: BLE001
        _warn(f"Failed to read base anomalies CSV '{base_path}': {exc}")
        return df

    if ts_col not in base.columns:
        _warn(
            f"Base anomalies CSV '{base_path}' has no '{ts_col}' column. "
            "Cannot join scores; severity will remain UNKNOWN."
        )
        return df

    # Normalize timestamps in BOTH frames to naive datetime (UTC semantics)
    base[ts_col] = pd.to_datetime(base[ts_col], errors="coerce")
    if getattr(base[ts_col].dt, "tz", None) is not None:
        base[ts_col] = base[ts_col].dt.tz_localize(None)

    # df[timestamp_col] has already passed through _ensure_datetime,
    # so it's also naive datetime64[ns].

    # Canonical targets – intersect with what actually appears in the refined file
    canonical_targets = ["CF_Solar", "CF_Wind", "Load_MW", "Price"]
    present_signals = set(str(s) for s in df[signal_col].unique())
    targets = [t for t in canonical_targets if t in present_signals]

    if not targets:
        _warn(
            "No canonical targets (CF_Solar/CF_Wind/Load_MW/Price) found in "
            f"'{signal_col}' column of refined anomalies. "
            "Skipping score attachment."
        )
        return df

    records = []
    use_combined = "combined_score" in base.columns

    for _, row in base.iterrows():
        ts = row[ts_col]
        if pd.isna(ts):
            continue
        for tgt in targets:
            abs_col = f"{tgt}_abs"
            thr_col = f"{tgt}_thr"
            score = None

            if abs_col in base.columns and thr_col in base.columns:
                val_abs = row[abs_col]
                val_thr = row[thr_col]
                if pd.notna(val_abs) and pd.notna(val_thr) and val_thr != 0:
                    score = max(0.0, float(val_abs) / float(val_thr) - 1.0)
            elif use_combined:
                cs = row["combined_score"]
                if pd.notna(cs):
                    score = float(cs)

            if score is not None:
                records.append(
                    {
                        ts_col: ts,
                        signal_col: tgt,
                        "anomaly_score": score,
                    }
                )

    if not records:
        _warn(
            "Could not derive any per-target scores from base anomalies CSV. "
            "Severity will remain UNKNOWN."
        )
        return df

    score_df = pd.DataFrame(records)
    score_df[ts_col] = pd.to_datetime(score_df[ts_col], errors="coerce")
    # Ensure also naive (just in case)
    if getattr(score_df[ts_col].dt, "tz", None) is not None:
        score_df[ts_col] = score_df[ts_col].dt.tz_localize(None)

    merged = df.merge(
        score_df,
        on=[ts_col, signal_col],
        how="left",
    )

    return merged



# -------------------------------------------------------------------
# Event builders
# -------------------------------------------------------------------


def _build_events_flag_mode(
    df_group: pd.DataFrame,
    ts_col: str,
    flag_col: str,
    score_col: Optional[str],
    signal_name: Optional[str],
    min_points: int,
    index_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build events when we have a series with a binary anomaly flag.

    Events are contiguous runs of flag == True. If index_col is provided,
    contiguity also requires index_col to increment by 1.
    """
    df_group = df_group.copy()
    is_anom = df_group[flag_col].astype(bool)
    if not is_anom.any():
        return pd.DataFrame()

    df_group["_is_anom"] = is_anom

    if index_col is not None and index_col in df_group.columns:
        idx = df_group[index_col].astype("int64")
        contiguous = (
            is_anom
            & is_anom.shift(fill_value=False)
            & (idx.diff() == 1)
        )
        new_block = is_anom & ~contiguous
    else:
        new_block = is_anom & ~is_anom.shift(fill_value=False)

    df_group["_block_id"] = new_block.cumsum()

    df_anom = df_group[df_group["_is_anom"]]
    if df_anom.empty:
        return pd.DataFrame()

    event_rows = []
    for _, df_block in df_anom.groupby("_block_id"):
        if len(df_block) < min_points:
            continue

        start_ts = df_block[ts_col].min()
        end_ts = df_block[ts_col].max()
        n_points = len(df_block)
        duration_hours = (end_ts - start_ts).total_seconds() / 3600.0

        if score_col is not None and score_col in df_block.columns:
            peak_score = float(df_block[score_col].max())
            mean_score = float(df_block[score_col].mean())
        else:
            peak_score = None
            mean_score = None

        severity_label = _compute_severity(peak_score)
        row = {
            "signal_id": signal_name if signal_name is not None else None,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_hours": duration_hours,
            "n_points": n_points,
            "peak_score": peak_score,
            "mean_score": mean_score,
            "severity_label": severity_label,
        }
        event_rows.append(row)

    if not event_rows:
        return pd.DataFrame()

    return pd.DataFrame(event_rows)


def _build_events_all_rows_mode(
    df_group: pd.DataFrame,
    ts_col: str,
    index_col: Optional[str],
    score_col: Optional[str],
    signal_name: Optional[str],
    min_points: int,
) -> pd.DataFrame:
    """
    Build events when all rows in df_group are anomalies (no flag column).

    If index_col is provided and present, events are contiguous runs where
    index_col increments by 1. Otherwise, each row becomes its own event.
    """
    df_group = df_group.copy()
    event_blocks = []

    if index_col is not None and index_col in df_group.columns:
        df_group = df_group.sort_values(index_col).reset_index(drop=True)
        prev_idx = None
        current_block_indices = []

        for i, row in df_group.iterrows():
            idx_val = row[index_col]
            if prev_idx is None or idx_val == prev_idx + 1:
                current_block_indices.append(i)
            else:
                if current_block_indices:
                    df_block = df_group.loc[current_block_indices]
                    if len(df_block) >= min_points:
                        event_blocks.append(df_block)
                current_block_indices = [i]
            prev_idx = idx_val

        if current_block_indices:
            df_block = df_group.loc[current_block_indices]
            if len(df_block) >= min_points:
                event_blocks.append(df_block)
    else:
        # No index, treat each row as its own event
        for _, row in df_group.iterrows():
            event_blocks.append(pd.DataFrame([row]))

    summary_rows = []
    for df_block in event_blocks:
        start_ts = df_block[ts_col].min()
        end_ts = df_block[ts_col].max()
        n_points = len(df_block)
        duration_hours = (end_ts - start_ts).total_seconds() / 3600.0

        if score_col is not None and score_col in df_block.columns:
            peak_score = float(df_block[score_col].max())
            mean_score = float(df_block[score_col].mean())
        else:
            peak_score = None
            mean_score = None

        severity_label = _compute_severity(peak_score)
        row = {
            "signal_id": signal_name if signal_name is not None else None,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_hours": duration_hours,
            "n_points": n_points,
            "peak_score": peak_score,
            "mean_score": mean_score,
            "severity_label": severity_label,
        }
        summary_rows.append(row)

    if not summary_rows:
        return pd.DataFrame()
    return pd.DataFrame(summary_rows)


# -------------------------------------------------------------------
# Main driver
# -------------------------------------------------------------------


def build_event_table(
    input_path: str,
    output_path: str,
    timestamp_col: str,
    flag_col: str,
    score_col: Optional[str],
    signal_col: Optional[str],
    index_col: Optional[str],
    min_points: int,
) -> None:
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        _fail(f"Input file not found: {input_path}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read input CSV '{input_path}': {exc}")

    if df.empty:
        _warn(f"Input CSV '{input_path}' is empty. No events will be written.")
        empty_cols = [
            "event_id",
            "signal_id",
            "start_ts",
            "end_ts",
            "duration_hours",
            "n_points",
            "peak_score",
            "mean_score",
            "severity_label",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(output_path, index=False)
        return

    _validate_columns(df, [timestamp_col])
    df = _ensure_datetime(df, timestamp_col)

    # ---- If we have no usable score columns, try to join from anomalies_2022.csv ----
    if (
        (score_col not in df.columns)
        and ("anomaly_score" not in df.columns)
        and ("combined_score" not in df.columns)
    ):
        df = _attach_scores_from_base(
            df=df,
            ts_col=timestamp_col,
            signal_col=signal_col,
            base_path="reports/tables/anomalies_2022.csv",
        )

    # --------- resolve score column (auto-fallback) ----------
    effective_score_col: Optional[str] = None
    if score_col and score_col in df.columns:
        effective_score_col = score_col
    else:
        for cand in ("anomaly_score", "combined_score"):
            if cand in df.columns:
                if score_col != cand:
                    _warn(
                        f"Score column '{score_col}' not found; using '{cand}' instead."
                    )
                effective_score_col = cand
                break

        if effective_score_col is None and score_col:
            _warn(
                f"Score column '{score_col}' not found and no fallback "
                "('anomaly_score'/'combined_score') present. "
                "Severity labels will be 'UNKNOWN'."
            )

    # Sort by timestamp and optionally signal
    sort_cols = [timestamp_col]
    if signal_col is not None and signal_col in df.columns:
        sort_cols.insert(0, signal_col)
    df = df.sort_values(sort_cols).reset_index(drop=True)

    all_events = []

    flag_available = flag_col in df.columns

    if signal_col is not None and signal_col in df.columns:
        for signal_name, df_signal in df.groupby(signal_col):
            if flag_available:
                events_df = _build_events_flag_mode(
                    df_group=df_signal,
                    ts_col=timestamp_col,
                    flag_col=flag_col,
                    score_col=effective_score_col,
                    signal_name=str(signal_name),
                    min_points=min_points,
                    index_col=index_col,
                )
            else:
                # For refined anomalies, there is usually no flag column: treat all rows as anomalies
                events_df = _build_events_all_rows_mode(
                    df_group=df_signal,
                    ts_col=timestamp_col,
                    index_col=index_col,
                    score_col=effective_score_col,
                    signal_name=str(signal_name),
                    min_points=min_points,
                )
            if not events_df.empty:
                all_events.append(events_df)
    else:
        # Single global series
        if flag_available:
            events_df = _build_events_flag_mode(
                df_group=df,
                ts_col=timestamp_col,
                flag_col=flag_col,
                score_col=effective_score_col,
                signal_name=None,
                min_points=min_points,
                index_col=index_col,
            )
        else:
            events_df = _build_events_all_rows_mode(
                df_group=df,
                ts_col=timestamp_col,
                index_col=index_col,
                score_col=effective_score_col,
                signal_name=None,
                min_points=min_points,
            )
        if not events_df.empty:
            all_events.append(events_df)

    if not all_events:
        _warn("No anomaly events found in the input data. Writing empty table.")
        empty_cols = [
            "event_id",
            "signal_id",
            "start_ts",
            "end_ts",
            "duration_hours",
            "n_points",
            "peak_score",
            "mean_score",
            "severity_label",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(output_path, index=False)
        return

    events = pd.concat(all_events, ignore_index=True)
    events.insert(0, "event_id", range(1, len(events) + 1))

    try:
        events.to_csv(output_path, index=False)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to write output CSV '{output_path}': {exc}")

    print(
        f"[INFO] Wrote {len(events)} anomaly events to '{output_path}' "
        f"from input '{input_path}'."
    )


def main() -> None:
    args = _parse_args()
    build_event_table(
        input_path=args.input,
        output_path=args.output,
        timestamp_col=args.timestamp_col,
        flag_col=args.flag_col,
        score_col=args.score_col,
        signal_col=args.signal_col,
        index_col=args.index_col,
        min_points=args.min_points,
    )


if __name__ == "__main__":
    main()
