# src/01_preprocess_build_features.py

#!/usr/bin/env python3
"""
Preprocess raw Austrian hourly data (2017–2022) and build engineered features.

Steps:
1. Load config (configs/base.yaml by default)
2. Load raw hourly data (load, RES, prices, weather)
3. Load installed capacity per production type
4. Compute:
   - Capacity factors (CF_Solar, CF_Wind)
   - Calendar features (hour/dow/month encodings)
   - Meteorological derived features (air_density, wind_speed_100m, pv_proxy)
   - Wind power proxy
   - Holiday/weekend/special-day flags
5. Save engineered CSVs to cfg['engineered_csv'] and cfg['engineered_full_csv']

You can override paths and date range via CLI:
    python src/01_preprocess_build_features.py \
        --config configs/base.yaml \
        --start-date 2017-01-01 \
        --end-date 2022-12-31
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


DEFAULT_CONFIG_PATH = "configs/base.yaml"


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
        description="Preprocess raw AT data and build engineered features."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to base config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date (inclusive) for the engineered dataset, e.g. 2017-01-01.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date (inclusive), e.g. 2022-12-31 23:00.",
    )
    parser.add_argument(
        "--raw-hourly-csv",
        default=None,
        help="Override path to raw hourly CSV (otherwise from config or default).",
    )
    parser.add_argument(
        "--capacity-csv",
        default=None,
        help="Override path to installed capacity CSV (otherwise from config or default).",
    )
    return parser.parse_args()


def _load_config(path: str) -> Dict:
    """Load YAML config and ensure it's a dict."""
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


def _resolve_date_range(
    cfg: Dict, start_cli: Optional[str], end_cli: Optional[str]
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Determine start/end timestamps for the engineered dataset.

    Priority:
    1) CLI overrides (--start-date, --end-date)
    2) Config keys cfg['data_start'], cfg['data_end'] if present
    3) Fallback to 2017-01-01 and 2022-12-31 23:00
    """
    if start_cli is not None and end_cli is not None:
        start_str = start_cli
        end_str = end_cli
    else:
        start_str = cfg.get("data_start", "2017-01-01")
        end_str = cfg.get("data_end", "2022-12-31 23:00")

    try:
        start_ts = pd.to_datetime(start_str)
        end_ts = pd.to_datetime(end_str)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to parse data_start/data_end as datetime: {exc}")

    if end_ts < start_ts:
        _fail(
            f"Invalid date range: end_date ({end_ts}) < start_date ({start_ts}). "
            "Please fix config or CLI arguments."
        )

    return start_ts, end_ts


def _resolve_paths(cfg: Dict, args: argparse.Namespace) -> Tuple[str, str, str, str]:
    """
    Resolve the key paths:
      - raw_hourly_csv
      - installed_capacity_csv
      - engineered_csv
      - engineered_full_csv
    """
    raw_default = cfg.get(
        "raw_hourly_csv",
        "data/AT_hourly_MW_and_Price.csv",
    )
    cap_default = cfg.get(
        "installed_capacity_csv",
        "data/Installed Capacity per Production Type_201701010000-202301010000.csv",
    )

    raw_hourly_csv = args.raw_hourly_csv or raw_default
    capacity_csv = args.capacity_csv or cap_default

    engineered_csv = cfg.get("engineered_csv")
    engineered_full_csv = cfg.get("engineered_full_csv")

    if not isinstance(engineered_csv, str) or not isinstance(engineered_full_csv, str):
        _fail(
            "Config keys 'engineered_csv' and 'engineered_full_csv' must be present "
            "and strings in base.yaml. Example:\n"
            "  engineered_csv: data/AT_engineered_hourly.csv\n"
            "  engineered_full_csv: data/AT_engineered_hourly_FULL.csv"
        )

    return raw_hourly_csv, capacity_csv, engineered_csv, engineered_full_csv


def _ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory for a file path exists."""
    out_dir = os.path.dirname(path) or "."
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to create directory '{out_dir}': {exc}")


def annual_to_hourly(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Map annual capacity series (indexed by year) to an hourly index."""
    years = pd.Series(index=idx, data=idx.year)
    mapped = years.map(s).astype(float)
    # Forward/backward fill just in case some year is missing
    return mapped.fillna(method="ffill").fillna(method="bfill")


def _load_raw_hourly(path: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Load raw hourly CSV and subset to [start_ts, end_ts]."""
    if not os.path.exists(path):
        _fail(f"Raw hourly CSV not found: {path}")

    try:
        raw = pd.read_csv(path, parse_dates=["Time (UTC)"])
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read raw hourly CSV '{path}': {exc}")

    if "Time (UTC)" not in raw.columns:
        _fail(
            "Raw hourly CSV must contain a 'Time (UTC)' column. "
            f"Available columns: {list(raw.columns)}"
        )

    raw = raw.set_index("Time (UTC)").sort_index()

    # Drop duplicate timestamps if any
    if raw.index.has_duplicates:
        _warn(
            "Raw hourly index has duplicate timestamps. "
            "Keeping the first occurrence per timestamp."
        )
        raw = raw[~raw.index.duplicated(keep="first")]

    # Subset to desired time window
    raw = raw.loc[start_ts:end_ts]

    if raw.empty:
        _fail(
            f"Raw hourly data is empty after subsetting to [{start_ts}, {end_ts}]. "
            "Check your date range or raw CSV."
        )

    # Check for key columns used below
    required_cols = [
        "Solar_MW",
        "Wind_MW",
        "temperature_2m (°C)",
        "surface_pressure (hPa)",
        "relative_humidity_2m (%)",
        "wind_speed_10m (m/s)",
        "shortwave_radiation (W/m²)",
        "DA_Price_EUR_MWh",
    ]
    missing = [c for c in required_cols if c not in raw.columns]
    if missing:
        _fail(
            f"Missing required column(s) in raw hourly CSV: {missing}. "
            f"Available columns: {list(raw.columns)}"
        )

    return raw


def _load_capacity(path: str) -> pd.DataFrame:
    """Load installed capacity CSV and prepare it for solar/wind capacity extraction."""
    if not os.path.exists(path):
        _fail(f"Installed capacity CSV not found: {path}")

    try:
        cap = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read capacity CSV '{path}': {exc}")

    if "Production Type" not in cap.columns:
        _fail(
            "Installed capacity CSV must contain 'Production Type' column. "
            f"Available columns: {list(cap.columns)}"
        )

    # Melt year columns into rows
    cap_melt = cap.melt(
        id_vars=["Production Type"],
        var_name="year",
        value_name="MW",
    )
    # Extract year as int from column names
    cap_melt["year"] = cap_melt["year"].astype(str).str.extract(r"(\d{4})").astype(int)
    cap_melt["ptype_norm"] = cap_melt["Production Type"].astype(str).str.lower()

    return cap_melt


def _capacity_series(cap_melt: pd.DataFrame, pattern: str) -> pd.Series:
    """
    Aggregate capacity per year for a given production type pattern ('solar', 'wind').
    """
    mask = cap_melt["ptype_norm"].str.contains(pattern)
    sub = cap_melt[mask].copy()
    if sub.empty:
        _warn(
            f"No installed capacity rows matched pattern '{pattern}'. "
            "Resulting capacity factors may be NaN."
        )
        return pd.Series(dtype=float)

    return sub.groupby("year")["MW"].sum()


def _add_calendar_features(df: pd.DataFrame) -> None:
    """Add hour/dow/month sinusoidal encodings in-place."""
    idx = df.index
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12.0)


def _add_meteo_features(df: pd.DataFrame) -> None:
    """
    Add derived meteorological features in-place:
      - air_density_kgm3
      - wind_speed_100m (m/s)
      - pv_proxy
      - wind_power_proxy
    """
    T_C = df["temperature_2m (°C)"]
    T_K = T_C + 273.15
    p_Pa = df["surface_pressure (hPa)"] * 100.0
    RH = (df["relative_humidity_2m (%)"] / 100.0).clip(0.0, 1.0)

    # Saturation vapor pressure (Pa)
    e_s = 610.94 * np.exp((17.625 * T_C) / (T_C + 243.04))
    # Actual vapor pressure
    e = RH * e_s

    # Specific humidity q
    denom = (p_Pa - (1 - 0.622) * e).clip(lower=1.0)
    q = 0.622 * e / denom

    # Air density
    rho_air = p_Pa / (287.05 * (T_K * (1.0 + 0.61 * q)))
    df["air_density_kgm3"] = rho_air

    # Wind speed at 100m (power-law profile)
    v10 = df["wind_speed_10m (m/s)"].clip(lower=0.0)
    v100 = v10 * (100.0 / 10.0) ** 0.143
    df["wind_speed_100m (m/s)"] = v100

    # Simple PV power proxy
    sw = df["shortwave_radiation (W/m²)"].clip(lower=0.0)
    temp_penalty = (1.0 - 0.005 * (T_C - 25.0).clip(lower=0.0))
    df["pv_proxy"] = (sw * temp_penalty).clip(lower=0.0)

    # Wind power proxy ~ rho * v^3 with upper clipping to avoid extreme spikes
    wind_power_raw = (rho_air * (v100 ** 3)).fillna(0.0)
    upper_clip = np.nanpercentile(wind_power_raw, 99)
    df["wind_power_proxy"] = wind_power_raw.clip(upper=upper_clip)


def _add_holiday_flags(df: pd.DataFrame, cfg: Dict) -> None:
    """
    Add is_public_holiday, is_weekend, is_special_day flags in-place.

    Expects cfg['holidays_csv'] to point to CSV from 00_make_holidays.py
    with a 'date' column (either tz-aware UTC or naive).
    """
    holidays_csv = cfg.get("holidays_csv")
    if not isinstance(holidays_csv, str):
        _fail(
            "Config key 'holidays_csv' must be present and a string in base.yaml. "
            "Run 00_make_holidays.py first and set e.g.: "
            "holidays_csv: data/AT_holidays_2017_2022.csv"
        )

    if not os.path.exists(holidays_csv):
        _fail(
            f"Holidays CSV not found at '{holidays_csv}'. "
            "Run 'make holidays' or 00_make_holidays.py first."
        )

    try:
        hol = pd.read_csv(holidays_csv, parse_dates=["date"])
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read holidays CSV '{holidays_csv}': {exc}")

    if "date" not in hol.columns:
        _fail(
            "Holidays CSV must contain a 'date' column. "
            f"Available columns: {list(hol.columns)}"
        )

    # Normalize holiday dates:
    # - If tz-aware → convert to UTC, normalize to midnight, drop tz.
    # - If tz-naive → just normalize.
    if hasattr(hol["date"].dtype, "tz") and hol["date"].dt.tz is not None:
        hol["date"] = (
            hol["date"]
            .dt.tz_convert("UTC")
            .dt.normalize()
            .dt.tz_localize(None)
        )
    else:
        hol["date"] = hol["date"].dt.normalize()

    idx_dates = df.index.normalize()
    df["is_public_holiday"] = idx_dates.isin(hol["date"]).astype(int)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["is_special_day"] = (
        (df["is_public_holiday"] == 1) | (df["is_weekend"] == 1)
    ).astype(int)


# ---------------------------- main ----------------------------


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)

    # Resolve paths and date range
    start_ts, end_ts = _resolve_date_range(cfg, args.start_date, args.end_date)
    (
        raw_hourly_csv,
        capacity_csv,
        engineered_csv,
        engineered_full_csv,
    ) = _resolve_paths(cfg, args)

    print(f"[INFO] Using raw hourly CSV: {raw_hourly_csv}")
    print(f"[INFO] Using capacity CSV: {capacity_csv}")
    print(f"[INFO] Date range: {start_ts} → {end_ts}")

    # Load raw data
    raw = _load_raw_hourly(raw_hourly_csv, start_ts, end_ts)

    # Load capacity and derive solar/wind capacities per year
    cap_melt = _load_capacity(capacity_csv)
    solar_cap = _capacity_series(cap_melt, "solar")
    wind_cap = _capacity_series(cap_melt, "wind")

    # Build base dataframe
    df = raw.copy()

    # Map annual capacities to hourly index
    df["Solar_Cap_MW"] = annual_to_hourly(solar_cap, df.index)
    df["Wind_Cap_MW"] = annual_to_hourly(wind_cap, df.index)

    # Capacity factors (clip to [0, 1.25] as before)
    df["CF_Solar"] = (df["Solar_MW"] / df["Solar_Cap_MW"]).clip(0.0, 1.25)
    df["CF_Wind"] = (df["Wind_MW"] / df["Wind_Cap_MW"]).clip(0.0, 1.25)

    # Calendar features
    _add_calendar_features(df)

    # Meteorological derived features
    _add_meteo_features(df)

    # Rename price column for consistency
    if "DA_Price_EUR_MWh" not in df.columns:
        _fail(
            "Expected 'DA_Price_EUR_MWh' column in raw data. "
            f"Available columns: {list(df.columns)}"
        )
    df.rename(columns={"DA_Price_EUR_MWh": "Price_EUR_MWh"}, inplace=True)

    # Holidays & weekend flags
    _add_holiday_flags(df, cfg)

    # Ensure output directories and save
    _ensure_parent_dir(engineered_csv)
    _ensure_parent_dir(engineered_full_csv)

    try:
        df.to_csv(engineered_csv)
        df.to_csv(engineered_full_csv)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to write engineered CSV(s): {exc}")

    print(
        f"[INFO] Saved engineered CSVs to '{engineered_csv}' "
        f"and '{engineered_full_csv}' with {len(df)} rows."
    )


if __name__ == "__main__":
    main()
