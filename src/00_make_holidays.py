# src/00_make_holidays.py

#!/usr/bin/env python3
"""
Generate public holiday calendar for Austria (or another country) and
store it as a CSV, based on the global config.

Default behaviour:
- Reads configs/base.yaml
- Uses years 2017–2022 (inclusive), unless overridden by config or CLI
- Writes a holidays CSV to cfg["holidays_csv"]
- Ensures cfg["data_path"] exists

CLI overrides:
    python src/00_make_holidays.py \
        --config configs/base.yaml \
        --country AT \
        --start-year 2017 \
        --end-year 2022

If something is wrong (missing config keys, unreadable file, etc.), the script
fails with a clear error message instead of a cryptic traceback.
"""

import argparse
import os
from typing import Dict, Iterable, List

import holidays
import pandas as pd
import yaml


DEFAULT_CONFIG_PATH = "configs/base.yaml"
DEFAULT_COUNTRY = "AT"
DEFAULT_START_YEAR = 2017
DEFAULT_END_YEAR = 2022


# -------------------------- helpers --------------------------


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
        description="Generate public holidays CSV based on configs/base.yaml."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to base config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--country",
        default=DEFAULT_COUNTRY,
        help=f"Country code for holidays package (default: {DEFAULT_COUNTRY}).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Start year for holidays (default: from config or 2017).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year for holidays (default: from config or 2022).",
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


def _resolve_years(cfg: Dict, start_year_cli: int | None, end_year_cli: int | None) -> List[int]:
    """
    Determine which years to generate holidays for, following this priority:

    1) CLI overrides (--start-year / --end-year) if provided
    2) Config keys cfg['holidays_start_year'], cfg['holidays_end_year'] if present
    3) Fallback to DEFAULT_START_YEAR..DEFAULT_END_YEAR
    """
    if start_year_cli is not None and end_year_cli is not None:
        start_year = start_year_cli
        end_year = end_year_cli
    else:
        # Try config keys if CLI not fully specified
        start_year = cfg.get("holidays_start_year", DEFAULT_START_YEAR)
        end_year = cfg.get("holidays_end_year", DEFAULT_END_YEAR)

    if not isinstance(start_year, int) or not isinstance(end_year, int):
        _fail(
            f"Invalid holidays year range in config/CLI: start_year={start_year}, "
            f"end_year={end_year}"
        )
    if end_year < start_year:
        _fail(
            f"Invalid holidays year range: end_year ({end_year}) < start_year ({start_year})."
        )

    years = list(range(start_year, end_year + 1))
    return years


def _generate_holidays(country: str, years: Iterable[int]) -> pd.DataFrame:
    """
    Generate a DataFrame with UTC timestamps and holiday names for the given
    country and years.
    """
    try:
        country_holidays = holidays.country_holidays(country, years=years)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to generate holidays for country '{country}': {exc}")

    rows = []
    for d, name in sorted(country_holidays.items()):
        ts = pd.Timestamp(d).tz_localize("UTC")
        rows.append({"date": ts, "holiday_name": str(name)})

    if not rows:
        _warn(
            f"No holidays returned for country='{country}' and years={list(years)}. "
            "Output will be an empty CSV."
        )
        return pd.DataFrame(columns=["date", "holiday_name"])

    out = (
        pd.DataFrame(rows)
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


def _ensure_output_dirs(cfg: Dict, path_from_config: str) -> None:
    """
    Ensure that both cfg['data_path'] and the directory of the output file exist.
    """
    data_path = cfg.get("data_path")
    if not isinstance(data_path, str):
        _fail(
            "Config key 'data_path' is missing or not a string in base.yaml. "
            "Please add e.g.: data_path: data"
        )

    try:
        os.makedirs(data_path, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to create data_path directory '{data_path}': {exc}")

    out_dir = os.path.dirname(path_from_config) or "."
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to create output directory '{out_dir}': {exc}")


# -------------------------- main --------------------------


def main() -> None:
    args = _parse_args()

    cfg = _load_config(args.config)

    # Resolve years
    years = _resolve_years(cfg, args.start_year, args.end_year)

    # Determine output CSV path from config
    holidays_csv = cfg.get("holidays_csv")
    if not isinstance(holidays_csv, str):
        _fail(
            "Config key 'holidays_csv' is missing or not a string in base.yaml. "
            "Please add e.g.: holidays_csv: data/AT_holidays_2017_2022.csv"
        )

    # Ensure output directories exist
    _ensure_output_dirs(cfg, holidays_csv)

    # Generate holiday DataFrame
    df_holidays = _generate_holidays(args.country, years)

    # Save
    try:
        df_holidays.to_csv(holidays_csv, index=False)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to write holidays CSV to '{holidays_csv}': {exc}")

    print(
        f"[INFO] Saved {holidays_csv} with {len(df_holidays)} rows "
        f"for country={args.country}, years={years[0]}–{years[-1]}"
    )


if __name__ == "__main__":
    main()
