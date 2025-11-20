# src/00_make_holidays.py

import os
import yaml
import pandas as pd
import holidays


def main():
    # Load base config
    with open("configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Generate Austrian public holidays for dataset range (2017–2022)
    years = range(2017, 2023)  # inclusive of 2022
    at_holidays = holidays.country_holidays("AT", years=years)

    rows = []
    for d, name in sorted(at_holidays.items()):
        ts = pd.Timestamp(d).tz_localize("UTC")
        rows.append({"date": ts, "holiday_name": name})

    out = (
        pd.DataFrame(rows)
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Ensure data directory exists
    os.makedirs(cfg["data_path"], exist_ok=True)

    # Save to configured CSV path
    out.to_csv(cfg["holidays_csv"], index=False)
    print(f"Saved {cfg['holidays_csv']} with {len(out)} rows")


if __name__ == "__main__":
    main()
