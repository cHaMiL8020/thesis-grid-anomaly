# src/07_apply_asp.py

#!/usr/bin/env python3
"""
Apply ASP rules (rules/07_asp_rules.lp) to refine anomalies.

Inputs:
  - configs/base.yaml (engineered_csv, holidays_csv)
  - reports/tables/anomalies_2022.csv (Raw ML output)
  - rules/market_rules.lp (The domain logic)

Updated Fact Generation:
  - pred(shortwave_radiation, V, T).  # For night-time solar validation
  - pred(wind_mw, V, T).               # For physical ramp-rate checks
  - pred(load_mw, V, T).               # For co-anomaly verification
  - pred(load_ref, R, T).              # Baseline for market logic

Output:
  - reports/tables/anomalies_refined.csv
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
DEFAULT_RULES_PATH = "rules/market_rules.lp"
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
    except Exception as exc:
        _fail(f"Failed to read config YAML '{path}': {exc}")
    return cfg

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply ASP rules to refine anomalies.")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--anoms-csv", default=DEFAULT_ANOM_CSV)
    parser.add_argument("--rules-path", default=DEFAULT_RULES_PATH)
    parser.add_argument("--out-csv", default=DEFAULT_REFINED_CSV)
    return parser.parse_args()

def to_hour_index(ts: pd.Timestamp) -> int:
    """Map a UTC timestamp → integer hour index since Unix epoch."""
    ts = pd.to_datetime(ts, utc=True)
    return int(ts.timestamp() // 3600)

def _build_load_reference(df_engineered: pd.DataFrame) -> pd.Series:
    """Compute climatological median load for hour-of-week."""
    idx = df_engineered.index
    how = idx.dayofweek * 24 + idx.hour
    clim = pd.DataFrame({"how": how, "load": df_engineered["Actual_Load_MW"]}).groupby("how")["load"].median()
    return clim.reindex(how).fillna(method='ffill').set_axis(idx)

# ---------------------------- Core ASP Logic ----------------------------

def _generate_facts(df_anom: pd.DataFrame, df_eng: pd.DataFrame, target_map: Dict[str, str]) -> str:
    """
    Generate logic facts from dataframes for Clingo.
    Explanation: This converts time-series data into 'First-Order Logic' atoms.
    ASP solvers work best with integers, so we round physical values. 
    """
    facts = []
    load_ref = _build_load_reference(df_eng)
    
    for ts, row in df_anom.iterrows():
        t = to_hour_index(ts)
        facts.append(f"hour({t}).")
        
        # 1. Statistical Anomaly Facts (ML Layer)
        for disp_name, asp_name in target_map.items():
            if row.get(f"{disp_name}_anom", 0) == 1:
                facts.append(f'anomaly("{disp_name}", {t}).')

        # 2. Calendar facts
        if "is_public_holiday" in df_eng.columns:
            if df_eng.loc[ts, "is_public_holiday"] == 1:
                facts.append(f"holiday({t}).")

        # 3. Physics & Market facts (Symbolic Layer inputs)
        # Scaled/Rounded to integers for ASP compatibility 
        if "shortwave_radiation (W/m²)" in df_eng.columns:
            rad = int(round(df_eng.loc[ts, "shortwave_radiation (W/m²)"]))
            facts.append(f"pred(shortwave_radiation, {rad}, {t}).")
            
        if "Wind_MW" in df_eng.columns:
            w_mw = int(round(df_eng.loc[ts, "Wind_MW"]))
            facts.append(f"pred(wind_mw, {w_mw}, {t}).")

        if "Actual_Load_MW" in df_eng.columns:
            l_mw = int(round(df_eng.loc[ts, "Actual_Load_MW"]))
            l_ref = int(round(load_ref.loc[ts]))
            facts.append(f"pred(load_mw, {l_mw}, {t}).")
            facts.append(f"pred(load_ref, {l_ref}, {t}).")

    return "\n".join(facts)

def _run_clingo(facts_program: str, rules_path: str) -> List[Tuple[str, int]]:
    """Runs the Clingo solver and parses the 'valid_anomaly' atoms."""
    ctl = clingo.Control()
    ctl.add("base", [], facts_program)
    ctl.load(rules_path)
    ctl.ground([("base", [])])
    
    results = []
    def on_model(m):
        for atom in m.symbols(shown=True):
            if atom.name == "valid_anomaly" and len(atom.arguments) == 2:
                target = str(atom.arguments[0]).strip('"')
                h_idx = int(atom.arguments[1].number)
                results.append((target, h_idx))
    
    ctl.solve(on_model=on_model)
    return results

# ---------------------------- Main ----------------------------

def main() -> None:
    args = _parse_args()
    base_cfg = _load_yaml(args.base_config)
    
    # Load inputs 
    df_eng = pd.read_csv(base_cfg["engineered_csv"], parse_dates=["Time (UTC)"]).set_index("Time (UTC)")
    df_anom = pd.read_csv(args.anoms_csv, parse_dates=["Time (UTC)"]).set_index("Time (UTC)")
    
    # Map ML targets to names used in ASP rules 
    anom_cols = [c for c in df_anom.columns if c.endswith("_anom")]
    target_names = {c[:-5]: c[:-5].lower() for c in anom_cols}
    
    # 1. Generate Facts 
    facts = _generate_facts(df_anom, df_eng, target_names)
    
    # 2. Reasoning Fusion 
    refined_pairs = _run_clingo(facts, args.rules_path)
    
    # 3. Save Refined Table 
    refined_rows = []
    for target, h_idx in refined_pairs:
        # Find original row by hour index
        # Explanation: Re-matching logic atoms back to time-series rows.
        mask = (df_anom.index.view(np.int64) // 3600_000_000_000) == h_idx
        if any(mask):
            ts = df_anom.index[mask][0]
            refined_rows.append({
                "target": target,
                "Time (UTC)": ts,
                "hour_index": h_idx,
                "anomaly_score": df_anom.loc[ts, "combined_score"],
                "final_flag": 1
            })
            
    df_refined = pd.DataFrame(refined_rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_refined.to_csv(args.out_csv, index=False)
    print(f"[INFO] ASP Reasoning complete. Saved {len(df_refined)} refined anomalies.")

if __name__ == "__main__":
    main()