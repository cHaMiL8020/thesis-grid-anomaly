# src/07_apply_asp.py

#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Tuple
import clingo
import numpy as np
import pandas as pd
import yaml

# Default IO paths
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

# ---------------------------- Core ASP Logic ----------------------------

def _generate_facts(df_anom: pd.DataFrame, df_eng: pd.DataFrame, target_map: Dict[str, str]) -> str:
    facts = []
    # Build a load reference for holiday logic
    idx = df_eng.index
    how = idx.dayofweek * 24 + idx.hour
    
    # Corrected ffill() to avoid FutureWarning
    load_ref = pd.DataFrame({"how": how, "load": df_eng["Actual_Load_MW"]}).groupby("how")["load"].median().reindex(how).ffill().set_axis(idx)
    
    for ts, row in df_anom.iterrows():
        t = to_hour_index(ts)
        facts.append(f"hour({t}).")
        
        # 1. Statistical Anomaly Facts - USE LOWERCASE ASP NAME
        for disp_name, asp_name in target_map.items():
            if row.get(f"{disp_name}_anom", 0) == 1:
                facts.append(f'anomaly("{asp_name}", {t}).')

        # 2. Calendar facts
        if "is_public_holiday" in df_eng.columns and df_eng.loc[ts, "is_public_holiday"] == 1:
            facts.append(f"holiday({t}).")

        # 3. Physics & Market facts (Rounded to integers)
        if "shortwave_radiation (W/m²)" in df_eng.columns:
            rad = int(round(df_eng.loc[ts, "shortwave_radiation (W/m²)"]))
            facts.append(f"pred(shortwave_radiation, {rad}, {t}).")
            
        if "Wind_MW" in df_eng.columns:
            w_mw = int(round(df_eng.loc[ts, "Wind_MW"]))
            facts.append(f"pred(wind_mw, {w_mw}, {t}).")

        if "Actual_Load_MW" in df_eng.columns:
            l_mw = int(round(df_eng.loc[ts, "Actual_Load_MW"]))
            r_mw = int(round(load_ref.loc[ts]))
            facts.append(f"pred(load_mw, {l_mw}, {t}).")
            facts.append(f"pred(load_ref, {r_mw}, {t}).")

    return "\n".join(facts)

def _run_clingo(facts_program: str, rules_path: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Runs the Clingo solver and parses both valid and vetoed atoms."""
    ctl = clingo.Control()
    ctl.add("base", [], facts_program)
    ctl.load(rules_path)
    ctl.ground([("base", [])])
    
    valid = []
    vetoed = []

    def on_model(m):
        for atom in m.symbols(shown=True):
            # Check name first to ensure we only parse atoms where argument 1 is a number
            if atom.name in ["valid_anomaly", "vetoed"] and len(atom.arguments) == 2:
                target = str(atom.arguments[0]).strip('"')
                # h_idx is the second argument in these specific atoms
                h_idx = int(atom.arguments[1].number)
                if atom.name == "valid_anomaly":
                    valid.append((target, h_idx))
                elif atom.name == "vetoed":
                    vetoed.append((target, h_idx))
    
    ctl.solve(on_model=on_model)
    return valid, vetoed

def main() -> None:
    args = _parse_args()
    base_cfg = _load_yaml(args.base_config)
    
    # Load inputs 
    df_eng = pd.read_csv(base_cfg["engineered_csv"], parse_dates=["Time (UTC)"]).set_index("Time (UTC)")
    df_anom = pd.read_csv(args.anoms_csv, parse_dates=["Time (UTC)"]).set_index("Time (UTC)")
    
    # Map ML targets (e.g., 'Price_anom') to lowercase ASP names ('price')
    anom_cols = [c for c in df_anom.columns if c.endswith("_anom")]
    target_names = {c[:-5]: c[:-5].lower() for c in anom_cols}
    
    # 1. Generate Facts 
    facts = _generate_facts(df_anom, df_eng, target_names)
    
    # 2. Reasoning Fusion (Capture both lists)
    valid_pairs, veto_pairs = _run_clingo(facts, args.rules_path)
    
    # 3. Build Result Rows
    refined_rows = []
    # Helper to match hour index back to timestamp
    time_map = {int(ts.timestamp() // 3600): ts for ts in df_anom.index}

    # Add confirmed anomalies
    for target, h_idx in valid_pairs:
        ts = time_map.get(h_idx)
        if ts:
            refined_rows.append({
                "target": target, "Time (UTC)": ts, "hour_index": h_idx,
                "anomaly_score": df_anom.loc[ts, "combined_score"] if "combined_score" in df_anom.columns else 0,
                "final_flag": 1, "is_vetoed": 0
            })

    # Add vetoed anomalies (rejected by logic)
    for target, h_idx in veto_pairs:
        ts = time_map.get(h_idx)
        if ts:
            refined_rows.append({
                "target": target, "Time (UTC)": ts, "hour_index": h_idx,
                "anomaly_score": df_anom.loc[ts, "combined_score"] if "combined_score" in df_anom.columns else 0,
                "final_flag": 0, "is_vetoed": 1
            })
            
    df_refined = pd.DataFrame(refined_rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_refined.to_csv(args.out_csv, index=False)
    print(f"[INFO] ASP Reasoning complete. Confirmed: {len(valid_pairs)}, Vetoed: {len(veto_pairs)}")

if __name__ == "__main__":
    main()