import os, subprocess, tempfile, pandas as pd, numpy as np

ANOM_CSV = "reports/tables/anomalies_2022.csv"
RULES_LP = "src/07_asp_rules.lp"
OUT_CSV  = "reports/tables/anomalies_with_asp_2022.csv"

def main():
    if not os.path.exists(ANOM_CSV):
        raise FileNotFoundError("Run make detect first.")
    df = pd.read_csv(ANOM_CSV, parse_dates=["Time (UTC)"]).set_index("Time (UTC)").sort_index()

    # We'll emit a single facts file for all hours (T index as integer counter)
    df = df.reset_index()
    df["T"] = np.arange(len(df), dtype=int)

    # Minimal facts: anomalies per target, abs residuals, predicted values
    tgt = ["CF_Solar","CF_Wind","Load_MW","Price"]
    fact_lines = []
    for _, row in df.iterrows():
        T = int(row["T"])
        fact_lines.append(f"hour({T}).")
        # bucket holiday
        h = int(row.get("is_public_holiday", 0)) if "is_public_holiday" in row else 0
        fact_lines.append(f"bucket(holiday,{h},{T}).")
        for name in tgt:
            if f"{name}_pred" in df.columns:
                val = float(row[f"{name}_pred"])
                fact_lines.append(f"pred({name},{val},{T}).")
            if f"{name}_abs" in df.columns:
                ab  = float(row[f"{name}_abs"])
                fact_lines.append(f"resid({name},{ab},{T}).")
            if f"{name}_anom" in df.columns and int(row[f"{name}_anom"])==1:
                fact_lines.append(f"anom({name},{T}).")

        # Derived for rules (optional; placeholders if not present)
        # solar elevation proxy not available -> skip night rule unless added
        # ramp via dLoad_pred:
        if "Load_MW_pred" in df.columns:
            # difference proxy
            fact_lines.append(f"pred(dLoad_pred,{float(row['Load_MW_pred']) - float(df.loc[max(T-1,0),'Load_MW_pred']) if T>0 else 0.0},{T}).")

    facts = "\n".join(fact_lines) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lp", delete=False) as tf:
        tf.write(facts)
        facts_path = tf.name

    # Run clingo
    cmd = ["clingo", RULES_LP, facts_path, "--outf=2", "--quiet=1"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError("clingo failed")

    # Basic parse: look for "final(Name,T)" atoms in the JSON output
    import json
    out = json.loads(res.stdout)
    finals = set()
    for call in out.get("Call", []):
        for w in call.get("Witnesses", []):
            for val in w.get("Value", []):
                if val.startswith("final(") and val.endswith(")"):
                    body = val[len("final("):-1]
                    name, t = body.split(",")
                    finals.add((name, int(t)))

    # Merge back
    for name in tgt:
        df[f"{name}_finalASP"] = 0
    for name,t in finals:
        col = f"{name}_finalASP"
        if col in df.columns:
            df.loc[df["T"]==t, col] = 1

    os.makedirs("reports/tables", exist_ok=True)
    df.drop(columns=["T"]).to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

if __name__ == "__main__":
    main()
