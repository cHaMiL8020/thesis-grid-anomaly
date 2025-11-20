# src/08_eval_metrics.py (or 08_eval.py)

import os
import pandas as pd
import numpy as np

IN_CSV  = "reports/tables/anomalies_2022.csv"
OUT_SUM = "reports/tables/metrics_summary.csv"
OUT_MON = "reports/tables/metrics_monthly.csv"

def rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))

def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError("Run `make detect` (Step 05) first to create anomalies CSV.")

    df = (
        pd.read_csv(IN_CSV, parse_dates=["Time (UTC)"])
          .set_index("Time (UTC)")
          .sort_index()
    )

    targets = ["CF_Solar", "CF_Wind", "Load_MW", "Price"]

    # ---------- Global metrics ----------
    rows = []
    for t in targets:
        true = df[f"{t}_true"]
        pred = df[f"{t}_pred"]
        anom_col = f"{t}_anom"
        anom_count = int(df.get(anom_col, pd.Series(0, index=df.index)).sum())

        rows.append({
            "target": t,
            "RMSE": rmse(true, pred),
            "MAE": mae(true, pred),
            "anomaly_count": anom_count,
        })

    summ = pd.DataFrame(rows)
    os.makedirs("reports/tables", exist_ok=True)
    summ.to_csv(OUT_SUM, index=False)

    # ---------- Monthly metrics ----------
    mons = []
    for (y, m), g in df.groupby([df.index.year, df.index.month]):
        for t in targets:
            true = g[f"{t}_true"]
            pred = g[f"{t}_pred"]
            anom_col = f"{t}_anom"
            anom_count = int(g.get(anom_col, pd.Series(0, index=g.index)).sum())

            mons.append({
                "year": int(y),
                "month": int(m),
                "target": t,
                "RMSE": rmse(true, pred),
                "MAE": mae(true, pred),
                "anom": anom_count,
            })

    pd.DataFrame(mons).to_csv(OUT_MON, index=False)
    print(f"Saved {OUT_SUM} and {OUT_MON}")

if __name__ == "__main__":
    main()
