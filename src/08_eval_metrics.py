import os, pandas as pd, numpy as np

IN_CSV  = "reports/tables/anomalies_2022.csv"
OUT_SUM = "reports/tables/metrics_summary.csv"
OUT_MON = "reports/tables/metrics_monthly.csv"

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def mae(a,b):  return float(np.mean(np.abs(a-b)))

def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError("Run make detect first.")
    df = pd.read_csv(IN_CSV, parse_dates=["Time (UTC)"]).set_index("Time (UTC)").sort_index()
    targets = ["CF_Solar","CF_Wind","Load_MW","Price"]

    rows = []
    for t in targets:
        true = df[f"{t}_true"].values
        pred = df[f"{t}_pred"].values
        rows.append({"target": t, "RMSE": rmse(true,pred), "MAE": mae(true,pred),
                    "anomaly_count": int(df.get(f"{t}_anom", pd.Series(0,index=df.index)).sum())})
    summ = pd.DataFrame(rows)
    os.makedirs("reports/tables", exist_ok=True)
    summ.to_csv(OUT_SUM, index=False)

    mons = []
    for (y,m), g in df.groupby([df.index.year, df.index.month]):
        for t in targets:
            mons.append({
                "year": y, "month": m, "target": t,
                "RMSE": rmse(g[f"{t}_true"], g[f"{t}_pred"]),
                "MAE":  mae(g[f"{t}_true"], g[f"{t}_pred"]),
                "anom": int(g.get(f"{t}_anom", pd.Series(0,index=g.index)).sum())
            })
    pd.DataFrame(mons).to_csv(OUT_MON, index=False)
    print(f"Saved {OUT_SUM} and {OUT_MON}")

if __name__ == "__main__":
    main()
