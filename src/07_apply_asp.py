# src/07_apply_asp.py
import os, glob
import re
import yaml
import pandas as pd
import tempfile
import clingo  # Python API (works in Codespaces)

# ----------------- Configs -----------------
base = yaml.safe_load(open("configs/base.yaml"))
ANOM_CSV = "reports/tables/anomalies_2022.csv"
if not os.path.exists(ANOM_CSV):
    cands = sorted(glob.glob("reports/tables/anomalies_*.csv"))
    if cands:
        ANOM_CSV = cands[-1]  # pick most recent anomalies file
    else:
        raise FileNotFoundError(
            "Missing anomalies CSV. Run:\n"
            "  make train\n  make thresholds\n  make detect\n"
        )
RULES    = "rules/market_rules.lp"

# ----------------- Helpers -----------------
def to_hour_index(ts: pd.Timestamp) -> int:
    ts = pd.to_datetime(ts, utc=True)
    return int(ts.timestamp() // 3600)

def hours_of_utc_day(ts: pd.Timestamp):
    ts = pd.to_datetime(ts, utc=True)
    start = pd.Timestamp(year=ts.year, month=ts.month, day=ts.day, tz="UTC")
    return [to_hour_index(start + pd.Timedelta(hours=h)) for h in range(24)]

# ----------------- Read holidays (robust) -----------------
holidays_path = base["holidays_csv"]
hol = pd.read_csv(holidays_path)
date_cols = [c for c in hol.columns if "date" in c.lower() or "time" in c.lower()]
if not date_cols:
    raise ValueError(f"No date/time column in {holidays_path}. Columns={list(hol.columns)}")
date_col = date_cols[0]
hol[date_col] = pd.to_datetime(hol[date_col], utc=True, errors="coerce")
hol = hol.dropna(subset=[date_col]).copy()

holiday_hours = set()
for ts in hol[date_col]:
    holiday_hours.update(hours_of_utc_day(ts))

# ----------------- Read anomalies -----------------
df = pd.read_csv(ANOM_CSV, parse_dates=["Time (UTC)"]).sort_values("Time (UTC)")
targets = ["CF_Solar","CF_Wind","Load_MW","Price"]

# ----------------- Build ASP facts (integers only) -----------------
facts = []

for _, r in df.iterrows():
    h = to_hour_index(r["Time (UTC)"])
    for t in targets:
        flag_col  = f"{t}_anom"
        if flag_col in r and pd.notna(r[flag_col]) and int(r[flag_col]) == 1:
            facts.append(f'anomaly("{t}", {h}).')


# holiday facts (24 hours per holiday day)
for h in sorted(holiday_hours):
    facts.append(f'anomaly("{t}", {h}).')

asp_input = "\n".join(facts)

# ----------------- Ensure rules exist -----------------
if not os.path.exists(RULES):
    os.makedirs("rules", exist_ok=True)
    with open(RULES, "w") as f:
        f.write("""\
% Input facts:
%   anomaly(Target, HourIndex, MagnitudeInt).
%   holiday(HourIndex).
%
% Output:
%   valid_anomaly(Target, HourIndex).

% Rule 1: Persistence >= 3 consecutive hours
persistent(T, G) :-
    anomaly(G, T, _), anomaly(G, T-1, _), anomaly(G, T-2, _).

% Rule 2: Ignore anomalies on public holidays
valid_anomaly(G, T) :- persistent(T, G), not holiday(T).

#show valid_anomaly/2.
""")

# ----------------- Solve with Clingo (Python API) -----------------
print("Running Clingo reasoning via Python API ...")

with tempfile.NamedTemporaryFile("w", delete=False) as tmp_f:
    tmp_f.write(asp_input)
    facts_path = tmp_f.name

ctl = clingo.Control()
ctl.load(facts_path)
ctl.load(RULES)
ctl.ground([("base", [])])

models = []
def on_model(m): models.append(str(m))
ctl.solve(on_model=on_model)

os.remove(facts_path)

out = "\n".join(models)

# ----------------- Parse clingo output -----------------
pat = re.compile(r'valid_anomaly\("(?P<target>[^"]+)",\s*(?P<hour>\d+)\)')
records = []
for line in out.splitlines():
    for m in pat.finditer(line):
        records.append((m.group("target"), int(m.group("hour"))))

ref = pd.DataFrame(records, columns=["target","hour_index"])
if not ref.empty:
    ref["Time (UTC)"] = pd.to_datetime(ref["hour_index"] * 3600, unit="s", utc=True)
    ref = ref.sort_values(["Time (UTC)","target"])

os.makedirs("reports/tables", exist_ok=True)
ref.to_csv("reports/tables/anomalies_refined.csv", index=False)
print(f"Saved refined anomalies → reports/tables/anomalies_refined.csv ({len(ref)} rows)")
