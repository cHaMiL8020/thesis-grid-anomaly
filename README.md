# Finance-Aware, Weather-Informed Anomaly Detection for Electricity Markets & Grid Operations  
### Combining Learning (dCeNN–ELM) and Reasoning (ASP) for Multivariate Time Series Under Uncertainty

This repository contains the full implementation for my master’s thesis:

> **Finance-Aware, Weather-Informed Anomaly Detection for Electricity Markets & Grid Operations: Combining Learning and Reasoning for Multivariate Time Series Under Uncertainty Using an Efficient dCeNN–ELM–ASP Approach**

The project builds a **lightweight, edge-friendly** anomaly detection system for electricity markets that:

- Learns short-term dependencies in **load, renewable generation, and price** time series.
- Uses **weather, seasons, holidays, and physics-based proxies** as contextual signals.
- Produces anomaly scores using a **dCeNN encoder + ELM forecasting heads** (Phase 1: Learning).
- Refines raw anomalies using **Answer Set Programming (Clingo)** rules (Phase 2: Reasoning).
- Evaluates anomalies through a **finance-aware utility mapping**.
- Exports a compact **edge deployment bundle** for Raspberry Pi / Jetson devices.

The pipeline is **100% reproducible** via a Makefile and runs fully in GitHub Codespaces or Ubuntu.

---

## 1. Project Structure
```
thesis-grid-anomaly/
│
├─ configs/
│  ├─ base.yaml
│  ├─ features.yaml
│  ├─ dcenn.yaml
│  ├─ thresholds.yaml
│
├─ data/
│  ├─ raw CSVs
│  ├─ engineered CSVs (generated)
│
├─ src/
│  ├─ 00_make_holidays.py
│  ├─ 01_preprocess_build_features.py
│  ├─ 02_split_and_scale.py
│  ├─ 03_train_dcenn_elm.py
│  ├─ 04_calibrate_thresholds.py
│  ├─ 05_detect_anomalies.py
│  ├─ 06_finance_mapping.py
│  ├─ 07_apply_asp.py
│  ├─ 08_eval_metrics.py
│  ├─ 09_edge_export.py
│
├─ rules/market_rules.lp
│
├─ artifacts/ (generated)
│
├─ reports/
│  ├─ tables/
│  ├─ figures/
│
├─ edge/
│  ├─ model_bundle.npz
│  ├─ runtime_infer.py
│
├─ Makefile
├─ requirements.txt
└─ README.md
```

---

## 2. Research Motivation
Electricity markets are becoming increasingly volatile due to:
- High penetration of **wind + solar**
- **Weather-driven uncertainty**
- **Market shocks** (2021–22 crisis)
- **Grid congestion** & flexibility needs

This thesis builds a **hybrid learning + reasoning** system to detect anomalies meaningfully under uncertainty.

---

## 3. Data Sources
### Austria (2017–2022)
- ENTSO-E market data: Load, wind, solar, prices
- Open-Meteo weather data: temperature, wind, radiation, humidity
- Installed capacity datasets for CF conversion
- Calendar: holidays, weekends, seasonal encoding
- Derived physics-based features (air density, wind power proxy, PV proxy)

---

## 4. Pipeline Overview
### **Phase 1 — Learning**
1. Preprocessing + feature engineering
2. Supervised dataset creation
3. Standardization
4. Train Tiny dCeNN encoder
5. Fit ELM regression heads
6. Calibrate thresholds (conformal)
7. Detect anomalies
8. Finance mapping

### **Phase 2 — Reasoning (ASP)**
- Persistence rules
- Holiday-aware filtering
- Cross-signal correlation pruning
- Produce refined anomalies

### **Phase 3 — Edge Deployment**
- Export compressed inference bundle
- Minimal Python runtime for Raspberry Pi

---

## 5. Environment Setup
```
python3 -m venv .venv
source .venv/bin/activate

sudo apt-get update
sudo apt-get install -y make

pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install clingo
```

---

## 6. Running the Full Pipeline
### Run everything (Phase 1 + Phase 2)
```
make all
```

### Run specific steps
```
make holidays
make preprocess
make split
make train
make thresholds
make detect
make finance
make asp
make eval
make edge
```

---

## 7. Output Artifacts
### Tables
- anomalies_2022.csv
- anomalies_refined.csv
- metrics_summary.csv
- metrics_monthly.csv
- finance_backtest_2022.csv

### Figures
- residual_hist_val.png
- anomaly_combined_2022.png
- utility_vs_time_2022.png

### Models
- dcenn_encoder.pt
- elm_heads.npz
- thresholds.json
- model_bundle.npz

---

## 8. Method Summary
### dCeNN Encoder
- Lightweight discrete CNN capturing short-range dependencies

### ELM Regression Heads
- Randomized fast linear readout layer for forecasting

### Conformal Thresholds
- Distribution-free calibration using validation residuals

### ASP Reasoning
- Removes irrelevant anomalies using logical rules

### Finance Mapping
- Converts anomalies into market-relevant utility

### Edge Bundle
- Small footprint inference package

---

## 9. Future Work
- Benchmark vs GRU/LSTM/XGBoost
- Add rule-based extreme weather constraints
- Multi-country generalization
- Full edge latency profiling

---

## 10. License
For research and educational use.

---

## 11. Contact
**Chamil Oshan Abeysekara**  
MSc Autonomous Systems & Robotics  
University of Klagenfurt (2025)

