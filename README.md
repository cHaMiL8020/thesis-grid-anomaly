# Finance-Aware, Weather-Informed Anomaly Detection for Electricity Markets & Grid Operations  
### Combining Learning (dCeNN–ELM) and Reasoning (ASP) for Multivariate Time Series Under Uncertainty

This repository contains the current implementation for my master’s thesis:

> **Finance-Aware, Weather-Informed Anomaly Detection for Electricity Markets & Grid Operations: Combining Learning and Reasoning for Multivariate Time Series Under Uncertainty Using an Efficient dCeNN–ELM–ASP Approach**

This project develops a **lightweight, edge-ready anomaly detection system** that combines:

- **Learning** → Tiny dCeNN encoder + ELM regression heads  
- **Reasoning** → Answer Set Programming (Clingo)  
- **Finance-awareness** → Utility-based anomaly scoring  
- **Edge deployment** → Compact NDArray runtime for Raspberry Pi / Jetson  

The pipeline is **100% reproducible** using the Makefile and runs seamlessly in GitHub Codespaces or Ubuntu.

---

# 1. Project Structure
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
│  ├─ raw/ (source CSVs)
│  ├─ AT_engineered_hourly.csv
│  ├─ AT_engineered_hourly_FULL.csv
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
├─ rules/
│  ├─ market_rules.lp
│
├─ artifacts/      # Models, scaler, thresholds (generated)
│
├─ reports/
│  ├─ tables/      # CSV outputs
│  ├─ figures/     # PNG plots
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

# 2. Research Motivation

Electricity markets are increasingly complex due to:

- High renewable penetration  
- Weather-driven uncertainty  
- Extreme events (European gas crisis 2021–2022)  
- Grid flexibility and resilience requirements  

Traditional anomaly detection fails when data is **non-stationary, multivariate, noisy, and context-dependent**.

This thesis proposes a **hybrid ML + Logic Reasoning approach** that:

 Learns patterns → **dCeNN–ELM**  
 Detects anomalies → **Conformal thresholds**  
 Filters them → **ASP rules**  
 Maps anomaly behaviour to financial action → **Utility model**  
 Deploys on hardware → **Edge bundle**

---

# 3. Data Sources (Austria 2017–2022)

- ENTSO-E load, wind, solar, electricity prices  
- Open-Meteo weather (temperature, radiation, wind, humidity, pressure)  
- Installed capacity data (for CF conversion)  
- Calendar: holidays, weekends, seasons  
- Physics-based proxies (wind power, PV power, air density)

---

# 4. Pipeline Overview

## **Phase 1 — Learning**
1. Feature engineering + physic proxies  
2. Construct supervised dataset  
3. Standardize features  
4. Train **Tiny dCeNN encoder**  
5. Train **ELM heads**  
6. Conformal residual calibration  
7. Anomaly scoring  
8. Finance-aware mapping  

## **Phase 2 — Reasoning**
- Holiday filtering  
- Persistence rules (≥3-hour anomalies)  
- Logical pruning of insignificant anomalies  
- Output: **refined anomalies**

## **Phase 3 — Edge Deployment**
- Bundle encoder weights, ELM weights, scaler, mask  
- Portable inference script for Pi/Jetson

---

# 5. Environment Setup

### Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate

sudo apt-get update
sudo apt-get install -y make

pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install clingo
```

---

# 6. Running the Pipeline

### Run EVERYTHING:
```bash
make all
```

### Run individual steps:
```bash
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

# 7. Output Artifacts

### **Tables**
- anomalies_2022.csv  
- anomalies_refined.csv  
- metrics_summary.csv  
- metrics_monthly.csv  
- finance_backtest_2022.csv  

### **Figures**
- residual_hist_val.png  
- anomaly_combined_2022.png  
- utility_vs_time_2022.png  

### **Models**
- dcenn_encoder.pt  
- elm_heads.npz  
- scaler.npz  
- thresholds.json  
- model_bundle.npz  

---

# 8. Key Results

### **Residual Distribution**
- Residuals are near-Gaussian and stable → ideal for conformal thresholds  
- Price residuals are very sharp → accurate predictions  

### **Combined Anomaly Score**
- Captures real 2022 regime shifts (Feb–Mar crisis, summer instability)  
- Seasonal patterns visible  
- No drift, no false saturation  

### **Finance Mapping**
- Strategy significantly outperforms passive baseline  
- Gains align with real volatility periods  
- Demonstrates *economic value* of anomaly detection  

---

# 9. Method Summary

| Component | Description |
|----------|-------------|
| dCeNN Encoder | Lightweight masked linear iterative network (Chua CNN-inspired) |
| ELM Heads | Randomized fast regression for multi-output forecasting |
| Conformal Thresholds | Distribution-free anomaly calibration |
| ASP Layer | Logical consistency, holiday filtering, persistence rules |
| Finance Mapping | Converts anomalies into actionable market utility |
| Edge Bundle | Exportable inference module for low-power devices |

---

# 10. Future Work
- Benchmark vs GRU, LSTM, TCN, XGBoost  
- Multi-horizon forecasting  
- Extreme weather ASP rules  
- Multi-country generalization  
- Edge latency & power profiling  

---

# 11. Citation

```
Abeysekara, C.O. (2025). Finance-Aware, Weather-Informed Anomaly Detection 
for Electricity Markets & Grid Operations (dCeNN–ELM–ASP). 
GitHub: https://github.com/cHaMiL8020/thesis-grid-anomaly
```

---

# 12. Contact  
**Chamil Oshan Abeysekara**  
MSc Autonomous Systems & Robotics  
University of Klagenfurt (2025)
