# Finance-Aware, Weather-Informed Anomaly Detection for Electricity Markets & Grid Operations

**Combining Learning (dCeNN–ELM) and Reasoning (ASP) for Multivariate Time-Series under Uncertainty**

This repository contains the full implementation of my **Master’s thesis** in *Autonomous Systems and Robotics* at the **University of Klagenfurt, Austria**.

The project presents a **Neuro-Symbolic Anomaly Detection framework** for electricity markets and grid operations, combining efficient neural learning with rule-based symbolic reasoning to deliver **physically valid, economically meaningful anomaly signals**.

---

## Core Idea

Traditional machine learning models often detect *statistical anomalies* that are **physically impossible or economically irrelevant**.

This work addresses that gap by integrating:
- **Learning (Neural Layer)** – detects statistical deviations  
- **Reasoning (Symbolic Layer)** – enforces physics & market constraints  
- **Finance Awareness** – evaluates real economic utility  
- **Edge Readiness** – deployable on low-power hardware  

---

## System Architecture

### Phase 1 – Neural Learning (The “Brain”)
- **dCeNN (Discrete Cellular Neural Network)** encoder  
  - Captures spatio-temporal dependencies with local connectivity  
  - Lightweight and parameter-efficient  
- **ELM (Extreme Learning Machine)** regression heads  
  - Closed-form ridge regression  
  - Ultra-fast training and inference  

Forecasted variables include electricity price, system load, and renewable generation.

---

### Phase 2 – Symbolic Reasoning (The “Filter”)
- **Answer Set Programming (ASP)** using **Clingo**
- Injects domain knowledge to remove false positives

Example rules:
- Solar anomalies vetoed when radiation = 0 (night-time)
- Wind ramps exceeding physical limits filtered
- Price spikes confirmed only with co-occurring load or generation shifts

---

### Phase 3 – Financial Utility Mapping
- Refined anomaly signals are evaluated in a market backtest
- Converts detections into **profit/loss utility**
- Demonstrates reduced cost of false alarms

---

## Data Sources (Austria, 2017–2022)
- ENTSO-E electricity prices, load, wind and solar generation
- Open-Meteo weather data
- Physics-informed engineered proxies

---

## Project Structure

```text
thesis-grid-anomaly/
│
├─ configs/        # YAML configs (features, thresholds, models)
├─ data/           # Raw & engineered datasets
├─ src/            # Modular pipeline (00–11)
│  ├─ 00–02  Data engineering & preprocessing
│  ├─ 03–05  Neural training & statistical detection
│  ├─ 06     Finance-aware utility mapping
│  ├─ 07     ASP symbolic reasoning
│  ├─ 08–09  Metrics & edge export
│  └─ 10–11  Event clustering & master visualizations
│
├─ rules/          # ASP rules (Clingo)
├─ artifacts/      # Trained models, scalers, thresholds
├─ reports/        # CSV outputs & plots
├─ edge/           # Edge-ready inference bundle
├─ Makefile        # End-to-end orchestration
└─ README.md 

---


## Edge Deployment
- Pure NumPy inference
- 72-hour ring buffer for temporal features
- Optimized for Raspberry Pi / NVIDIA Jetson

---

## Method Summary

| Component | Role | Logic |
|--------|------|------|
| dCeNN | Feature extraction | Neural |
| ELM | Fast regression | Linear |
| ASP | Rule enforcement | Symbolic |
| Finance Mapping | Market utility | Economic |

---

Reproducing the Results

Run the full pipeline:
```bash
make all
```
---

Key steps:
```bash
make all
make train
make asp
make finance
make plot_event_table_all
```

---

## Author

**Chamil Oshan Abeysekara**  
Master’s Candidate – Autonomous Systems & Robotics  
University of Klagenfurt, Austria

---

## Citation

```bibtex
@thesis{abeysekara2025grid,
  title  = {Finance-Aware, Weather-Informed Anomaly Detection for Electricity Markets},
  author = {Abeysekara, Chamil Oshan},
  year   = {2025},
  school = {University of Klagenfurt}
}
```
