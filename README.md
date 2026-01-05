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
- **dCeNN (Discrete Cellular Neural Network)** encoder (PyTorch, training-time only)
- **ELM (Extreme Learning Machine)** regression heads using closed-form ridge regression
- Optimized for fast training and lightweight inference

### Phase 2 – Symbolic Reasoning (The “Filter”)
- **Answer Set Programming (ASP)** using **Clingo**
- Enforces physical plausibility and market logic
- Filters statistically valid but logically impossible anomalies

### Phase 3 – Financial Utility Mapping
- Refined anomaly signals evaluated via backtesting
- Outputs actionable profit/loss utilities for market decisions

---

## Data Sources (Austria, 2017–2022)

- **ENTSO-E**: Day-ahead prices, system load, wind & solar generation
- **Open-Meteo**: Radiation, wind speed (100m), air density
- **Engineered proxies**: Physics-informed PV and wind power features

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
```

---

## Requirements

### Python Version
- **Python 3.8+** (recommended: 3.10)

### Core Python Dependencies
```text
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
holidays>=0.53
pyyaml>=6.0
tqdm>=4.66
clingo>=5.8.0
cffi>=2.0.0
```

### Library Usage Overview

- **Python**: Primary language for data engineering, modeling, and orchestration
- **PyTorch**: Used *only during training* for the dCeNN encoder (edge inference is NumPy-only)
- **Clingo (Potassco)**: ASP solver for symbolic reasoning (`pip install clingo`)
- **Scikit-Learn**: Data normalization (StandardScaler)
- **NumPy & Pandas**: Matrix operations and time-series handling
- **Holidays**: Generates Austrian public holiday facts (`00_make_holidays.py`)

---

## Reproducing the Results

Run the full pipeline:
```bash
make all
```

Key steps:
```bash
make train
make asp
make finance
make plot_event_table_all
```

---

## Edge Deployment

- Exported as `model_bundle.npz`
- 72-hour ring buffer for temporal features
- Zero ML dependencies at inference (NumPy-only)
- Optimized for Raspberry Pi & NVIDIA Jetson

---

## Method Summary

| Component | Role | Logic Type |
|--------|------|-----------|
| dCeNN | Feature extraction | Neural |
| ELM | Fast regression | Linear |
| Conformal Prediction | Thresholding | Statistical |
| ASP (Clingo) | Rule enforcement | Symbolic |
| Finance Mapping | Market utility | Economic |

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
