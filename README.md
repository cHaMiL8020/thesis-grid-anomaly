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

### Phase 1 – Neural Learning (The "Brain")
- **dCeNN (Discrete Cellular Neural Network)** encoder (PyTorch, training-time only)
- **ELM (Extreme Learning Machine)** regression heads using closed-form ridge regression
- Optimized for fast training and lightweight inference

### Phase 2 – Symbolic Reasoning (The "Filter")
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
│  ├─ 10–11  Event clustering & master visualizations
│  └─ 12-20  Benchmark Process
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

### Full Pipeline Execution
Run the entire workflow end-to-end:
```bash
make all
```

### Key Individual Steps

**Phase 1: Learning Pipeline**
```bash
make holidays          # Generate Austrian holidays
make preprocess        # Build features from raw data
make split             # Train/test split and scaling
make train             # Train dCeNN-ELM models
make thresholds        # Calibrate anomaly thresholds
make detect            # Detect statistical anomalies
```

**Phase 2: Symbolic Reasoning**
```bash
make asp               # Apply ASP constraints, refine anomalies
```

**Phase 3: Finance & Evaluation**
```bash
make finance           # Map refined anomalies to financial utility
make eval              # Compute performance metrics
make edge              # Export edge-ready model bundle
```

**Visualization & Analysis**
```bash
make event_table       # Build event table from anomalies
make plot_event_table_all  # Generate timeline plots for all signals
```

---

## Makefile Commands Reference

| Command | Purpose |
|---------|---------|
| `make all` | Run full pipeline: data → training → reasoning → finance → evaluation |
| `make holidays` | Generate Austrian public holiday facts |
| `make preprocess` | Build engineered features from raw ENTSO-E & weather data |
| `make split` | Perform train/test split and feature scaling |
| `make train` | Train dCeNN encoder and ELM regression heads |
| `make thresholds` | Calibrate conformal prediction thresholds |
| `make detect` | Run statistical anomaly detection |
| `make asp` | Apply ASP reasoning to refine anomalies |
| `make finance` | Map refined anomalies to economic profit/loss |
| `make eval` | Compute evaluation metrics on test set |
| `make edge` | Export lightweight model bundle for edge deployment |
| `make event_table` | Build structured event table from detected anomalies |
| `make plot_event_table_all` | Generate visualization plots for Price, Load, Solar CF, Wind CF |
| `make clean` | Remove artifacts, reports, and intermediate files |

---

## Edge Deployment

The framework is optimized for deployment on resource-constrained edge devices:

- **Export Format**: `model_bundle.npz` (NumPy binary archive)
- **Dependencies at Inference**: NumPy only (no PyTorch, Scikit-Learn, or ML libraries)
- **Ring Buffer**: 72-hour temporal feature buffer for online inference
- **Target Hardware**: Raspberry Pi, NVIDIA Jetson, IoT gateways
- **Inference Script**: See `src/09_edge_export.py`

---

## Method Summary

| Component | Role | Logic Type |
|-----------|------|-----------|
| dCeNN | Feature extraction | Neural |
| ELM | Fast regression | Linear |
| Conformal Prediction | Thresholding | Statistical |
| ASP (Clingo) | Rule enforcement | Symbolic |
| Finance Mapping | Market utility | Economic |

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and commit: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request

Please ensure all code follows the existing style and includes appropriate documentation.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Author

**Chamil Oshan Abeysekara**  
Master's Candidate – Autonomous Systems & Robotics  
University of Klagenfurt, Austria

**Contact:** chamilabeysekara@gmail.com  
**GitHub:** [cHaMiL8020](https://github.com/cHaMiL8020)

---

## Citation

If you use this work in your research, please cite:

```bibtex
@thesis{abeysekara2025grid,
  title  = {Finance-Aware, Weather-Informed Anomaly Detection for Electricity Markets},
  author = {Abeysekara, Chamil Oshan},
  year   = {2025},
  school = {University of Klagenfurt}
}
```
---

## Support & Issues
