# Finance-Aware, Weather-Informed Anomaly Detection for Electricity Markets & Grid Operations  
### Combining Learning and Reasoning for Anomaly Detection in Multivariate Time Series under Uncertainty  
**Approach:** Efficient dCeNN–ELM Pipeline (Phase 1 Implementation)

---

## 🧩 Overview

This repository implements the **Phase 1 baseline learning pipeline** of a master’s thesis on hybrid learning-reasoning systems for smart grid anomaly detection.  
It builds an **edge-friendly deep-learning pipeline** that forecasts and detects anomalies in Austrian electricity-market variables using **weather, finance, and capacity factors**.

**Core idea:**  
Combine a *Discrete Cellular Neural Network (dCeNN)* encoder with *Extreme Learning Machine (ELM)* heads for efficient multivariate forecasting,  
then detect anomalies through conformal threshold calibration and map their financial impact.

---

## ⚙️ Features Implemented

| Component | Description |
|------------|-------------|
| **Data preprocessing** | Merges ENTSO-E load/price data, Open-Meteo weather grid, and renewable capacity factors. Adds engineered features: air density, PV proxy, wind-power proxy, holiday/time encodings. |
| **Model training** | Trains a lightweight **dCeNN → ELM** hybrid network for multivariate time-series prediction (solar CF, wind CF, load, price). |
| **Threshold calibration** | Adaptive anomaly bands computed using conformal quantiles (α = 0.90) per target. |
| **Anomaly detection** | Identifies anomalies in 2022 test data, highlighting market-stress and renewable-transition events. |
| **Finance mapping** | Computes utility of anomaly-aware buy/sell policies vs baseline. |
| **Evaluation** | RMSE/MAE metrics (validation & test), monthly summaries, visualizations. |
| **Edge export** | Exports compact model bundle (`edge/model_bundle.npz`) for Raspberry Pi or Jetson Nano inference. |

---


## 🧠 Data Sources

| Domain | Source | Resolution | Key Fields |
|---------|---------|-------------|-------------|
| **Load & Price** | ENTSO-E Transparency Platform | Hourly | Total Load (MW), Day-Ahead Price (€/MWh) |
| **Weather** | Open-Meteo Historical API | 0.1° – 0.25° grid | Temperature, Wind Speed, Radiation, Pressure, Humidity, Precipitation |
| **Installed Capacity** | ENTSO-E (Annual → Hourly interpolation) | Hourly (derived) | Wind & Solar Capacity (MW) |
| **Calendar** | Austrian public holidays (2017-2022) | Daily | Holiday/Weekend/Season encodings |

Train = 2017–2020 Validation = 2021 Test = 2022

---

## 🚀 Quick Start (Ubuntu / Codespaces)

```bash
# 1️⃣ Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2️⃣ Install system deps & Python packages
sudo apt-get update
sudo apt-get install -y make
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3️⃣ Run the full pipeline
make holidays     # generate AT_public_holidays_2017_2022.csv
make preprocess   # build engineered hourly dataset
make split        # split & scale train/val/test sets
make train        # train dCeNN–ELM model
make thresholds   # calibrate anomaly thresholds
make detect       # detect anomalies (2022)
make finance      # finance utility mapping
make eval         # compute metrics
make edge         # export edge model bundle


Detected anomalies (2022):

Price spikes in late 2021–22 crisis period

Load surges during winter and holidays

Renewable generation ramps during wind/solar transitions

Finance mapping output:
utility_vs_time_2022.png shows cumulative utility vs baseline.
(Current version shows net loss → policy sign inversion planned.)


Chamil Oshan Abeysekara
MSc Autonomous Systems & Robotics — University of Klagenfurt
📧 [chamilab@edu.aau.at]
🌐 github.com/cHaMiL8020


