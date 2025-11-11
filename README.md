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

## 📂 Project Structure

