#  Energy Consumption Forecast

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-orange)
![Prophet](https://img.shields.io/badge/Facebook-Prophet-blue)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A time series forecasting system that predicts hourly electricity consumption using two approaches — Facebook Prophet and a custom LSTM neural network — trained on 13 years of real energy data.

 **[Live Demo on Streamlit](https://energy-consumption-forecast-efjzbrfe6hhqm7gbqttdf8.streamlit.app/)**

---

## Project Overview

This project compares classical and deep learning approaches to time series forecasting:

- **Baseline**: Facebook Prophet (automatic trend + seasonality decomposition)
- **Advanced**: LSTM neural network (PyTorch) with 60-day memory window

The goal is to demonstrate that deep learning captures complex non-linear temporal patterns that classical models miss.

---

##  Results

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Facebook Prophet | 1247 MW | 1641 MW | 7.98% |
| LSTM (PyTorch) | 537 MW | 720 MW | **3.61%** |

> **LSTM achieves 55% better accuracy than Prophet** by learning complex non-linear dependencies across 60-day sequences.

---

##  Project Structure

```
energy-consumption-forecast/
├── notebooks/
│   ├── 01_EDA_Energy.ipynb           # Exploratory Data Analysis
│   └── 02_Modeling.ipynb             # Prophet + LSTM modeling
├── app/
│   ├── streamlit_app.py              # Streamlit web application
│   ├── lstm_model.pth                # Trained LSTM weights
│   ├── scaler.pkl                    # MinMaxScaler
│   └── metrics.pkl                   # Evaluation metrics
├── data/
│   └── AEP_hourly.csv                # 13 years of hourly energy data
├── requirements.txt
└── README.md
```



---

##  Dataset

- **Source**: [AEP Hourly Energy Consumption — Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- **Period**: October 2004 → August 2018 (13 years)
- **Size**: 121,273 hourly observations
- **Range**: 9,581 MW (night minimum) → 25,695 MW (summer peak)

---

##  Key Findings from EDA

- **Double seasonality**: peaks in January (heating) and July/August (air conditioning)
- **Daily pattern**: consumption minimum at 4-5 AM, peak at 6-7 PM
- **Weekly pattern**: 8% lower consumption on Sundays vs weekdays
- **Long-term trend**: steady decline since 2008 (energy efficiency improvements)

---

## Models

### Facebook Prophet
- Automatically decomposes series into trend + yearly + weekly seasonality
- Multiplicative seasonality mode for better fit
- Trained on 4,690 days, tested on 365 days

### LSTM Neural Network
- 2-layer LSTM with 64 hidden units + 20% dropout
- 60-day sliding window as input sequence
- Trained for 50 epochs with Adam optimizer (lr=0.001)
- GPU-accelerated training on Google Colab

---

##  Run Locally

```bash
git clone https://github.com/badaraaliouguindo/energy-consumption-forecast
cd energy-consumption-forecast
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

##  Key Learnings

- Time series require careful train/test splitting — never shuffle temporal data
- Prophet is fast and interpretable but assumes regular seasonality patterns
- LSTM captures non-linear dependencies that classical models miss
- MinMaxScaler normalization is critical for neural network convergence
- Longer input sequences (60 days) improve LSTM accuracy on energy data

---

##  Tech Stack

`Python` `PyTorch` `Facebook Prophet` `Streamlit` `Plotly` `Pandas` `scikit-learn` `Google Colab`

---

##  Author

**Badara Aliou Guindo** — Master's student in Data Science & AI
[GitHub](https://github.com/badaraaliouguindo) • [HuggingFace](https://huggingface.co/alioubguindo)
