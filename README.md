# Learning Adaptive Wiener Processes for Market-Driven Equity Forecasting through Deep Learning SDE Parameter Estimation

## 📌 Overview

This repository contains the source code and implementation details for the research paper *"Learning Adaptive Wiener Processes for Market-Driven Equity Forecasting through Deep Learning SDE Parameter Estimation."*

Traditional financial modeling often relies on Geometric Brownian Motion (GBM) with constant drift and volatility parameters, which fails in real-world markets characterized by non-stationarity and regime shifts. This project introduces a **Physics-Informed Hybrid Architecture** that combines **Kolmogorov-Arnold Networks (KAN)** with **Recurrent Neural Networks (RNNs)**.

Instead of predicting stock prices directly, the model uses a custom Wiener Process-based Loss Function to dynamically estimate time-varying **Drift ($\mu$)** and **Volatility ($\sigma$)** parameters of the underlying Stochastic Differential Equation (SDE), enhancing generalizability across asset classes and volatility regimes.

## 📂 Repository Structure

The data and implementations are organized by market sector and stock ticker. The project covers 10 stocks across 5 major sectors:

- **`.`**
  - **`├── Technology`**
    - **`├── AAPL`**
      - **`├── LSTM_based_implementation.ipynb`**  # Hybrid KAN-LSTM (Wiener Loss)
      - **`├── GRU_based_implementation.ipynb`**   # Hybrid KAN-GRU (Wiener Loss)
      - **`├── KAN.ipynb`**                        # Baseline Stacked KAN (MSE Loss)
      - **`├── LSTM.ipynb`**                       # Baseline Standard LSTM (MSE Loss)
      - **`└── GRU.ipynb`**                        # Baseline Standard GRU (MSE Loss)
    - **`└── MSFT`**
      - **`└── ...`**                              # Same structure as AAPL
  - **`├── Banking`**
    - **`├── JPM`**
    - **`└── BAC`**
  - **`├── Healthcare`**
    - **`├── JNJ`**
    - **`└── PFE`**
  - **`├── Entertainment`**
    - **`├── DIS`**
    - **`└── NFLX`**
  - **`├── Energy`**
    - **`├── CVX`**
    - **`└── ENB`**
  - **`└── Stock_Analysis`**
    - **`└── statistical_analysis.ipynb`**        # Descriptive stats, volatility plots, sector-wise return analysis



## 🛠️ Code Description & Methodology

### 1. Hybrid Physics-Informed Models

Located in `LSTM_based_implementation.ipynb` and `GRU_based_implementation.ipynb`.  

- **Input:** Look-back window of historical scaled prices ($T=120$)  
- **Architecture:**  
  - **Encoder:** DenseKAN layer with learnable B-spline activations for non-linear feature extraction  
  - **Decoder:** RNN (LSTM or GRU) capturing temporal dependencies  
  - **Output:** Two neurons representing instantaneous Drift ($\mu$) and Log-Volatility ($\log \sigma$)  
- **Loss Function:** Custom Wiener Loss simulating the next price step using GBM:  
```markdown
The next price step is modeled as `S_{t+1} = S_t * exp(μ - 0.5σ^2 + σ Z)`.

The model optimizes `μ` and `σ` to maximize the likelihood of observed prices.
```markdown



### 2. Baseline Models

Standalone models trained with conventional Mean Squared Error (MSE) loss:

- `KAN.ipynb`: Stacked Kolmogorov-Arnold Network (non-linear, no recurrence)  
- `LSTM.ipynb`: Standard LSTM network  
- `GRU.ipynb`: Standard GRU network  

### 3. Statistical Analysis

`Stock_Analysis/statistical_analysis.ipynb` contains scripts to:

- Compute descriptive statistics (mean, std dev, skewness, kurtosis)  
- Visualize 120-day rolling volatility to illustrate market regime shifts  
- Analyze return distributions across sectors  

## 🚀 Getting Started

### Prerequisites

- Python 3.8+  
- `tensorflow` (2.x)  
- `tfkan` (KAN layers for TensorFlow)  
- `yahooquery` or `yfinance` (data fetching)  
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`  

**Installation:**

```bash
pip install tensorflow tfkan yahooquery pandas numpy matplotlib scikit-learn
