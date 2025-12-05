Learning Adaptive Wiener Processes for Market-Driven Equity Forecasting through Deep Learning SDE Parameter Estimation

📌 Overview

This repository contains the source code and implementation details for the research paper "Learning Adaptive Wiener Processes for Market-Driven Equity Forecasting through Deep Learning SDE Parameter Estimation."

Traditional financial modeling often relies on Geometric Brownian Motion (GBM) with constant drift and volatility parameters, an assumption that fails in real-world markets characterized by non-stationarity and regime shifts. This project introduces a novel Physics-Informed Hybrid Architecture that combines Kolmogorov-Arnold Networks (KAN) with Recurrent Neural Networks (RNNs).

Instead of predicting stock prices directly, our model utilizes a custom Wiener Process-based Loss Function to dynamically estimate the time-varying Drift ($\mu$) and Volatility ($\sigma$) parameters of the underlying Stochastic Differential Equation (SDE). This approach significantly enhances model generalizability across diverse asset classes and volatility regimes.

📂 Repository Structure

The data and implementations are organized by market sector and individual stock ticker. The project covers 10 stocks across 5 major sectors.

├── 📁 Technology
│   ├── 📁 AAPL
│   │   ├── 📄 LSTM_based_implementation.ipynb  # Hybrid KAN-LSTM (Wiener Loss)
│   │   ├── 📄 GRU_based_implementation.ipynb   # Hybrid KAN-GRU (Wiener Loss)
│   │   ├── 📄 KAN.ipynb                        # Baseline Stacked KAN (MSE Loss)
│   │   ├── 📄 LSTM.ipynb                       # Baseline Standard LSTM (MSE Loss)
│   │   └── 📄 GRU.ipynb                        # Baseline Standard GRU (MSE Loss)
│   └── 📁 MSFT
│       └── ... (Same file structure as above)
│
├── 📁 Banking
│   ├── 📁 JPM
│   └── 📁 BAC
│
├── 📁 Healthcare
│   ├── 📁 JNJ
│   └── 📁 PFE
│
├── 📁 Entertainment
│   ├── 📁 DIS
│   └── 📁 NFLX
│
├── 📁 Energy
│   ├── 📁 CVX
│   └── 📁 ENB
│
└── 📁 Stock_Analysis
    └── 📄 statistical_analysis.ipynb           # Descriptive analysis, volatility plotting, and statistical tests for all assets


🛠️ Code Description & Methodology

1. Hybrid Physics-Informed Models (The Core Research)

Located in LSTM_based_implementation.ipynb and GRU_based_implementation.ipynb.

These notebooks implement the proposed Adaptive Wiener Process framework.

Input: A look-back window of historical scaled prices ($T=120$).

Architecture:

Encoder: A DenseKAN layer that uses learnable B-spline activation functions to extract non-linear features from the time series.

Decoder: An RNN layer (LSTM or GRU) that captures temporal dependencies.

Output: Two neurons representing the instantaneous Drift ($\mu$) and Log-Volatility ($\log \sigma$).

Loss Function: A custom physics-informed loss that simulates the next price step using the GBM formula: $S_{t+1} = S_t \cdot \exp(\mu - 0.5\sigma^2 + \sigma Z)$. The model optimizes $\mu$ and $\sigma$ to maximize the likelihood of the observed price path.

2. Baseline Models

To benchmark performance, we provide standalone implementations of standard deep learning models trained on conventional Mean Squared Error (MSE) loss:

KAN.ipynb: A Stacked Kolmogorov-Arnold Network (purely non-linear, no recurrence).

LSTM.ipynb: A standard Long Short-Term Memory network.

GRU.ipynb: A standard Gated Recurrent Unit network.

3. Statistical Analysis

The Stock_Analysis folder contains scripts to:

Calculate descriptive statistics (Mean, Std Dev, Skewness, Kurtosis).

Visualize 120-day rolling volatility to demonstrate market regime shifts.

Analyze the distribution of returns across different sectors.

🚀 Getting Started

Prerequisites

The project requires Python 3.8+ and the following libraries:

tensorflow (2.x)

tfkan (Library for KAN layers in TensorFlow)

yahooquery or yfinance (For data fetching)

pandas, numpy, matplotlib, scikit-learn

Installation

pip install tensorflow tfkan yahooquery pandas numpy matplotlib scikit-learn


Usage

Navigate to a specific stock folder (e.g., Technology/AAPL).

Open LSTM_based_implementation.ipynb to run the proposed Hybrid KAN-LSTM model.

Execute the cells sequentially to:

Fetch data.

Preprocess and create sequences (Window=120).

Train the model using the Custom Wiener Loss.

Visualize the predicted Drift ($\mu$) and Volatility ($\sigma$) parameters.

📊 Key Findings

Adaptability: The Hybrid KAN-RNN architecture successfully estimates adaptive volatility parameters that correlate with real-world market events (e.g., the 2020 crash).

Generalizability: By constraining the model to physical SDE laws, the hybrid approach generalizes better across diverse sectors (e.g., high-volatility Tech vs. low-volatility Healthcare) compared to baseline models that often overfit to specific price trends.

Interpretability: Unlike "black-box" price predictors, this architecture provides transparent outputs ($\mu, \sigma$) that offer actionable insights into the market's perceived risk and trend.

Contact: [dasj@myumanitoba.ca]
