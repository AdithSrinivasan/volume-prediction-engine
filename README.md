# volume-prediction-engine

> Equity volume prediction modelling across small, mid, and large cap stocks using classical ML and deep learning.

---

## Overview

**volume-prediction-engine** is a machine learning pipeline for predicting equity trading volume across market cap tiers — small, mid, and large cap. The project benchmarks a suite of models from interpretable tree-based methods to gradient boosting ensembles and neural networks, providing a comparative analysis of predictive performance across different equity segments.

This project was developed as part of FINM 33160 Machine Learning in Finance.

---

## Models

| Model | Type | Notes |
|---|---|---|
| Decision Tree | Tree-based | Baseline interpretable model |
| Random Forest | Ensemble (Bagging) | Reduces variance via bootstrapped trees |
| AdaBoost | Ensemble (Boosting) | Sequential weak learner boosting |
| LightGBM | Gradient Boosting | Leaf-wise growth, fast on large datasets |
| XGBoost | Gradient Boosting | Regularised boosting, strong benchmark |
| Neural Network | Deep Learning | MLP with configurable depth/width |

---

## Market Cap Tiers

| Tier | Definition (approx.) |
|---|---|
| Small Cap | < $2B market cap |
| Mid Cap | $2B – $10B market cap |
| Large Cap | > $10B market cap |

---

## Project Structure

```
volume-prediction-engine/
│
├── data/
│   ├── raw/                  # Raw OHLCV data
│   ├── processed/            # Cleaned & feature-engineered data
│   └── splits/               # Train/val/test splits by cap tier
│
├── features/
│   ├── feature_engineering.py
│   └── feature_selection.py
│
├── models/
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── adaboost.py
│   ├── lightgbm_model.py
│   ├── xgboost_model.py
│   └── neural_network.py
│
├── evaluation/
│   ├── metrics.py            # MAE, RMSE, MAPE, R²
│   └── comparison.py         # Cross-model benchmarking
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
│
├── results/
│   └── model_comparison.csv
│
├── requirements.txt
└── README.md
```

---

## Features

Feature engineering draws on standard equity market signals:

- **Price-based**: OHLC, returns, log returns, price momentum
- **Volume-based**: Lagged volume, rolling averages (5d, 10d, 20d), VWAP
- **Volatility**: ATR, rolling std of returns, Bollinger Bands
- **Microstructure**: Bid-ask spread proxies, intraday range
- **Calendar effects**: Day-of-week, month-end, earnings window flags
- **Market cap tier**: Encoded as a feature and used for stratified analysis

---

## Evaluation Metrics

Models are evaluated on:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error
- **R²** — Coefficient of determination
- **Directional Accuracy** — % of correct volume up/down predictions

Results are reported separately per market cap tier to surface differential performance across equity segments.

---

## Getting Started

### Prerequisites

```bash
python >= 3.9
```

### Installation

```bash
git clone https://github.com/AdithSrinivasan/volume-prediction-engine.git
cd volume-prediction-engine
pip install -r requirements.txt
```

### Run Training

```bash
python models/random_forest.py --tier large --train
python models/xgboost_model.py --tier small --train
```

### Run All Models & Compare

```bash
python evaluation/comparison.py --tier all
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
torch
matplotlib
seaborn
yfinance
jupyter
```

---

## Results

Model performance summary across cap tiers (to be populated after training):

| Model | Small Cap RMSE | Mid Cap RMSE | Large Cap RMSE |
|---|---|---|---|
| Decision Tree | — | — | — |
| Random Forest | — | — | — |
| AdaBoost | — | — | — |
| LightGBM | — | — | — |
| XGBoost | — | — | — |
| Neural Network | — | — | — |

---

## License

MIT

---

*Built for ML in Finance — University of Chicago, FINM 33160
