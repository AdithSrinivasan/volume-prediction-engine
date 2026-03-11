# Volume Prediction Engine — Project Memory

## Key files
- `lightgbm.py` — main model script (LightGBM, Optuna HPO, 5-fold TS CV, 4 plots)
- `features.py` — feature engineering
- `market_data.py` — data pull
- `BTC-USDT_features_15min.parquet` — 2,975 rows × 85 features (May 2025)

## Python env
- Use `python3.11` (at `/opt/homebrew/bin/python3.11`)
- Packages: pandas, pyarrow, lightgbm==4.6.0, optuna, scikit-learn, matplotlib, plotnine
- Install path: `/opt/homebrew/lib/python3.11/site-packages`

## Filename shadow issue
- `lightgbm.py` shadows the `lightgbm` package on import
- Fixed at top of file: remove script dir from `sys.path` before `import_module("lightgbm")`

## Data notes
- 15-min OHLCV + 60 volume lags + microstructure features; DatetimeIndex `timestamp`
- Volume heavily right-skewed (skew ~10); model trains on log1p(target)
- First ~60 rows have lag_volume_2-60 set to 0 (no history before dataset start)

## Model results (run 2026-03-11)
- CV mean R²: -0.022 (typical for noisy financial next-step prediction)
- Test R²: 0.0004, RMSE: 9,658, MAPE: 299%, Directional Acc: 0.705
- Most important features: num_trades, trade_notional, lag_volume_1/2/3
