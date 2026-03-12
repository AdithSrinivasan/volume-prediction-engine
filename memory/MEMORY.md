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

### LightGBM (Optuna HPO, 5-fold TS CV)
- Train/Val: R²=0.444, RMSE=177, MAPE=38.8%, DirAcc=0.690
- Test:      R²=0.339, RMSE=127, MAPE=41.1%, DirAcc=0.668
- CV mean R²: 0.293 (fold R²s: 0.218, 0.163, 0.432, 0.296, 0.357)
- Top features: num_trades, trade_notional, lag_volume_1/2/3

### Neural Net — PyTorch MLP (Optuna HPO, 5-fold TS CV)
- Architecture: [Linear → BN → GELU → Dropout] × n layers, log1p target
- Train/Val: R²=0.451, RMSE=176, MAPE=37.1%, DirAcc=0.695
- Test:      R²=0.377, RMSE=123, MAPE=40.7%, DirAcc=0.681
- CV mean R²: 0.270 (fold R²s: 0.145, 0.126, 0.406, 0.350, 0.322)
- Top features (permutation): return, momentum, abs_return, trade_rate_sec, trade_notional
- NN beats LightGBM on test R² (+0.038), RMSE (−4), and DirAcc (+0.013); LightGBM has higher CV mean R² (+0.023)
- Output plots: outputs/neural_net/nn_*.png (6 PNGs: trainval, test, cv_folds, optuna, feature_importance, training_curve)
