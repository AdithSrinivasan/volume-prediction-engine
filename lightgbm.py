#!/usr/bin/env python3.11
"""
BTC-USDT 15-min Volume Prediction — LightGBM
============================================
Target : volume at t+1 (next 15-minute bucket)
Model  : LightGBM GBDT, trained on log1p(volume) to handle heavy right-skew
CV     : 5-fold TimeSeriesSplit on the first 90 % of data
HPO    : Optuna TPE (50 trials), minimising CV RMSE in original space
Plots  : matplotlib (dark theme) — 4 output PNGs
"""

import warnings
warnings.filterwarnings("ignore")

# ── The filename "lightgbm.py" would shadow the installed package.
# Remove this script's directory from sys.path before importing lgb.
import sys as _sys, importlib as _il
_here = _sys.path[0]
_sys.path = [p for p in _sys.path if p != _here]
lgb = _il.import_module("lightgbm")
_sys.path.insert(0, _here)
del _here, _il

import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── constants ──────────────────────────────────────────────────────────────────
DATA_PATH  = "BTC-USDT_features_15min.parquet"
TEST_FRAC  = 0.10
N_SPLITS   = 5
N_TRIALS   = 50
SEED       = 42

DARK_BG    = "#1A1A2E"
PANEL_BG   = "#16213E"
TEXT_COLOR = "#E0E0E0"
GRID_COLOR = "#2A2A4A"
C = {
    "blue":   "#2196F3",
    "orange": "#FF9800",
    "green":  "#4CAF50",
    "red":    "#F44336",
    "purple": "#9C27B0",
    "grey":   "#90A4AE",
}

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   "#3A3A5A",
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "grid.color":       GRID_COLOR,
    "grid.alpha":       0.5,
    "font.family":      "sans-serif",
    "font.size":        10,
})

fmt_k = FuncFormatter(lambda x, _: f"{x:,.0f}")


# ── 1. Load & construct target ─────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH).sort_index()

# Target: volume in the next 15-min bucket (t+1).
# We shift volume up by 1 row so that at row t, target = volume at t+1.
df["target"] = df["volume"].shift(-1)
df = df.dropna(subset=["target"])          # drop final row (no t+1 label)

# NOTE — leakage audit:
#   All feature columns reflect period t only.
#   • lag_volume_1…60 : lags t-1 … t-60      ✓
#   • volume           : current period (= lag_0)  ✓
#   • buy/sell cols    : period-t breakdown    ✓
#   • cumulative_volume: running total to t    ✓
#   Nothing from t+1 leaks into features.
FEATURE_COLS = [c for c in df.columns if c != "target"]

X          = df[FEATURE_COLS].values
y_raw      = df["target"].values          # original-scale target (for metrics)
y          = np.log1p(y_raw)              # log1p-transformed target (for model)
timestamps = df.index

print(f"Dataset   : {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"Date range: {timestamps[0]}  →  {timestamps[-1]}")
print(f"Volume    : mean={y_raw.mean():,.0f}  median={np.median(y_raw):,.0f}"
      f"  skew={pd.Series(y_raw).skew():.1f}")


# ── 2. Temporal train/test split (last 10 % = held-out test) ──────────────────
split_idx   = int(len(X) * (1 - TEST_FRAC))
X_tv, X_te  = X[:split_idx],  X[split_idx:]
y_tv, y_te  = y[:split_idx],  y[split_idx:]          # log1p
yr_tv, yr_te = y_raw[:split_idx], y_raw[split_idx:]  # original
ts_tv, ts_te = timestamps[:split_idx], timestamps[split_idx:]

print(f"Train/Val : {len(X_tv):,} rows  |  Test: {len(X_te):,} rows")


# ── 3. Optuna HPO — optimise CV RMSE in ORIGINAL space ────────────────────────
tscv = TimeSeriesSplit(n_splits=N_SPLITS)


def objective(trial: optuna.Trial) -> float:
    params = {
        "objective":         "regression",
        "metric":            "rmse",
        "verbosity":         -1,
        "n_estimators":      trial.suggest_int("n_estimators",    200, 1500),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves",       20, 300),
        "max_depth":         trial.suggest_int("max_depth",         3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample":         trial.suggest_float("subsample",      0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha",  1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state":      SEED,
    }
    fold_rmses = []
    for tr_idx, va_idx in tscv.split(X_tv):
        Xtr, Xva = X_tv[tr_idx], X_tv[va_idx]
        ytr, yva = y_tv[tr_idx], y_tv[va_idx]
        yr_va    = yr_tv[va_idx]

        m = lgb.LGBMRegressor(**params)
        m.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        pred_log = m.predict(Xva, num_iteration=m.best_iteration_)
        pred_raw = np.expm1(pred_log)
        fold_rmses.append(np.sqrt(mean_squared_error(yr_va, pred_raw)))

    return float(np.mean(fold_rmses))


print(f"\nRunning Optuna HPO ({N_TRIALS} trials, {N_SPLITS}-fold TimeSeriesSplit)…")
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = {
    **study.best_params,
    "objective":    "regression",
    "metric":       "rmse",
    "verbosity":    -1,
    "random_state": SEED,
}
print(f"\nBest CV RMSE (orig. scale): {study.best_value:,.2f}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")


# ── 4. Full 5-fold CV evaluation with best params ─────────────────────────────
print(f"\n{'─'*60}")
print(f"{'Fold':<8}{'R²':>8}{'RMSE':>12}{'MAE':>12}{'MAPE %':>10}")
print(f"{'─'*60}")

cv_records   = []
fold_pred_dfs = []
fold_best_iters = []

for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_tv), 1):
    Xtr, Xva = X_tv[tr_idx], X_tv[va_idx]
    ytr, yva = y_tv[tr_idx], y_tv[va_idx]
    yr_va    = yr_tv[va_idx]
    ts_va    = ts_tv[va_idx]

    m = lgb.LGBMRegressor(**best_params)
    m.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    fold_best_iters.append(m.best_iteration_)

    pred_log = m.predict(Xva, num_iteration=m.best_iteration_)
    pred_raw = np.expm1(pred_log)

    r2   = r2_score(yr_va, pred_raw)
    rmse = np.sqrt(mean_squared_error(yr_va, pred_raw))
    mae  = mean_absolute_error(yr_va, pred_raw)
    mape = float(np.mean(np.abs((yr_va - pred_raw) / (yr_va + 1e-8))) * 100)

    cv_records.append(dict(fold=fold, R2=r2, RMSE=rmse, MAE=mae, MAPE=mape))
    fold_pred_dfs.append(pd.DataFrame({
        "timestamp": ts_va, "actual": yr_va, "pred": pred_raw, "fold": fold
    }))

    print(f"  {fold:<6}{r2:>8.4f}{rmse:>12,.1f}{mae:>12,.1f}{mape:>10.2f}")

cv_df  = pd.DataFrame(cv_records)
all_cv = pd.concat(fold_pred_dfs, ignore_index=True)
avg_best_iter = int(np.round(np.mean(fold_best_iters)))

print(f"{'─'*60}")
print(f"  {'Mean':6}{cv_df.R2.mean():>8.4f}±{cv_df.R2.std():.4f}"
      f"  RMSE={cv_df.RMSE.mean():,.1f}±{cv_df.RMSE.std():.1f}"
      f"  MAPE={cv_df.MAPE.mean():.2f}%")


# ── 5. Final model — train on full train/val, evaluate on test ─────────────────
# Use the mean best_iteration from CV to avoid peeking at the test set.
print(f"\nTraining final model (n_estimators={avg_best_iter}) on full train/val set…")
final_params = {**best_params, "n_estimators": avg_best_iter}

final_model = lgb.LGBMRegressor(**final_params)
final_model.fit(X_tv, y_tv, callbacks=[lgb.log_evaluation(-1)])

pred_log_te = final_model.predict(X_te)
pred_raw_te = np.expm1(pred_log_te)

test_r2   = r2_score(yr_te, pred_raw_te)
test_rmse = np.sqrt(mean_squared_error(yr_te, pred_raw_te))
test_mae  = mean_absolute_error(yr_te, pred_raw_te)
test_mape = float(np.mean(np.abs((yr_te - pred_raw_te) / (yr_te + 1e-8))) * 100)

# Directional accuracy: did model correctly predict whether volume rises vs falls?
# Compare predicted t+1 direction vs current t volume (lag_volume_1 col = volume at t)
# Column 0 in feature matrix is 'volume' (current period) — use it as baseline.
vol_current_te = X_te[:, FEATURE_COLS.index("volume")]
dir_acc = float(np.mean(
    np.sign(pred_raw_te - vol_current_te) == np.sign(yr_te - vol_current_te)
))

print(f"\n{'═'*40}")
print(f"  Test-set Results")
print(f"{'─'*40}")
print(f"  R²              : {test_r2:.4f}")
print(f"  RMSE            : {test_rmse:,.2f}")
print(f"  MAE             : {test_mae:,.2f}")
print(f"  MAPE            : {test_mape:.2f} %")
print(f"  Directional Acc : {dir_acc:.4f}")
print(f"{'═'*40}")


# ── 6. Plots ───────────────────────────────────────────────────────────────────

# ── Fig 1: Overview (test predictions + scatter + residuals + CV R²) ──────────
fig1 = plt.figure(figsize=(18, 11))
fig1.patch.set_facecolor(DARK_BG)
gs = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.45, wspace=0.35)

# A — Time-series: actual vs predicted on test set
ax_ts = fig1.add_subplot(gs[0, :])
ax_ts.plot(ts_te, yr_te,      color=C["blue"],   lw=0.9, label="Actual",    alpha=0.9)
ax_ts.plot(ts_te, pred_raw_te, color=C["orange"], lw=0.9, label="Predicted", alpha=0.85)
ax_ts.set_title("Test Set — Actual vs Predicted Volume (t+1)", fontsize=12, fontweight="bold")
ax_ts.set_ylabel("Volume (BTC)")
ax_ts.yaxis.set_major_formatter(fmt_k)
ax_ts.legend(loc="upper right")
ax_ts.grid(True, alpha=0.4)
ax_ts.text(0.02, 0.95,
    f"R²={test_r2:.3f}   RMSE={test_rmse:,.0f}   MAPE={test_mape:.1f}%   DirAcc={dir_acc:.3f}",
    transform=ax_ts.transAxes, fontsize=9, color=TEXT_COLOR,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, alpha=0.7))

# B — Scatter: actual vs predicted
ax_sc = fig1.add_subplot(gs[1, 0])
ax_sc.scatter(yr_te, pred_raw_te, s=8, alpha=0.45, color=C["blue"])
lo = min(yr_te.min(), pred_raw_te.min())
hi = max(yr_te.max(), pred_raw_te.max())
ax_sc.plot([lo, hi], [lo, hi], "--", color=C["orange"], lw=1.5, label="Perfect fit")
ax_sc.set_xlabel("Actual Volume")
ax_sc.set_ylabel("Predicted Volume")
ax_sc.set_title("Actual vs Predicted (Scatter)", fontsize=10, fontweight="bold")
ax_sc.xaxis.set_major_formatter(fmt_k)
ax_sc.yaxis.set_major_formatter(fmt_k)
ax_sc.legend(fontsize=8)
ax_sc.grid(True, alpha=0.4)

# C — Residual distribution
residuals = yr_te - pred_raw_te
ax_rd = fig1.add_subplot(gs[1, 1])
ax_rd.hist(residuals, bins=50, color=C["blue"], alpha=0.8, edgecolor="none")
ax_rd.axvline(0, color=C["orange"], lw=1.5, linestyle="--")
ax_rd.set_xlabel("Residual  (Actual − Predicted)")
ax_rd.set_ylabel("Count")
ax_rd.set_title("Residual Distribution", fontsize=10, fontweight="bold")
ax_rd.xaxis.set_major_formatter(fmt_k)
ax_rd.grid(True, alpha=0.4)
ax_rd.text(0.97, 0.95,
    f"μ = {residuals.mean():,.0f}\nσ = {residuals.std():,.0f}",
    transform=ax_rd.transAxes, fontsize=8, ha="right", va="top", color=TEXT_COLOR,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, alpha=0.7))

# D — R² by CV fold
ax_cv = fig1.add_subplot(gs[1, 2])
folds = cv_df["fold"].values
bars  = ax_cv.bar(folds, cv_df["R2"].values, color=C["blue"], alpha=0.85, width=0.6)
ax_cv.axhline(cv_df["R2"].mean(), color=C["orange"], lw=1.5, linestyle="--",
              label=f"Mean = {cv_df.R2.mean():.3f}")
ax_cv.set_xlabel("CV Fold")
ax_cv.set_ylabel("R²")
ax_cv.set_title("R² by CV Fold", fontsize=10, fontweight="bold")
ax_cv.set_xticks(folds)
ax_cv.legend(fontsize=8)
ax_cv.grid(True, alpha=0.4, axis="y")
for bar, val in zip(bars, cv_df["R2"].values):
    ax_cv.text(bar.get_x() + bar.get_width() / 2,
               bar.get_height() + 0.002,
               f"{val:.3f}", ha="center", va="bottom", fontsize=8)

fig1.suptitle("BTC-USDT 15-min Volume Prediction — LightGBM",
              fontsize=14, fontweight="bold", y=0.99)
fig1.savefig("volume_prediction_overview.png", dpi=150, bbox_inches="tight")
print("\nSaved: volume_prediction_overview.png")
plt.close(fig1)


# ── Fig 2: Feature importance (top 30) ────────────────────────────────────────
importances = pd.Series(final_model.feature_importances_, index=FEATURE_COLS)
top30 = importances.nlargest(30).sort_values()

fig2, ax2 = plt.subplots(figsize=(10, 10))
fig2.patch.set_facecolor(DARK_BG)
ax2.set_facecolor(PANEL_BG)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top30)))
ax2.barh(range(len(top30)), top30.values, color=colors, alpha=0.9)
ax2.set_yticks(range(len(top30)))
ax2.set_yticklabels(top30.index, fontsize=9)
ax2.set_xlabel("Feature Importance (split count)")
ax2.set_title("Top 30 Feature Importances", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.4, axis="x")
fig2.tight_layout()
fig2.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
print("Saved: feature_importance.png")
plt.close(fig2)


# ── Fig 3: CV fold predictions overlaid on timeline ───────────────────────────
fold_palette = [C["blue"], C["green"], C["purple"], C["orange"], C["red"]]

fig3, ax3 = plt.subplots(figsize=(16, 5))
fig3.patch.set_facecolor(DARK_BG)
ax3.set_facecolor(PANEL_BG)

for fold_id, grp in all_cv.groupby("fold"):
    ax3.plot(grp["timestamp"], grp["actual"],
             color=C["grey"], lw=0.6, alpha=0.45)
    ax3.plot(grp["timestamp"], grp["pred"],
             color=fold_palette[int(fold_id) - 1], lw=0.9, alpha=0.85,
             label=f"Fold {fold_id}  R²={cv_df.loc[cv_df.fold==fold_id,'R2'].values[0]:.3f}")

ax3.set_title("CV Folds — Predicted (coloured) vs Actual (grey)",
              fontsize=12, fontweight="bold")
ax3.set_ylabel("Volume (BTC)")
ax3.yaxis.set_major_formatter(fmt_k)
ax3.legend(loc="upper right", fontsize=8)
ax3.grid(True, alpha=0.4)
fig3.tight_layout()
fig3.savefig("cv_fold_predictions.png", dpi=150, bbox_inches="tight")
print("Saved: cv_fold_predictions.png")
plt.close(fig3)


# ── Fig 4: Optuna optimisation history + hyperparameter importance ─────────────
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
fig4.patch.set_facecolor(DARK_BG)

# 4A — trial RMSE history
trial_vals  = [t.value for t in study.trials if t.value is not None]
best_so_far = np.minimum.accumulate(trial_vals)
ax4a.set_facecolor(PANEL_BG)
ax4a.scatter(range(len(trial_vals)), trial_vals,
             s=14, alpha=0.35, color=C["blue"], label="Trial RMSE")
ax4a.plot(range(len(best_so_far)), best_so_far,
          color=C["orange"], lw=2, label="Best so far")
ax4a.set_xlabel("Trial")
ax4a.set_ylabel("CV RMSE (original scale)")
ax4a.set_title("Optuna Optimisation History", fontsize=11, fontweight="bold")
ax4a.yaxis.set_major_formatter(fmt_k)
ax4a.legend()
ax4a.grid(True, alpha=0.4)

# 4B — hyperparameter importance
ax4b.set_facecolor(PANEL_BG)
try:
    param_imp     = optuna.importance.get_param_importances(study)
    params_sorted = dict(sorted(param_imp.items(), key=lambda x: x[1]))
    ax4b.barh(list(params_sorted.keys()), list(params_sorted.values()),
              color=C["blue"], alpha=0.85)
    ax4b.set_xlabel("Relative Importance (fANOVA)")
    ax4b.set_title("Hyperparameter Importance", fontsize=11, fontweight="bold")
    ax4b.grid(True, alpha=0.4, axis="x")
except Exception:
    ax4b.text(0.5, 0.5, "Not available", ha="center", va="center",
              transform=ax4b.transAxes)
    ax4b.set_visible(False)

fig4.suptitle("Optuna Hyperparameter Search", fontsize=13, fontweight="bold")
fig4.tight_layout()
fig4.savefig("optuna_results.png", dpi=150, bbox_inches="tight")
print("Saved: optuna_results.png")
plt.close(fig4)

print("\nAll done.")
