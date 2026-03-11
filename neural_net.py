#!/usr/bin/env python3.11
"""
BTC-USDT 15-min Volume Prediction — PyTorch MLP
================================================
Target  : volume at t+1 (next 15-minute bucket)
Model   : Multi-layer Perceptron  [Linear → BN → GELU → Dropout] × n → Linear(1)
Scaling : StandardScaler per fold (fit on train, applied to val/test)
Target  : log1p(volume) → back-transformed for all reported metrics
CV      : 5-fold TimeSeriesSplit on first 90 % of data
HPO     : Optuna TPE (50 trials), minimising CV RMSE in original space
Plots   : 6 PNGs  (nn_*.png, dark theme, mirrors lightgbm.py structure)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(4)

# ── constants ──────────────────────────────────────────────────────────────────
DATA_PATH    = "BTC-USDT_features_15min.parquet"
TEST_FRAC    = 0.10
N_SPLITS     = 5
N_TRIALS     = 50
MAX_EPOCHS   = 200   # full CV / final model
PATIENCE     = 25
HPO_EPOCHS   = 80    # reduced budget during Optuna search
HPO_PATIENCE = 12
SEED         = 42

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available()          else
    torch.device("cpu")
)

# ── style (matches lightgbm.py) ────────────────────────────────────────────────
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
df["target"] = df["volume"].shift(-1)
df = df.dropna(subset=["target"])

FEATURE_COLS = [c for c in df.columns if c != "target"]
X          = df[FEATURE_COLS].values.astype(np.float32)
y_raw      = df["target"].values.astype(np.float32)
y          = np.log1p(y_raw)
timestamps = df.index

print(f"Dataset   : {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"Date range: {timestamps[0]}  →  {timestamps[-1]}")
print(f"Device    : {DEVICE}")


# ── 2. Train/test split ────────────────────────────────────────────────────────
split_idx    = int(len(X) * (1 - TEST_FRAC))
X_tv, X_te   = X[:split_idx],  X[split_idx:]
y_tv, y_te   = y[:split_idx],  y[split_idx:]
yr_tv, yr_te = y_raw[:split_idx], y_raw[split_idx:]
ts_tv, ts_te = timestamps[:split_idx], timestamps[split_idx:]

print(f"Train/Val : {len(X_tv):,} rows  |  Test: {len(X_te):,} rows")


# ── 3. Model ───────────────────────────────────────────────────────────────────
class VolumeNet(nn.Module):
    """MLP: [Linear → BatchNorm1d → GELU → Dropout] × n_layers → Linear(1)."""

    def __init__(self, input_dim: int, hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_model(input_dim: int, hp: dict) -> VolumeNet:
    torch.manual_seed(SEED)
    return VolumeNet(input_dim, hp["hidden_size"], hp["n_layers"], hp["dropout"]).to(DEVICE)


# ── 4. Training loop ───────────────────────────────────────────────────────────
def train_model(
    model: VolumeNet,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    hp: dict,
    max_epochs: int = MAX_EPOCHS,
    patience: int   = PATIENCE,
    record_history: bool = False,
):
    """
    Train with AdamW + CosineAnnealingLR + early stopping on val MSE.
    Returns (best_val_loss, best_epoch, [train_hist, val_hist]).
    history lists are empty when record_history=False.
    """
    Xtr_t = torch.from_numpy(X_tr).to(DEVICE)
    ytr_t = torch.from_numpy(y_tr).to(DEVICE)
    Xva_t = torch.from_numpy(X_va).to(DEVICE)
    yva_t = torch.from_numpy(y_va).to(DEVICE)

    drop_last = len(X_tr) > hp["batch_size"]
    ds  = TensorDataset(Xtr_t, ytr_t)
    dl  = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True, drop_last=drop_last)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=hp["lr"] * 0.01
    )

    best_val   = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    wait       = 0
    best_epoch = 0
    train_hist: list[float] = []
    val_hist:   list[float] = []

    for epoch in range(max_epochs):
        model.train()
        batch_losses = []
        for Xb, yb in dl:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xva_t), yva_t).item()

        if record_history:
            train_hist.append(float(np.mean(batch_losses)))
            val_hist.append(val_loss)

        if val_loss < best_val - 1e-7:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait       = 0
            best_epoch = epoch
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return best_val, best_epoch, train_hist, val_hist


# ── 5. Metrics helper ──────────────────────────────────────────────────────────
vol_idx = FEATURE_COLS.index("volume")


def compute_metrics(yr_true: np.ndarray, yr_pred: np.ndarray, X_cur: np.ndarray) -> dict:
    r2   = r2_score(yr_true, yr_pred)
    rmse = float(np.sqrt(mean_squared_error(yr_true, yr_pred)))
    mae  = float(mean_absolute_error(yr_true, yr_pred))
    mape = float(np.mean(np.abs((yr_true - yr_pred) / (yr_true + 1e-8))) * 100)
    vol_cur = X_cur[:, vol_idx]
    dacc = float(np.mean(
        np.sign(yr_pred - vol_cur) == np.sign(yr_true - vol_cur)
    ))
    return dict(R2=r2, RMSE=rmse, MAE=mae, MAPE=mape, DirAcc=dacc)


def predict(model: VolumeNet, X_s: np.ndarray) -> np.ndarray:
    """Return original-scale predictions."""
    model.eval()
    with torch.no_grad():
        log_pred = model(torch.from_numpy(X_s).to(DEVICE)).cpu().numpy()
    return np.expm1(np.clip(log_pred, 0, None))


# ── 6. Optuna HPO ──────────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=N_SPLITS)


def objective(trial: optuna.Trial) -> float:
    hp = {
        "n_layers":    trial.suggest_int("n_layers",    2, 5),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256, 512]),
        "dropout":     trial.suggest_float("dropout",   0.05, 0.5),
        "lr":          trial.suggest_float("lr",        1e-4, 5e-3, log=True),
        "batch_size":  trial.suggest_categorical("batch_size", [64, 128, 256]),
        "weight_decay":trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
    }

    fold_rmses = []
    for tr_idx, va_idx in tscv.split(X_tv):
        Xtr, Xva = X_tv[tr_idx], X_tv[va_idx]
        ytr, yva = y_tv[tr_idx], y_tv[va_idx]
        yr_va    = yr_tv[va_idx]

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
        Xva_s = sc.transform(Xva).astype(np.float32)

        model = make_model(Xtr_s.shape[1], hp)
        train_model(model, Xtr_s, ytr, Xva_s, yva, hp,
                    max_epochs=HPO_EPOCHS, patience=HPO_PATIENCE)

        fold_rmses.append(float(np.sqrt(mean_squared_error(yr_va, predict(model, Xva_s)))))

    return float(np.mean(fold_rmses))


print(f"\nRunning Optuna HPO ({N_TRIALS} trials, {N_SPLITS}-fold TimeSeriesSplit)…")
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_hp = study.best_params
print(f"\nBest CV RMSE (orig. scale): {study.best_value:,.2f}")
print("Best params:")
for k, v in best_hp.items():
    print(f"  {k}: {v}")


# ── 7. Full 5-fold CV evaluation ───────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"{'Fold':<8}{'R²':>8}{'RMSE':>12}{'MAE':>12}{'MAPE %':>10}")
print(f"{'─'*60}")

cv_records     = []
fold_pred_dfs  = []
fold_best_ep   = []

for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_tv), 1):
    Xtr, Xva = X_tv[tr_idx], X_tv[va_idx]
    ytr, yva = y_tv[tr_idx], y_tv[va_idx]
    yr_va    = yr_tv[va_idx]
    ts_va    = ts_tv[va_idx]

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
    Xva_s = sc.transform(Xva).astype(np.float32)

    model = make_model(Xtr_s.shape[1], best_hp)
    _, best_epoch, _, _ = train_model(model, Xtr_s, ytr, Xva_s, yva, best_hp)
    fold_best_ep.append(best_epoch)

    pred_raw = predict(model, Xva_s)
    m = compute_metrics(yr_va, pred_raw, Xva)

    cv_records.append({**m, "fold": fold})
    fold_pred_dfs.append(pd.DataFrame({
        "timestamp": ts_va, "actual": yr_va, "pred": pred_raw, "fold": fold
    }))
    print(f"  {fold:<6}{m['R2']:>8.4f}{m['RMSE']:>12,.1f}{m['MAE']:>12,.1f}{m['MAPE']:>10.2f}")

cv_df  = pd.DataFrame(cv_records)
all_cv = pd.concat(fold_pred_dfs, ignore_index=True)
avg_ep = int(np.round(np.mean(fold_best_ep)))

print(f"{'─'*60}")
print(f"  {'Mean':6}{cv_df.R2.mean():>8.4f}±{cv_df.R2.std():.4f}"
      f"  RMSE={cv_df.RMSE.mean():,.1f}±{cv_df.RMSE.std():.1f}"
      f"  MAPE={cv_df.MAPE.mean():.2f}%")


# ── 8. Final model (trained on full train/val, fixed epochs = avg CV best) ─────
print(f"\nTraining final model (epochs={avg_ep}) on full train/val set…")

final_sc = StandardScaler()
X_tv_s   = final_sc.fit_transform(X_tv).astype(np.float32)
X_te_s   = final_sc.transform(X_te).astype(np.float32)

final_model = make_model(X_tv_s.shape[1], best_hp)

# Override scheduler/patience: run exactly avg_ep epochs, record history
# We pass a dummy val set (last 10% of tv) only for loss tracking, not early stopping
val_split   = int(len(X_tv_s) * 0.9)
_, _, tr_hist, va_hist = train_model(
    final_model,
    X_tv_s[:val_split], y_tv[:val_split],
    X_tv_s[val_split:], y_tv[val_split:],
    best_hp,
    max_epochs=avg_ep + PATIENCE,
    patience=PATIENCE,
    record_history=True,
)

pred_raw_te = predict(final_model, X_te_s)
pred_raw_tv = predict(final_model, X_tv_s)

tv_met = compute_metrics(yr_tv, pred_raw_tv, X_tv)
te_met = compute_metrics(yr_te, pred_raw_te, X_te)

print(f"\n{'═'*58}")
print(f"  {'Metric':<16}{'Train/Val':>18}{'Test':>18}")
print(f"{'─'*58}")
for k in ("R2", "RMSE", "MAE", "MAPE", "DirAcc"):
    fmt = ".4f" if k in ("R2", "DirAcc") else ",.2f" if k != "MAPE" else ".2f"
    unit = " %" if k == "MAPE" else ""
    print(f"  {k:<16}{tv_met[k]:>17{fmt}}{unit}  {te_met[k]:>15{fmt}}{unit}")
print(f"{'═'*58}")


# ── 9. Permutation feature importance (test set) ───────────────────────────────
print("\nComputing permutation feature importance…")
np.random.seed(SEED)
N_REPEATS = 5

final_model.eval()
base_rmse = te_met["RMSE"]
importances = np.zeros(X_te_s.shape[1])

for j in range(X_te_s.shape[1]):
    scores = []
    for _ in range(N_REPEATS):
        X_perm = X_te_s.copy()
        np.random.shuffle(X_perm[:, j])
        pr = predict(final_model, X_perm)
        scores.append(float(np.sqrt(mean_squared_error(yr_te, pr))))
    importances[j] = np.mean(scores) - base_rmse  # increase in RMSE when feature is shuffled


# ── 10. Plots ──────────────────────────────────────────────────────────────────

def _annotate(ax, met):
    ax.text(0.02, 0.95,
        f"R²={met['R2']:.3f}   RMSE={met['RMSE']:,.0f}"
        f"   MAPE={met['MAPE']:.1f}%   DirAcc={met['DirAcc']:.3f}",
        transform=ax.transAxes, fontsize=9, color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, alpha=0.7))


def _overview_fig(ts, yr_true, yr_pred, met, cv_df_, title, fname):
    """2×3 overview: time-series (top full-width) + scatter + residuals + CV R²."""
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # A — time-series
    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.plot(ts, yr_true,  color=C["blue"],   lw=0.8, label="Actual",    alpha=0.9)
    ax_ts.plot(ts, yr_pred,  color=C["orange"], lw=0.8, label="Predicted", alpha=0.85)
    ax_ts.set_title(title, fontsize=12, fontweight="bold")
    ax_ts.set_ylabel("Volume (BTC)")
    ax_ts.yaxis.set_major_formatter(fmt_k)
    ax_ts.legend(loc="upper right")
    ax_ts.grid(True, alpha=0.4)
    _annotate(ax_ts, met)

    # B — scatter
    ax_sc = fig.add_subplot(gs[1, 0])
    ax_sc.scatter(yr_true, yr_pred, s=5, alpha=0.3, color=C["blue"])
    lo, hi = min(yr_true.min(), yr_pred.min()), max(yr_true.max(), yr_pred.max())
    ax_sc.plot([lo, hi], [lo, hi], "--", color=C["orange"], lw=1.5, label="Perfect fit")
    ax_sc.set_xlabel("Actual Volume")
    ax_sc.set_ylabel("Predicted Volume")
    ax_sc.set_title("Actual vs Predicted (Scatter)", fontsize=10, fontweight="bold")
    ax_sc.xaxis.set_major_formatter(fmt_k)
    ax_sc.yaxis.set_major_formatter(fmt_k)
    ax_sc.legend(fontsize=8)
    ax_sc.grid(True, alpha=0.4)

    # C — residuals
    resid = yr_true - yr_pred
    ax_rd = fig.add_subplot(gs[1, 1])
    ax_rd.hist(resid, bins=55, color=C["blue"], alpha=0.8, edgecolor="none")
    ax_rd.axvline(0, color=C["orange"], lw=1.5, linestyle="--")
    ax_rd.set_xlabel("Residual  (Actual − Predicted)")
    ax_rd.set_ylabel("Count")
    ax_rd.set_title("Residual Distribution", fontsize=10, fontweight="bold")
    ax_rd.xaxis.set_major_formatter(fmt_k)
    ax_rd.grid(True, alpha=0.4)
    ax_rd.text(0.97, 0.95, f"μ = {resid.mean():,.0f}\nσ = {resid.std():,.0f}",
               transform=ax_rd.transAxes, fontsize=8, ha="right", va="top",
               color=TEXT_COLOR,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, alpha=0.7))

    # D — CV R² per fold (only shown in test overview; replaced by comparison in tv overview)
    ax_cv = fig.add_subplot(gs[1, 2])
    if cv_df_ is not None:
        folds = cv_df_["fold"].values
        bars  = ax_cv.bar(folds, cv_df_["R2"].values, color=C["blue"], alpha=0.85, width=0.6)
        ax_cv.axhline(cv_df_["R2"].mean(), color=C["orange"], lw=1.5, linestyle="--",
                      label=f"Mean = {cv_df_.R2.mean():.3f}")
        ax_cv.set_xlabel("CV Fold")
        ax_cv.set_ylabel("R²")
        ax_cv.set_title("R² by CV Fold", fontsize=10, fontweight="bold")
        ax_cv.set_xticks(folds)
        ax_cv.legend(fontsize=8)
        ax_cv.grid(True, alpha=0.4, axis="y")
        for bar, val in zip(bars, cv_df_["R2"].values):
            ax_cv.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.002,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    else:
        # train/val overview: show tv vs test comparison
        metric_labels = ["R²", "MAPE %", "DirAcc"]
        tv_v = [tv_met["R2"], tv_met["MAPE"] / 100, tv_met["DirAcc"]]
        te_v = [te_met["R2"], te_met["MAPE"] / 100, te_met["DirAcc"]]
        x = np.arange(len(metric_labels))
        w = 0.35
        ax_cv.bar(x - w / 2, tv_v, width=w, color=C["blue"],   alpha=0.85, label="Train/Val")
        ax_cv.bar(x + w / 2, te_v, width=w, color=C["orange"], alpha=0.85, label="Test")
        ax_cv.set_xticks(x)
        ax_cv.set_xticklabels(metric_labels)
        ax_cv.axhline(0, color=TEXT_COLOR, lw=0.5, alpha=0.4)
        ax_cv.set_title("Train/Val vs Test Metrics", fontsize=10, fontweight="bold")
        ax_cv.legend(fontsize=8)
        ax_cv.grid(True, alpha=0.4, axis="y")
        for xi, (tv, te) in zip(x, zip(tv_v, te_v)):
            ax_cv.text(xi - w / 2, tv + 0.01, f"{tv:.2f}", ha="center", fontsize=7)
            ax_cv.text(xi + w / 2, te + 0.01, f"{te:.2f}", ha="center", fontsize=7)

    fig.suptitle("BTC-USDT 15-min Volume Prediction — PyTorch MLP",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


# Fig 1 — test set overview
_overview_fig(ts_te, yr_te, pred_raw_te, te_met, cv_df,
              "Test Set — Actual vs Predicted Volume (t+1)",
              "nn_test_overview.png")

# Fig 2 — train/val set overview
_overview_fig(ts_tv, yr_tv, pred_raw_tv, tv_met, None,
              "Train/Val Set — Actual vs Predicted Volume (t+1)",
              "nn_trainval_overview.png")


# Fig 3 — CV fold predictions ──────────────────────────────────────────────────
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
fig3.savefig("nn_cv_fold_predictions.png", dpi=150, bbox_inches="tight")
print("Saved: nn_cv_fold_predictions.png")
plt.close(fig3)


# Fig 4 — Optuna history + hyperparameter importance ──────────────────────────
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
fig4.patch.set_facecolor(DARK_BG)

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
    ax4b.set_visible(False)

fig4.suptitle("Optuna Hyperparameter Search", fontsize=13, fontweight="bold")
fig4.tight_layout()
fig4.savefig("nn_optuna_results.png", dpi=150, bbox_inches="tight")
print("Saved: nn_optuna_results.png")
plt.close(fig4)


# Fig 5 — Permutation feature importance (top 30) ─────────────────────────────
imp_series = pd.Series(importances, index=FEATURE_COLS)
# Keep only features that increase RMSE when shuffled (positive importance)
top30 = imp_series.nlargest(30).sort_values()

fig5, ax5 = plt.subplots(figsize=(10, 10))
fig5.patch.set_facecolor(DARK_BG)
ax5.set_facecolor(PANEL_BG)
colors5 = plt.cm.Blues(np.linspace(0.4, 0.9, len(top30)))
ax5.barh(range(len(top30)), top30.values, color=colors5, alpha=0.9)
ax5.set_yticks(range(len(top30)))
ax5.set_yticklabels(top30.index, fontsize=9)
ax5.set_xlabel("Increase in Test RMSE when feature is shuffled")
ax5.set_title("Top 30 Permutation Feature Importances", fontsize=12, fontweight="bold")
ax5.grid(True, alpha=0.4, axis="x")
fig5.tight_layout()
fig5.savefig("nn_feature_importance.png", dpi=150, bbox_inches="tight")
print("Saved: nn_feature_importance.png")
plt.close(fig5)


# Fig 6 — Training loss curve (final model) ───────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(10, 5))
fig6.patch.set_facecolor(DARK_BG)
ax6.set_facecolor(PANEL_BG)
epochs = range(1, len(tr_hist) + 1)
ax6.plot(epochs, tr_hist, color=C["blue"],   lw=1.2, label="Train MSE (log1p scale)")
ax6.plot(epochs, va_hist, color=C["orange"], lw=1.2, label="Val MSE (log1p scale)")
ax6.set_xlabel("Epoch")
ax6.set_ylabel("MSE  (log1p space)")
ax6.set_title("Final Model Training Curve", fontsize=12, fontweight="bold")
ax6.legend()
ax6.grid(True, alpha=0.4)
best_ep_marker = int(np.argmin(va_hist))
ax6.axvline(best_ep_marker + 1, color=C["green"], lw=1.2, linestyle="--",
            label=f"Best val epoch ({best_ep_marker + 1})")
ax6.legend()
fig6.tight_layout()
fig6.savefig("nn_training_curve.png", dpi=150, bbox_inches="tight")
print("Saved: nn_training_curve.png")
plt.close(fig6)

print("\nAll done.")
