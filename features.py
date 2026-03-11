from pathlib import Path
import numpy as np
import pandas as pd

FREQ = "15min"
DEFAULT_INPUT = Path("HighFreq_export/BTC-USDT_trades.parquet")
DEFAULT_OUTPUT_DIR = Path("features_by_exchange")
MINUTES_PER_DAY = 24 * 60
LAG_HISTORY_MINUTES = 60  # How many minutes of lagged data to include (NOT raw lag)
MOMENTUM_MINUTES = 60  # How many minutes of momentum data to include (NOT raw lag)


def _sanitize_exchange(exchange: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in exchange)


def _build_exchange_features(exchange_df: pd.DataFrame) -> pd.DataFrame:
    exchange_df = exchange_df.sort_values("ts").reset_index(drop=True)
    bucket_minutes = pd.Timedelta(FREQ) / pd.Timedelta(minutes=1)
    periods_per_hour = max(1, int(round(60 / bucket_minutes)))

    exchange_df["bucket"] = exchange_df["ts"].dt.floor(FREQ)
    exchange_df["notional"] = exchange_df["qty"] * exchange_df["trade_price"]

    buy_mask = exchange_df["side"].eq("B")
    sell_mask = exchange_df["side"].eq("S")
    exchange_df["buy_qty"] = np.where(buy_mask, exchange_df["qty"], 0.0)
    exchange_df["sell_qty"] = np.where(sell_mask, exchange_df["qty"], 0.0)
    exchange_df["buy_notional"] = np.where(buy_mask, exchange_df["notional"], 0.0)
    exchange_df["sell_notional"] = np.where(sell_mask, exchange_df["notional"], 0.0)

    log_price = np.log(exchange_df["trade_price"])
    exchange_df["tick_log_return_sq"] = log_price.diff().pow(2)

    grouped = exchange_df.groupby("bucket", sort=True)

    features = grouped.agg(
        volume=("qty", "sum"),
        num_trades=("qty", "size"),
        avg_trade_size=("qty", "mean"),
        median_trade_size=("qty", "median"),
        trade_notional=("notional", "sum"),
        buy_volume=("buy_qty", "sum"),
        sell_volume=("sell_qty", "sum"),
        buy_notional=("buy_notional", "sum"),
        sell_notional=("sell_notional", "sum"),
        open_price=("trade_price", "first"),
        high_price=("trade_price", "max"),
        low_price=("trade_price", "min"),
        close_price=("trade_price", "last"),
        realized_var=("tick_log_return_sq", "sum"),
    )

    full_index = pd.date_range(
        features.index.min(),
        features.index.max(),
        freq=FREQ,
        name="timestamp",
    )
    features = features.reindex(full_index)

    zero_fill_cols = [
        "volume",
        "num_trades",
        "avg_trade_size",
        "median_trade_size",
        "trade_notional",
        "buy_volume",
        "sell_volume",
        "buy_notional",
        "sell_notional",
        "realized_var",
    ]
    features[zero_fill_cols] = features[zero_fill_cols].fillna(0.0)

    features["close_price"] = features["close_price"].ffill()
    for col in ["open_price", "high_price", "low_price"]:
        features[col] = features[col].where(features[col].notna(), features["close_price"])

    volume = features["volume"]
    num_lags = max(1, int(np.ceil(LAG_HISTORY_MINUTES / 60 * periods_per_hour)))
    for lag in range(1, num_lags + 1):
        features[f"lag_volume_{lag}"] = volume.shift(lag).fillna(0.0)

    day_index = features.index.normalize()
    features["session_day"] = day_index
    daily_volume_series = volume.groupby(day_index).sum()
    daily_features = pd.DataFrame(index=daily_volume_series.index)
    daily_features.index.name = "session_day"
    daily_features["prev_day_volume"] = daily_volume_series.shift(1) # shift to avoid lookahead
    prev_daily_volume = daily_volume_series.shift(1)
    daily_features["rolling_volume_mean"] = prev_daily_volume.rolling(7, min_periods=1).mean()
    daily_features["rolling_volume_std"] = prev_daily_volume.rolling(7, min_periods=2).std().fillna(0.0)

    features = features.join(daily_features, on="session_day")
    features = features.drop(columns=["session_day"])
    features["prev_day_volume"] = features["prev_day_volume"].fillna(0.0)
    features["rolling_volume_mean"] = features["rolling_volume_mean"].fillna(0.0)
    features["rolling_volume_std"] = features["rolling_volume_std"].fillna(0.0)
    features["cumulative_volume"] = volume.groupby(day_index).cumsum()

    minute_of_day = features.index.hour * 60 + features.index.minute
    features["time_of_day"] = minute_of_day

    bucket_seconds = int(pd.Timedelta(FREQ).total_seconds())
    features["trade_rate_sec"] = features["num_trades"] / bucket_seconds

    total_flow = features["buy_volume"] + features["sell_volume"]
    features["trade_imbalance"] = np.where(
        total_flow > 0,
        (features["buy_volume"] - features["sell_volume"]) / total_flow,
        0.0,
    )

    features["return"] = features["close_price"].pct_change().fillna(0.0)
    features["abs_return"] = features["return"].abs()
    features["realized_volatility"] = np.sqrt(features["realized_var"])
    features["price_range"] = features["high_price"] - features["low_price"]
    momentum_periods = max(1, int(np.ceil(MOMENTUM_MINUTES / 60 * periods_per_hour)))
    features["momentum"] = features["close_price"].pct_change(momentum_periods).fillna(0.0)

    return features.drop(columns=["realized_var"])


def build_features(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    df = pd.read_parquet(input_path, columns=["ts", "side", "qty", "trade_price", "Exchange"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for exchange, exchange_df in df.groupby("Exchange", sort=True):
        features = _build_exchange_features(exchange_df.copy())
        output_path = output_dir / f"BTC-USDT_features_{_sanitize_exchange(str(exchange))}_{FREQ}.parquet"
        features.to_parquet(output_path)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    output_paths = build_features()
    for output_path in output_paths:
        print(output_path.resolve())


if __name__ == "__main__":
    main()
