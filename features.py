import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FREQ = "1min"
DEFAULT_INPUT = Path("HighFreq_export/BTC-USDT_fewer_trades.parquet")
DEFAULT_OUTPUT = Path(f"BTC-USDT_features_{FREQ}.parquet")
MINUTES_PER_DAY = 24 * 60
NUM_LAGS = 60


def build_features(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_parquet(input_path, columns=["ts", "side", "qty", "trade_price"])
    df = df.sort_values("ts").reset_index(drop=True)

    df["bucket"] = df["ts"].dt.floor(FREQ)
    df["notional"] = df["qty"] * df["trade_price"]

    buy_mask = df["side"].eq("B")
    sell_mask = df["side"].eq("S")
    df["buy_qty"] = np.where(buy_mask, df["qty"], 0.0)
    df["sell_qty"] = np.where(sell_mask, df["qty"], 0.0)
    df["buy_notional"] = np.where(buy_mask, df["notional"], 0.0)
    df["sell_notional"] = np.where(sell_mask, df["notional"], 0.0)

    log_price = np.log(df["trade_price"])
    df["tick_log_return_sq"] = log_price.diff().pow(2)

    grouped = df.groupby("bucket", sort=True)

    minute_features = grouped.agg(
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
        minute_features.index.min(),
        minute_features.index.max(),
        freq=FREQ,
        name="timestamp",
    )
    minute_features = minute_features.reindex(full_index)

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
    minute_features[zero_fill_cols] = minute_features[zero_fill_cols].fillna(0.0)

    minute_features["close_price"] = minute_features["close_price"].ffill()
    for col in ["open_price", "high_price", "low_price"]:
        minute_features[col] = minute_features[col].where(minute_features[col].notna(), minute_features["close_price"])

    volume = minute_features["volume"]
    for lag in range(1, NUM_LAGS + 1):
        minute_features[f"lag_volume_{lag}"] = volume.shift(lag).fillna(0.0)

    day_index = minute_features.index.normalize()
    minute_features["session_day"] = day_index
    daily_volume_series = volume.groupby(day_index).sum()
    daily_features = pd.DataFrame(index=daily_volume_series.index)
    daily_features.index.name = "session_day"
    daily_features["daily_volume"] = daily_volume_series.shift(1)
    prev_daily_volume = daily_volume_series.shift(1)
    daily_features["rolling_volume_mean"] = prev_daily_volume.rolling(7, min_periods=1).mean()
    daily_features["rolling_volume_std"] = (
        prev_daily_volume.rolling(7, min_periods=2).std().fillna(0.0)
    )

    minute_features = minute_features.join(daily_features, on="session_day")
    minute_features = minute_features.drop(columns=["session_day"])
    minute_features["daily_volume"] = minute_features["daily_volume"].fillna(0.0)
    minute_features["rolling_volume_mean"] = minute_features["rolling_volume_mean"].fillna(0.0)
    minute_features["rolling_volume_std"] = minute_features["rolling_volume_std"].fillna(0.0)
    minute_features["cumulative_volume"] = volume.groupby(day_index).cumsum()

    minute_of_day = minute_features.index.hour * 60 + minute_features.index.minute
    minute_features["time_of_day"] = minute_of_day

    # Crypto trades 24/7, so these session dummies are UTC-day terciles.
    # minute_features["opening_dummy"] = (minute_of_day < 8 * 60).astype(np.int8)
    # minute_features["midday_dummy"] = (
    #     (minute_of_day >= 8 * 60) & (minute_of_day < 16 * 60)
    # ).astype(np.int8)
    # minute_features["close_dummy"] = (minute_of_day >= 16 * 60).astype(np.int8)
    #
    # theta = 2 * np.pi * minute_of_day / MINUTES_PER_DAY
    # minute_features["sin_time"] = np.sin(theta)
    # minute_features["cos_time"] = np.cos(theta)

    minute_features["trade_rate"] = minute_features["num_trades"] / 60.0

    total_flow = minute_features["buy_volume"] + minute_features["sell_volume"]
    minute_features["trade_imbalance"] = np.where(
        total_flow > 0,
        (minute_features["buy_volume"] - minute_features["sell_volume"]) / total_flow,
        0.0,
    )

    minute_features["return"] = minute_features["close_price"].pct_change().fillna(0.0)
    minute_features["abs_return"] = minute_features["return"].abs()
    minute_features["realized_volatility"] = np.sqrt(minute_features["realized_var"])
    minute_features["price_range"] = (
        minute_features["high_price"] - minute_features["low_price"]
    )
    minute_features["momentum"] = minute_features["close_price"].pct_change(60).fillna(0.0)

    minute_features = minute_features.drop(columns=["realized_var"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    minute_features.to_parquet(output_path)
    return output_path


def main() -> None:
    output_path = build_features()
    print(output_path.resolve())


if __name__ == "__main__":
    main()
