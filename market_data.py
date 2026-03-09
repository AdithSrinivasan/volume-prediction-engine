from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import databento as db


def load_env_file(env_path: Path) -> None:
    """Populate os.environ from a simple .env file if present."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def get_api_key() -> str:
    env_path = Path(__file__).resolve().parent / ".env"
    load_env_file(env_path)

    for key_name in ("DATABENTO_API_KEY", "DB_API_KEY", "DATABENTO_KEY"):
        api_key = os.getenv(key_name)
        if api_key:
            return api_key

    raise RuntimeError(
        "Missing Databento API key. Set DATABENTO_API_KEY in .env or the environment."
    )


def default_time_range() -> tuple[str, str]:
    """
    Return a trailing 7-day window ending 24 hours ago.

    Databento's Historical API is intended for data older than the last 24 hours,
    so this default avoids requesting too recent a range.
    """
    end_dt = datetime.now(timezone.utc) - timedelta(days=1)
    start_dt = end_dt - timedelta(days=30)
    return start_dt.isoformat(), end_dt.isoformat()


def parse_args() -> argparse.Namespace:
    default_start, default_end = default_time_range()

    parser = argparse.ArgumentParser(
        description="Download Databento MBP-10 data for SPY and save it to CSV."
    )
    parser.add_argument(
        "--dataset",
        default="ARCX.PILLAR",
        help="Databento dataset ID. Defaults to NYSE Arca Integrated for SPY.",
    )
    parser.add_argument(
        "--symbol",
        default="SPY",
        help="Ticker symbol to request.",
    )
    parser.add_argument(
        "--schema",
        default="mbp-10",
        help="Databento schema to request.",
    )
    parser.add_argument(
        "--start",
        default=default_start,
        help="Inclusive UTC start time in ISO-8601 format.",
    )
    parser.add_argument(
        "--end",
        default=default_end,
        help="Exclusive UTC end time in ISO-8601 format.",
    )
    parser.add_argument(
        "--stype-in",
        default="raw_symbol",
        help="Databento input symbology type.",
    )
    parser.add_argument(
        "--output",
        default="spy_mbp10_last_month.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = get_api_key()

    client = db.Historical(api_key)
    data = client.timeseries.get_range(
        dataset=args.dataset,
        schema=args.schema,
        symbols=args.symbol,
        start=args.start,
        end=args.end,
        stype_in=args.stype_in,
    )

    df = data.to_df()
    output_path = Path(args.output)
    df.to_csv(output_path)

    print(
        f"Saved {len(df):,} rows of {args.schema} data for {args.symbol} "
        f"from {args.start} to {args.end} to {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
