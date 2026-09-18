"""분봉/시간봉 수집. Yahoo 제공 범위: 1h = 최근 730일, 15m/5m/30m = 최근 60일, 1m = 최근 7일.

  python scripts/fetch_intraday.py --symbols symbols.txt --interval 1h
  python scripts/fetch_intraday.py --symbols symbols.txt --interval 15m
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import data  # noqa: E402

def download_bars(symbols: list[str], interval: str = "1h", period: str = "730d", batch_size: int = 50, pause: float = 1.0) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i : i + batch_size]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(chunk, period=period, interval=interval, auto_adjust=True, group_by="ticker", threads=True, progress=False, prepost=False)
                break
            except Exception as e:  # pragma: no cover
                logging.warning("batch %d attempt %d failed: %s", i, attempt, e)
                time.sleep(5 * (attempt + 1))
        if raw is None or raw.empty:
            continue
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([chunk, raw.columns])
        for sym in chunk:
            if sym not in raw.columns.get_level_values(0):
                continue
            df = raw[sym].dropna(how="all")
            if df.empty:
                continue
            df = df.rename(columns=str.lower)[data.FIELDS].copy()
            df["symbol"] = sym
            df.index.name = "datetime"
            frames.append(df.reset_index())
        logging.info("%s: %d/%d symbols", interval, min(i + batch_size, len(symbols)), len(symbols))
        time.sleep(pause)
    out = pd.concat(frames, ignore_index=True)
    dt = pd.to_datetime(out["datetime"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    out["datetime"] = dt
    return out.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True)
    p.add_argument("--interval", default="1h", help="1h (최근 730일) | 15m/5m (최근 60일) | 1m (최근 7일)")
    p.add_argument("--period", default=None, help="기본: 1h=730d, 15m/5m/30m=60d, 1m=7d")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    period = args.period or {"1h": "730d", "1m": "7d"}.get(args.interval, "60d")
    symbols = [s.strip() for s in Path(args.symbols).read_text().split() if s.strip()]
    df = download_bars(symbols, args.interval, period)
    out = data.CACHE_DIR / f"intraday_{args.interval}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logging.info("saved %d rows, %d symbols, %s ~ %s -> %s", len(df), df["symbol"].nunique(), df["datetime"].min(), df["datetime"].max(), out)


if __name__ == "__main__":
    main()
