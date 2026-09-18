"""실적 발표일 수집 (yfinance Ticker.get_earnings_dates). '재료(실적) 있는 급등' 과 '이유 없는 급등' 을 나누기 위한 데이터.

  python scripts/fetch_earnings.py --symbols symbols.txt   -> data/cache/earnings_dates.parquet (symbol, date)
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import data  # noqa: E402

OUT = data.CACHE_DIR / "earnings_dates.parquet"


def fetch_one(sym: str) -> list[tuple[str, pd.Timestamp]]:
    import yfinance as yf

    try:
        ed = yf.Ticker(sym).get_earnings_dates(limit=30)
    except Exception:
        return []
    if ed is None or len(ed) == 0:
        return []
    return [(sym, pd.Timestamp(d).tz_localize(None).normalize()) for d in ed.index]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    symbols = [s.strip() for s in Path(args.symbols).read_text().split() if s.strip()]
    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, s): s for s in symbols}
        for f in as_completed(futs):
            rows.extend(f.result())
            done += 1
            if done % 200 == 0:
                logging.info("earnings: %d/%d symbols, %d dates", done, len(symbols), len(rows))
    df = pd.DataFrame(rows, columns=["symbol", "date"]).drop_duplicates().sort_values(["symbol", "date"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    logging.info("saved %d rows for %d symbols -> %s", len(df), df["symbol"].nunique(), OUT)


if __name__ == "__main__":
    main()
