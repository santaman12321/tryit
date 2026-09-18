"""유니버스 + 일봉 데이터 수집.

사용:  python scripts/fetch_data.py --start 2023-06-01 --end 2026-09-18
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import data  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-06-01")
    p.add_argument("--end", default=None, help="exclusive; default=today")
    p.add_argument("--universe", default="sp1500", choices=["sp1500", "all"])
    p.add_argument("--indices", default="sp500,sp400,sp600")
    p.add_argument("--refresh-universe", action="store_true", help="유니버스 목록을 웹에서 다시 받는다")
    p.add_argument("--full", action="store_true", help="이미 받은 종목도 다시 받는다 (기본: 없는 종목만 추가)")
    p.add_argument("--limit", type=int, default=None, help="테스트용: 앞에서 N 종목만")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.universe == "sp1500":
        if args.refresh_universe or not data.UNIVERSE_FILE.exists():
            data.save_universe(data.fetch_universe(tuple(args.indices.split(","))), data.UNIVERSE_FILE)
    else:
        if args.refresh_universe or not data.UNIVERSE_ALL_FILE.exists():
            data.save_universe(data.fetch_universe_all(), data.UNIVERSE_ALL_FILE)
    uni = data.load_universe(args.universe)
    logging.info("universe %s: %d symbols", args.universe, len(uni))
    symbols = uni["symbol"].tolist()
    if args.limit:
        symbols = symbols[: args.limit]
    symbols = [data.BENCHMARK] + symbols

    existing = data.load_prices_long() if data.PRICES_FILE.exists() else None
    if existing is not None and not args.full:
        have = set(existing["symbol"].unique())
        symbols = [s for s in symbols if s not in have]
        logging.info("%d symbols already cached, %d to download", len(have), len(symbols))
        if not symbols:
            return

    end = args.end
    if end is None:
        import datetime as dt

        end = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    long_df = data.download_prices(symbols, args.start, end)
    if existing is not None and not args.full:
        long_df = data.merge_prices(existing, long_df)
    data.save_prices(long_df)
    n_sym = long_df["symbol"].nunique()
    logging.info("saved %d rows, %d symbols, %s ~ %s -> %s", len(long_df), n_sym, long_df["date"].min().date(), long_df["date"].max().date(), data.PRICES_FILE)


if __name__ == "__main__":
    main()
