"""매일 미국장 마감 후(한국시간 아침) 실행: 데이터 갱신 -> 신호 계산 -> 다음 장 주문 계획/제출.

예)  python scripts/run_daily.py --strategy breakout --universe sp1500 --broker dryrun        # 계획만 출력
     python scripts/run_daily.py --strategy breakout --universe sp1500 --broker kis --live    # KIS 모의/실계좌에 제출
cron 예)  0 7 * * 2-6  cd /path/tryit && python scripts/run_daily.py --strategy breakout --universe sp1500 --broker kis --live >> logs/daily.log 2>&1
          (미국장 마감 = 한국시간 05:00/06:00, 그 뒤 07:00 KST 실행)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import data  # noqa: E402
from quant.backtest import BacktestConfig  # noqa: E402
from quant.broker import make_broker  # noqa: E402
from quant.live import apply_plan, load_state, plan_orders, save_state  # noqa: E402
from quant.strategies import make_strategy  # noqa: E402


def update_prices(days: int = 10) -> None:
    """캐시된 모든 심볼의 최근 days 일을 다시 받아 합친다."""
    existing = data.load_prices_long()
    symbols = sorted(existing["symbol"].unique())
    start = (existing["date"].max() - dt.timedelta(days=days)).date().isoformat()
    end = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    new = data.download_prices(symbols, start, end)
    data.save_prices(data.merge_prices(existing, new))
    logging.info("prices updated to %s", new["date"].max().date() if not new.empty else "(no new data)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True)
    p.add_argument("--universe", default="sp1500", choices=["all", "sp1500"])
    p.add_argument("--param", action="append", default=[])
    p.add_argument("--broker", default="dryrun", choices=["dryrun", "kis"])
    p.add_argument("--live", action="store_true", help="실제 주문 제출 (없으면 계획만 출력)")
    p.add_argument("--no-update", action="store_true", help="데이터 갱신 생략")
    p.add_argument("--cash", type=float, default=100_000)
    p.add_argument("--max-positions", type=int, default=10)
    p.add_argument("--max-adv-pct", type=float, default=0.01)
    p.add_argument("--commission-pct", type=float, default=0.0025)
    p.add_argument("--state-dir", default="state")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.no_update:
        update_prices()
    params = {}
    for s in args.param:
        k, v = s.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        params[k] = v
    strategy = make_strategy(args.strategy, **params)
    uni = data.load_universe(args.universe)
    stocks, bench = data.split_benchmark(data.load_panels(universe=uni))
    config = BacktestConfig(initial_cash=args.cash, max_positions=args.max_positions, max_adv_pct=args.max_adv_pct or None, commission_pct=args.commission_pct)
    state_file = Path(args.state_dir) / f"{args.strategy}.json"
    broker = make_broker(args.broker, state_file, args.cash)
    state = load_state(state_file, args.cash)

    plan = plan_orders(strategy, stocks, bench, broker, state, config)
    logging.info("as of %s: %d exits, %d entries", plan.as_of, len(plan.exits), len(plan.entries))
    for n in plan.notes:
        logging.info("note: %s", n)
    apply_plan(plan, broker, state, stocks, dry_run=not args.live)
    save_state(state_file, state)
    logging.info("state -> %s (equity=%s)", state_file, state.get("equity"))


if __name__ == "__main__":
    main()
