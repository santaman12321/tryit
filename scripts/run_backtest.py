"""전략 백테스트 실행 + 리포트 생성.

예)  python scripts/run_backtest.py --strategy all --start 2025-09-18 --end 2026-09-17 --out reports/2025-09_2026-09
     python scripts/run_backtest.py --strategy pullback --param rsi_entry=15 --param stop_atr_mult=3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import data  # noqa: E402
from quant.backtest import BacktestConfig, run_backtest  # noqa: E402
from quant.metrics import summarize  # noqa: E402
from quant.report import monthly_table, plot_result, reason_table, summary_table  # noqa: E402
from quant.strategies import STRATEGIES, make_strategy  # noqa: E402

SUMMARY_KEYS = [
    "start", "end", "total_return", "cagr", "sharpe", "max_drawdown", "max_dd_days", "exposure", "avg_positions",
    "n_trades", "win_rate", "avg_win", "avg_loss", "payoff", "profit_factor", "expectancy", "avg_hold_days",
    "best_trade", "worst_trade", "bench_total_return", "bench_max_drawdown", "bench_sharpe",
]


def parse_param(s: str):
    k, v = s.split("=", 1)
    try:
        v = json.loads(v)
    except json.JSONDecodeError:
        pass
    return k, v


def run_one(name: str, params: dict, panels, bench, config: BacktestConfig):
    strat = make_strategy(name, **params)
    signals = strat.generate(panels, bench)
    result = run_backtest(panels, signals, strat, config)
    summary = summarize(result.equity, result.trades, result.n_positions, bench["close"])
    return strat, result, summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="all", help="all | " + ",".join(STRATEGIES))
    p.add_argument("--universe", default="all", choices=["all", "sp1500"], help="all=미국 전체 보통주, sp1500=S&P 1500")
    p.add_argument("--start", default="2025-09-18")
    p.add_argument("--end", default="2026-09-17")
    p.add_argument("--param", action="append", default=[], help="k=v (JSON 값), 단일 전략에만 적용")
    p.add_argument("--cash", type=float, default=100_000)
    p.add_argument("--max-positions", type=int, default=10)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--commission-pct", type=float, default=0.0025, help="편도 정률 수수료 (국내 증권사 해외주식 기본 0.25%%)")
    p.add_argument("--penny-slippage-bps", type=float, default=50.0, help="체결가 < $5 종목의 슬리피지")
    p.add_argument("--commission", type=float, default=0.0)
    p.add_argument("--sizing", default="equal", choices=["equal", "risk"])
    p.add_argument("--risk-per-trade", type=float, default=0.01)
    p.add_argument("--max-adv-pct", type=float, default=0.01, help="포지션 <= 20일 평균 거래대금 * 비율 (0=제한 없음)")
    p.add_argument("--out", default=None, help="리포트 디렉토리 (기본 reports/<start>_<end>)")
    p.add_argument("--title", default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    names = list(STRATEGIES) if args.strategy == "all" else args.strategy.split(",")
    params = dict(parse_param(s) for s in args.param)
    out = Path(args.out or f"reports/{args.start}_{args.end}")
    out.mkdir(parents=True, exist_ok=True)

    logging.info("loading panels (%s) ...", args.universe)
    uni = data.load_universe(args.universe)
    panels = data.load_panels(universe=uni)
    stocks, bench = data.split_benchmark(panels)
    logging.info("%d symbols, %s ~ %s", stocks["close"].shape[1], stocks["close"].index[0].date(), stocks["close"].index[-1].date())
    config = BacktestConfig(
        initial_cash=args.cash, max_positions=args.max_positions, slippage_bps=args.slippage_bps,
        penny_slippage_bps=args.penny_slippage_bps, commission=args.commission, commission_pct=args.commission_pct, sizing=args.sizing,
        risk_per_trade=args.risk_per_trade, max_adv_pct=args.max_adv_pct or None, start=args.start, end=args.end,
    )

    summaries, sections = {}, []
    for name in names:
        strat, result, summary = run_one(name, params if len(names) == 1 else {}, stocks, bench, config)
        name = strat.label
        summaries[name] = summary
        result.trades.to_csv(out / f"{name}_trades.csv", index=False)
        result.equity.to_frame().join(result.n_positions).to_csv(out / f"{name}_equity.csv")
        plot_result(result, bench["close"], out / f"{name}_equity.png", title=f"{name} {args.start}~{args.end}")
        logging.info("%s: trades=%d win=%.1f%% ret=%.1f%% mdd=%.1f%% pf=%.2f", name, summary["n_trades"],
                     summary["win_rate"] * 100, summary["total_return"] * 100, summary["max_drawdown"] * 100, summary["profit_factor"])
        sections.append(
            f"## {name}\n\n파라미터: `{json.dumps(strat.params, ensure_ascii=False)}`\n\n"
            f"![{name}]({name}_equity.png)\n\n### 청산 사유별\n\n{reason_table(result)}\n\n### 월별 수익률\n\n{monthly_table(result)}\n"
        )

    title = args.title or f"백테스트 {args.start} ~ {args.end}"
    md = [
        f"# {title}\n",
        f"유니버스 {args.universe} ({stocks['close'].shape[1]}종목) · 초기자금 ${config.initial_cash:,.0f} · 최대 {config.max_positions}종목 동일비중 · "
        f"슬리피지 {config.slippage_bps}bp/체결(주가<$5: {config.penny_slippage_bps}bp) · 수수료 편도 {config.commission_pct * 100:.2f}% + ${config.commission}/체결 · "
        f"포지션 상한 = 20일 평균거래대금의 {(config.max_adv_pct or 0) * 100:.0f}%\n",
        "## 전략 비교\n",
        summary_table(summaries, SUMMARY_KEYS),
        "",
        *sections,
    ]
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("report -> %s", out / "README.md")


if __name__ == "__main__":
    main()
