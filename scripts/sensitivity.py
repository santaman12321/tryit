"""파라미터 민감도(강건성) 분석: 격자의 모든 조합을 여러 기간(window)에 돌려 표로 만든다.

예)  python scripts/sensitivity.py --strategy formula --base '{"screen":"ross_5pillars"}' \\
        --grid '{"stop_pct":[0.05,0.1],"target_pct":[null,0.3],"max_hold_days":[2,5]}' \\
        --window train=2023-11-01:2025-09-17 --window test=2025-09-18:2026-09-17 --out reports/surge_study/grid_ross
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import data  # noqa: E402
from quant.backtest import BacktestConfig, run_backtest  # noqa: E402
from quant.metrics import summarize  # noqa: E402
from quant.strategies import make_strategy  # noqa: E402

COLS = ["n_trades", "win_rate", "total_return", "max_drawdown", "profit_factor", "expectancy", "sharpe"]


def run_grid(name: str, grid: dict, base: dict, stocks, bench, windows: dict, cfg_kw: dict) -> pd.DataFrame:
    keys = list(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = {**base, **dict(zip(keys, values))}
        strat = make_strategy(name, **params)
        signals = strat.generate(stocks, bench)
        row = {k: params[k] for k in keys}
        for wname, (ws, we) in windows.items():
            config = BacktestConfig(start=ws, end=we, **cfg_kw)
            res = run_backtest(stocks, signals, strat, config)
            summ = summarize(res.equity, res.trades, res.n_positions)
            row.update({f"{wname}_{c}": summ[c] for c in COLS})
            logging.info("%s [%s] trades=%d win=%.1f%% ret=%.1f%% mdd=%.1f%% pf=%.2f", dict(zip(keys, values)), wname, summ["n_trades"],
                         summ["win_rate"] * 100, summ["total_return"] * 100, summ["max_drawdown"] * 100, summ["profit_factor"])
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, keys: list[str], windows: list[str]) -> str:
    head = keys + [f"{w} {c}" for w in windows for c in ["거래", "승률", "수익률", "MDD", "PF", "기대/거래"]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in df.iterrows():
        vals = [str(r[k]) for k in keys]
        for w in windows:
            vals += [f"{int(r[f'{w}_n_trades'])}", f"{r[f'{w}_win_rate'] * 100:.1f}%", f"{r[f'{w}_total_return'] * 100:+.1f}%",
                     f"{r[f'{w}_max_drawdown'] * 100:.1f}%", f"{r[f'{w}_profit_factor']:.2f}", f"{r[f'{w}_expectancy'] * 100:+.2f}%"]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True)
    p.add_argument("--universe", default="all", choices=["all", "sp1500"])
    p.add_argument("--window", action="append", default=[], help="name=start:end (여러 번)")
    p.add_argument("--grid", required=True, help='JSON: {"param": [v1, v2, ...], ...}')
    p.add_argument("--base", default="{}", help="JSON: 고정 파라미터")
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--commission-pct", type=float, default=0.0025, help="편도 정률 수수료 (국내 증권사 해외주식 기본 0.25%%)")
    p.add_argument("--penny-slippage-bps", type=float, default=50.0)
    p.add_argument("--max-positions", type=int, default=10)
    p.add_argument("--max-adv-pct", type=float, default=0.01)
    p.add_argument("--out", required=True, help="출력 경로 접두어 (.csv/.md 생성)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    grid = json.loads(args.grid)
    base = json.loads(args.base)
    windows = {}
    for w in args.window or ["test=2025-09-18:2026-09-17"]:
        name, rng = w.split("=", 1)
        windows[name] = tuple(rng.split(":"))
    uni = data.load_universe(args.universe)
    stocks, bench = data.split_benchmark(data.load_panels(universe=uni))
    cfg_kw = dict(max_positions=args.max_positions, slippage_bps=args.slippage_bps, penny_slippage_bps=args.penny_slippage_bps,
                  commission_pct=args.commission_pct, max_adv_pct=args.max_adv_pct or None)
    df = run_grid(args.strategy, grid, base, stocks, bench, windows, cfg_kw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    md = (f"# {args.strategy} 민감도 ({args.universe})\n\n고정: `{json.dumps(base, ensure_ascii=False)}` · 기간: "
          + ", ".join(f"{k}={v[0]}~{v[1]}" for k, v in windows.items()) + f" · 최대 {args.max_positions}종목, 슬리피지 {args.slippage_bps}/{args.penny_slippage_bps}bp, 수수료 편도 {args.commission_pct * 100:.2f}%\n\n"
          + to_markdown(df, list(grid), list(windows)))
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    logging.info("-> %s", out.with_suffix(".md"))


if __name__ == "__main__":
    main()
