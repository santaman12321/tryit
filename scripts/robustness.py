"""토너먼트 리더보드 상위 모델 강건성 점검: 선택된 설정을 다시 돌려 거래 기록·자산곡선을 저장하고
이상치 의존도(상위 10건 손익 비중), 중앙값 거래, 검증 구간 전/후반 분할, 월별 수익률을 계산한다.

  python scripts/robustness.py --top 6 --out reports/surge_study/tournament
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import data  # noqa: E402
from quant.backtest import BacktestConfig, run_backtest  # noqa: E402
from quant.metrics import monthly_returns, summarize  # noqa: E402
from quant.report import plot_result  # noqa: E402
from quant.screens import SCREENS, compute_features, earnings_mask  # noqa: E402
from quant.strategies import make_strategy  # noqa: E402
from scripts.tournament import TEST, TRAIN, EntryStrategy, signals_for  # noqa: E402


def build(row, stocks, bench, feats, em):
    params = json.loads(row["params"])
    if row["kind"] == "strategy":
        strat = make_strategy(row["model"], **params)
        return strat, strat.generate(stocks, bench)
    fn, mode = SCREENS[row["model"]]
    entry = fn(stocks, feats, **params)
    if row["exclude_earnings"] and em is not None:
        entry = entry & ~em
    ex = json.loads(row["exit"])
    sig = signals_for(entry, feats, mode, stocks, ex.get("stop_pct"), ex.get("target_pct"))
    hold_mode = {"next_open": "swing", "close": "overnight", "open": "intraday"}[mode]
    return EntryStrategy(row["model"], sig, hold_mode, max_hold_days=ex.get("max_hold_days")), sig


def concentration(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"top10_share": float("nan"), "median_trade": float("nan"), "pnl_wo_top10": float("nan")}
    pnl = trades["pnl"].sort_values(ascending=False)
    total = pnl.sum()
    top10 = pnl.head(10).sum()
    return {"top10_share": float(top10 / total) if total > 0 else float("nan"), "median_trade": float(trades["ret_pct"].median()),
            "pnl_wo_top10": float(total - top10)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--out", default="reports/surge_study/tournament")
    p.add_argument("--commission-pct", type=float, default=0.0025)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    lb = pd.read_csv(out / "leaderboard.csv").head(args.top)
    uni = data.load_universe("all")
    stocks, bench = data.split_benchmark(data.load_panels(universe=uni))
    feats = compute_features(stocks)
    em = earnings_mask(stocks)
    mid = "2026-03-15"
    rows = []
    md = ["# 상위 모델 강건성 점검\n", f"검증 구간 전반 {TEST[0]}~{mid} / 후반 {mid}~{TEST[1]} 분할, 상위 10건 손익 비중(이상치 의존), 중앙값 거래.\n",
          "| 모델 | 구간 | 거래 | 승률 | 기대/거래 | 중앙값 거래 | 수익률 | MDD | 상위10건 손익비중 | 상위10건 제외 손익(USD) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, row in lb.iterrows():
        strat, sig = build(row, stocks, bench, feats, em)
        for wname, w in (("학습", TRAIN), ("검증", TEST), ("검증 전반", (TEST[0], mid)), ("검증 후반", (mid, TEST[1]))):
            cfg = BacktestConfig(start=w[0], end=w[1], commission_pct=args.commission_pct)
            res = run_backtest(stocks, sig, strat, cfg)
            s = summarize(res.equity, res.trades, res.n_positions, bench["close"])
            c = concentration(res.trades)
            md.append(f"| {row['model']} | {wname} | {s['n_trades']} | {s['win_rate'] * 100:.1f}% | {s['expectancy'] * 100:+.2f}% | {c['median_trade'] * 100:+.2f}% | "
                      f"{s['total_return'] * 100:+.1f}% | {s['max_drawdown'] * 100:.1f}% | {c['top10_share'] * 100:.0f}% | {c['pnl_wo_top10']:,.0f} |")
            rows.append({"model": row["model"], "window": wname, **s, **c})
            if wname == "검증":
                res.trades.to_csv(out / f"trades_{row['model']}_test.csv", index=False)
                plot_result(res, bench["close"], out / f"equity_{row['model']}_test.png", title=f"{row['model']} (선택 설정) 검증 {TEST[0]}~{TEST[1]}")
                m = monthly_returns(res.equity)
                md_month = " ".join(f"{d.strftime('%y-%m')}:{v * 100:+.0f}%" for d, v in m.items())
                rows[-1]["monthly"] = md_month
        logging.info("%s done", row["model"])
    md.append("\n## 검증 구간 월별 수익률\n")
    for r in rows:
        if r["window"] == "검증":
            md.append(f"- **{r['model']}**: {r['monthly']}")
    pd.DataFrame(rows).to_csv(out / "robustness.csv", index=False)
    (out / "robustness.md").write_text("\n".join(md), encoding="utf-8")
    logging.info("-> %s", out / "robustness.md")


if __name__ == "__main__":
    main()
