"""토너먼트 상위/생존 설정의 강건성 점검.

후보: (a) 리더보드(학습 선택) 상위 N, (b) 모든 설정 중 두 기간 모두 기대값 > 0 인 것 중 검증 기대값 상위, (c) 그중 두 기간 승률 ≥ 50% 상위.
각 후보를 학습 / 검증 / 검증 전반 / 검증 후반 에 다시 돌리고, 익절이 있는 설정은 체결 가정 변형(고가 버퍼 3%, 종가 확인)도 돌린다.
이상치 의존도 = 상위 10건 손익이 전체 손익에서 차지하는 비중.

  python scripts/robustness.py --out reports/surge_study/tournament
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

MID = "2026-03-15"


def build(row, stocks, bench, feats, em):
    params = json.loads(row["params"])
    if row["kind"] == "strategy":
        strat = make_strategy(row["model"], **params)
        return strat, strat.generate(stocks, bench), None
    fn, mode = SCREENS[row["model"]]
    entry = fn(stocks, feats, **params)
    if row["exclude_earnings"] and em is not None:
        entry = entry & ~em
    ex = json.loads(row["exit"])
    sig = signals_for(entry, feats, mode, stocks, ex.get("stop_pct"), ex.get("target_pct"))
    hold_mode = {"next_open": "swing", "close": "overnight", "open": "intraday"}[mode]
    return EntryStrategy(row["model"], sig, hold_mode, max_hold_days=ex.get("max_hold_days")), sig, ex.get("target_pct")


def concentration(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"top10_share": float("nan"), "median_trade": float("nan"), "pnl_wo_top10": float("nan"), "target_share": float("nan")}
    pnl = trades["pnl"].sort_values(ascending=False)
    total = pnl.sum()
    return {"top10_share": float(pnl.head(10).sum() / total) if total > 0 else float("nan"), "median_trade": float(trades["ret_pct"].median()),
            "pnl_wo_top10": float(total - pnl.head(10).sum()), "target_share": float((trades["reason"] == "target").mean())}


def candidates(out: Path, top: int) -> pd.DataFrame:
    lb = pd.read_csv(out / "leaderboard.csv").head(top)
    lb["source"] = "리더보드(학습 선택)"
    runs = pd.read_csv(out / "runs_all.csv")
    both = runs[(runs.train_n_trades >= 60) & (runs.test_n_trades >= 60) & (runs.train_expectancy > 0) & (runs.test_expectancy > 0)].copy()
    a = both.sort_values("test_expectancy", ascending=False).head(4).copy()
    a["source"] = "두 기간 + (검증 기대값 상위)"
    hw = both[(both.train_win_rate >= 0.5) & (both.test_win_rate >= 0.5)].copy()
    hw["minwin"] = hw[["train_win_rate", "test_win_rate"]].min(axis=1)
    b = hw.sort_values(["minwin", "test_expectancy"], ascending=False).head(4).copy()
    b["source"] = "두 기간 승률 ≥ 50%"
    cand = pd.concat([lb, a, b], ignore_index=True)
    cand = cand.drop_duplicates(subset=["model", "params", "exit", "exclude_earnings"]).reset_index(drop=True)
    return cand


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--out", default="reports/surge_study/tournament")
    p.add_argument("--commission-pct", type=float, default=0.0025)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    cand = candidates(out, args.top)
    uni = data.load_universe("all")
    stocks, bench = data.split_benchmark(data.load_panels(universe=uni))
    feats = compute_features(stocks)
    em = earnings_mask(stocks)
    windows = (("학습", TRAIN), ("검증", TEST), ("검증 전반", (TEST[0], MID)), ("검증 후반", (MID, TEST[1])))
    rows = []
    md = ["# 상위/생존 설정 강건성 점검\n",
          f"검증 구간을 전반({TEST[0]}~{MID})/후반({MID}~{TEST[1]})으로 나눠 재확인. 익절이 있는 설정은 체결 가정 변형: "
          "'버퍼3%' = 고가가 목표가보다 3% 더 높아야 체결 인정, '종가확인' = 종가가 목표가 이상일 때만 체결. 상위10 비중 = 상위 10건 손익 / 전체 손익.\n"]
    for ci, row in cand.iterrows():
        strat, sig, target = build(row, stocks, bench, feats, em)
        md.append(f"## {ci + 1}. {row['model']} — {row['source']}\n\n파라미터 `{row['params']}` · 청산 `{row['exit']}` · 재료제외 {'예' if row['exclude_earnings'] else '아니오'}\n\n"
                  "| 변형 | 구간 | 거래 | 승률 | 기대/거래 | 중앙값 | 수익률 | MDD | 상위10 비중 | 익절 비율 |\n|---|---|---|---|---|---|---|---|---|---|")
        variants = [("기본", {})]
        if target:
            variants += [("버퍼3%", {"target_buffer": 0.03}), ("종가확인", {"target_at_close": True})]
        for vname, vkw in variants:
            for wname, w in windows:
                if vname != "기본" and wname.startswith("검증 "):
                    continue
                cfg = BacktestConfig(start=w[0], end=w[1], commission_pct=args.commission_pct, **vkw)
                res = run_backtest(stocks, sig, strat, cfg)
                s = summarize(res.equity, res.trades, res.n_positions, bench["close"])
                c = concentration(res.trades)
                md.append(f"| {vname} | {wname} | {s['n_trades']} | {s['win_rate'] * 100:.1f}% | {s['expectancy'] * 100:+.2f}% | {c['median_trade'] * 100:+.2f}% | "
                          f"{s['total_return'] * 100:+.1f}% | {s['max_drawdown'] * 100:.1f}% | {c['top10_share'] * 100:.0f}% | {c['target_share'] * 100:.0f}% |")
                rows.append({"cand": ci + 1, "model": row["model"], "source": row["source"], "params": row["params"], "exit": row["exit"],
                             "exclude_earnings": row["exclude_earnings"], "variant": vname, "window": wname, **s, **c})
                if vname == "기본" and wname == "검증":
                    tag = f"{ci + 1}_{row['model']}"
                    res.trades.to_csv(out / f"trades_{tag}_test.csv", index=False)
                    plot_result(res, bench["close"], out / f"equity_{tag}_test.png", title=f"{row['model']} 검증 {TEST[0]}~{TEST[1]}")
                    m = monthly_returns(res.equity)
                    rows[-1]["monthly"] = " ".join(f"{d.strftime('%y-%m')}:{v * 100:+.0f}%" for d, v in m.items())
        md.append("")
        logging.info("candidate %d %s done", ci + 1, row["model"])
    md.append("## 검증 구간 월별 수익률 (기본 변형)\n")
    for r in rows:
        if r.get("monthly"):
            md.append(f"- **{r['cand']}. {r['model']}**: {r['monthly']}")
    pd.DataFrame(rows).to_csv(out / "robustness.csv", index=False)
    (out / "robustness.md").write_text("\n".join(md), encoding="utf-8")
    logging.info("-> %s", out / "robustness.md")


if __name__ == "__main__":
    main()
