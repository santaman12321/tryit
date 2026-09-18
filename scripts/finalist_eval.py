"""최종 후보 전략들을 전체 기간에 돌려 자산곡선을 저장하고, ETF 변동성 타게팅과의 50/50 블렌드까지 비교한다.

  python scripts/finalist_eval.py --out reports/best_model
후보는 아래 FINALISTS 에 (이름, 전략, 유니버스, 파라미터, 최대종목수) 로 적는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import data  # noqa: E402
from quant.backtest import BacktestConfig, run_backtest  # noqa: E402
from quant.metrics import summarize  # noqa: E402
from quant.strategies import make_strategy  # noqa: E402

START, END = "2024-01-02", "2026-09-17"
COST = 0.0025 + 0.0005


def etf_voltarget(target=0.30) -> pd.Series:
    """ETF 변동성 타게팅 30% + 200일선 (TQQQ 근사, 주간 조정) 일별 수익률 (scripts/etf_strategies.py 와 동일 규칙)."""
    px = pd.read_parquet("data/cache/etf_close.parquet")
    q = px["QQQ"].dropna()
    qr = q.pct_change()
    rv = qr.rolling(20).std() * np.sqrt(252)
    expo = (target / rv).clip(upper=3.0).where(q > q.rolling(200).mean(), 0.0)
    expo_w = expo.resample("W-FRI").last().reindex(q.index, method="ffill").fillna(0.0)
    w = expo_w / 3.0
    lev = 3 * qr - 0.01 / 252
    cash = px["BIL"].pct_change().reindex(q.index).fillna(0.0)
    r = (w.shift(1).fillna(0) * lev).fillna(0.0) - (w - w.shift(1)).abs().shift(1).fillna(0) * COST + (1 - w.shift(1).fillna(0)).clip(lower=0) * cash
    return r


def yearly(r: pd.Series) -> dict:
    return {str(y): float(v) for y, v in ((1 + r).groupby(r.index.year).prod() - 1).items()}


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    years = len(r) / 252
    return {"total": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / years) - 1), "mdd": float((eq / eq.cummax() - 1).min()),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"), **yearly(r)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalists", required=True, help="JSON 파일: [{name, strategy, universe, params, max_positions}]")
    ap.add_argument("--out", default="reports/best_model")
    args = ap.parse_args()
    out = Path(args.out)
    finalists = json.loads(Path(args.finalists).read_text())
    panels_cache = {}
    curves = {}
    rows = []
    for fz in finalists:
        uni_name = fz["universe"]
        if uni_name not in panels_cache:
            uni = data.load_universe(uni_name)
            panels_cache[uni_name] = data.split_benchmark(data.load_panels(universe=uni))
        stocks, bench = panels_cache[uni_name]
        strat = make_strategy(fz["strategy"], **fz.get("params", {}))
        cfg = BacktestConfig(start=START, end=END, max_positions=fz.get("max_positions", 10), commission_pct=0.0025)
        res = run_backtest(stocks, strat.generate(stocks, bench), strat, cfg)
        r = res.equity.pct_change().fillna(0.0)
        curves[fz["name"]] = r
        s = summarize(res.equity, res.trades, res.n_positions, bench["close"])
        rows.append({"name": fz["name"], **stats(r), "win_rate": s["win_rate"], "n_trades": s["n_trades"], "avg_hold": s["avg_hold_days"]})
        res.equity.to_csv(out / f"equity_{fz['name']}.csv")
        res.trades.to_csv(out / f"trades_{fz['name']}.csv", index=False)
    etf = etf_voltarget(0.30).reindex(pd.DatetimeIndex(sorted(set().union(*[c.index for c in curves.values()]))))
    etf = etf.loc[START:END].fillna(0.0)
    curves["ETF 변동성타게팅30%+200일선"] = etf
    rows.append({"name": "ETF 변동성타게팅30%+200일선", **stats(etf), "win_rate": float("nan"), "n_trades": 0, "avg_hold": float("nan")})
    spy = pd.read_parquet("data/cache/etf_close.parquet")["SPY"].pct_change().loc[START:END]
    rows.append({"name": "SPY 바이앤홀드", **stats(spy.fillna(0)), "win_rate": float("nan"), "n_trades": 0, "avg_hold": float("nan")})
    for fz in finalists:
        r1 = curves[fz["name"]].reindex(etf.index).fillna(0.0)
        blend = 0.5 * r1 + 0.5 * etf  # 일별 리밸런스 근사
        rows.append({"name": f"블렌드 50% {fz['name']} + 50% ETF", **stats(blend), "win_rate": float("nan"), "n_trades": 0, "avg_hold": float("nan")})
        curves[f"블렌드 {fz['name']}"] = blend
    df = pd.DataFrame(rows)
    df.to_csv(out / "finalists.csv", index=False)
    ycols = [c for c in df.columns if c.isdigit()]
    lines = [f"# 최종 후보 비교 ({START}~{END}, 국내 증권사 비용)\n", "| 모델 | 총수익 | CAGR | MDD | 샤프 | " + " | ".join(ycols) + " | 승률 | 거래 | 평균보유일 |", "|---|---|---|---|---|" + "---|" * len(ycols) + "---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.name} | {r.total * 100:+.1f}% | {r.cagr * 100:+.1f}% | {r.mdd * 100:.1f}% | {r.sharpe:.2f} | " + " | ".join(f"{getattr(r, '_' + str(i + 6)) * 100:+.1f}%" if False else f"{df.loc[r.Index, c] * 100:+.1f}%" for i, c in enumerate(ycols))
                     + f" | {'-' if pd.isna(r.win_rate) else f'{r.win_rate * 100:.1f}%'} | {int(r.n_trades)} | {'-' if pd.isna(r.avg_hold) else f'{r.avg_hold:.0f}'} |")
    (out / "finalists.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, r in curves.items():
        ax.plot(r.index, (1 + r).cumprod().values, label=name, lw=1.4)
    ax.plot(spy.index, (1 + spy.fillna(0)).cumprod().values, label="SPY", color="gray", ls="--")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Finalists, net of Korean-broker costs, {START}~{END}")
    fig.tight_layout()
    fig.savefig(out / "finalists.png", dpi=110)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
