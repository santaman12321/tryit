"""차트/마크다운 리포트."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import BacktestResult  # noqa: E402
from .metrics import drawdown, exit_reason_table, monthly_returns  # noqa: E402

PCT_KEYS = {
    "total_return", "cagr", "ann_vol", "max_drawdown", "exposure", "win_rate", "avg_win", "avg_loss",
    "expectancy", "best_trade", "worst_trade", "bench_total_return", "bench_cagr", "bench_max_drawdown",
}
LABELS = {
    "total_return": "총수익률", "cagr": "CAGR", "ann_vol": "연변동성", "sharpe": "샤프", "sortino": "소르티노",
    "max_drawdown": "최대낙폭(MDD)", "max_dd_days": "최장 낙폭기간(일)", "calmar": "칼마", "exposure": "시장노출(일 비율)",
    "avg_positions": "평균 보유종목수", "n_trades": "거래횟수", "win_rate": "승률", "avg_win": "평균 수익(승)",
    "avg_loss": "평균 손실(패)", "payoff": "손익비", "profit_factor": "프로핏팩터", "expectancy": "기대수익/거래",
    "avg_hold_days": "평균 보유일", "best_trade": "최고 거래", "worst_trade": "최악 거래", "net_pnl": "순손익(USD)",
    "bench_total_return": "SPY 총수익률", "bench_cagr": "SPY CAGR", "bench_max_drawdown": "SPY MDD", "bench_sharpe": "SPY 샤프",
    "start": "시작", "end": "종료", "trading_days": "거래일수",
}


def fmt(key: str, v) -> str:
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and (v != v)):
        return "-"
    if key in PCT_KEYS:
        return f"{v * 100:.2f}%"
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        return f"{v:,.2f}"
    return str(v)


def plot_result(result: BacktestResult, benchmark: pd.Series | None, path: Path, title: str | None = None) -> None:
    eq = result.equity / result.equity.iloc[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(eq.index, eq.values, label=f"{result.strategy}", color="#1f77b4", lw=1.6)
    if benchmark is not None:
        b = benchmark.reindex(eq.index).ffill()
        b = b / b.iloc[0]
        ax1.plot(b.index, b.values, label="SPY buy&hold", color="#7f7f7f", lw=1.2, ls="--")
    ax1.set_ylabel("Growth of 1")
    ax1.set_title(title or f"{result.strategy}: equity curve")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    dd = drawdown(result.equity)
    ax2.fill_between(dd.index, dd.values * 100, 0, color="#d62728", alpha=0.4)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summary_table(summaries: dict[str, dict], keys: list[str]) -> str:
    names = list(summaries)
    lines = ["| 지표 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in keys:
        lines.append(f"| {LABELS.get(k, k)} | " + " | ".join(fmt(k, summaries[n].get(k)) for n in names) + " |")
    return "\n".join(lines)


def monthly_table(result: BacktestResult) -> str:
    m = monthly_returns(result.equity)
    lines = ["| 월 | 수익률 |", "|---|---|"]
    for d, v in m.items():
        lines.append(f"| {d.strftime('%Y-%m')} | {v * 100:+.2f}% |")
    return "\n".join(lines)


def reason_table(result: BacktestResult) -> str:
    t = exit_reason_table(result.trades)
    if t.empty:
        return "(거래 없음)"
    lines = ["| 청산 사유 | 건수 | 승률 | 평균수익 |", "|---|---|---|---|"]
    for reason, row in t.iterrows():
        lines.append(f"| {reason} | {int(row['n'])} | {row['win_rate'] * 100:.1f}% | {row['avg_ret'] * 100:+.2f}% |")
    return "\n".join(lines)
