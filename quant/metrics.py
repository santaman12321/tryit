"""성과 지표."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown_duration(equity: pd.Series) -> int:
    """고점 회복까지 걸린 최장 거래일 수."""
    peak = equity.cummax()
    under = equity < peak
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return int(longest)


def summarize(equity: pd.Series, trades: pd.DataFrame, n_positions: pd.Series | None = None, benchmark: pd.Series | None = None) -> dict:
    eq = equity.dropna()
    ret = eq.pct_change().dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    years = max(len(eq) / TRADING_DAYS, 1e-9)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    vol = ret.std() * math.sqrt(TRADING_DAYS) if len(ret) > 1 else float("nan")
    sharpe = ret.mean() / ret.std() * math.sqrt(TRADING_DAYS) if len(ret) > 1 and ret.std() > 0 else float("nan")
    downside = ret[ret < 0].std() * math.sqrt(TRADING_DAYS) if (ret < 0).sum() > 1 else float("nan")
    sortino = ret.mean() * TRADING_DAYS / downside if downside and downside > 0 else float("nan")
    dd = drawdown(eq)
    mdd = float(dd.min())
    out = {
        "start": eq.index[0].date().isoformat(),
        "end": eq.index[-1].date().isoformat(),
        "trading_days": int(len(eq)),
        "total_return": float(total),
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": mdd,
        "max_dd_days": max_drawdown_duration(eq),
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"),
        "exposure": float((n_positions > 0).mean()) if n_positions is not None else float("nan"),
        "avg_positions": float(n_positions.mean()) if n_positions is not None else float("nan"),
    }
    out.update(trade_stats(trades))
    if benchmark is not None:
        b = benchmark.reindex(eq.index).ffill().dropna()
        if len(b) > 1:
            b_total = b.iloc[-1] / b.iloc[0] - 1.0
            b_ret = b.pct_change().dropna()
            out["bench_total_return"] = float(b_total)
            out["bench_cagr"] = float((1 + b_total) ** (1 / years) - 1)
            out["bench_max_drawdown"] = float(drawdown(b).min())
            out["bench_sharpe"] = float(b_ret.mean() / b_ret.std() * math.sqrt(TRADING_DAYS)) if b_ret.std() > 0 else float("nan")
    return out


def trade_stats(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": float("nan"), "avg_win": float("nan"), "avg_loss": float("nan"),
                "payoff": float("nan"), "profit_factor": float("nan"), "expectancy": float("nan"),
                "avg_hold_days": float("nan"), "best_trade": float("nan"), "worst_trade": float("nan"),
                "net_pnl": 0.0}
    r = trades["ret_pct"]
    wins, losses = r[r > 0], r[r <= 0]
    gross_win = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    return {
        "n_trades": int(n),
        "win_rate": float(len(wins) / n),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "payoff": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else float("nan"),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "expectancy": float(r.mean()),
        "avg_hold_days": float(trades["hold_days"].mean()),
        "best_trade": float(r.max()),
        "worst_trade": float(r.min()),
        "net_pnl": float(trades["pnl"].sum()),
    }


def monthly_returns(equity: pd.Series) -> pd.Series:
    m = equity.resample("ME").last()
    first = equity.iloc[0]
    prev = m.shift(1).fillna(first)
    return (m / prev - 1.0).rename("monthly_return")


def exit_reason_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby("reason")["ret_pct"]
    return pd.DataFrame({"n": g.size(), "win_rate": g.apply(lambda s: (s > 0).mean()), "avg_ret": g.mean()})
