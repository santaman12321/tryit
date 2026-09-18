"""이벤트 스터디: 스크린이 켜진 (일자, 종목) 마다 이후 수익률을 계산해 기대값/승률을 본다.

entry_mode
  next_open : t 종가 신호 -> t+1 시가 진입.  ret_oc = t+1 시가->종가(단타), ret_h = t+1 시가 -> t+h 종가(스윙)
  close     : t 종가(마감 직전) 진입 -> ret_on = t+1 시가 (오버나잇), ret_h = t+h 종가
  open      : t 시가 진입(갭) -> ret_oc = t 종가, hi/lo = 당일 고가/저가 대비, ret_h = t+h 종가
모든 수익률은 비용 전. 표에서는 왕복 비용(cost_bps)을 빼 '비용 후 기대값' 도 같이 보여준다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

HORIZONS = (1, 2, 3, 5, 10)


def forward_returns(panels: dict, events: pd.DataFrame, entry_mode: str = "next_open", horizons=HORIZONS,
                    features: dict[str, pd.DataFrame] | None = None, feature_names: list[str] | None = None) -> pd.DataFrame:
    C = panels["close"]
    idx, cols = C.index, list(C.columns)
    ev = events.reindex(index=idx, columns=cols).fillna(False).to_numpy(bool)
    O = panels["open"].reindex(index=idx, columns=cols).to_numpy(float)
    H = panels["high"].reindex(index=idx, columns=cols).to_numpy(float)
    L = panels["low"].reindex(index=idx, columns=cols).to_numpy(float)
    Cn = C.to_numpy(float)
    n = len(idx)
    hmax = max(horizons)
    ti, si = np.nonzero(ev)
    if entry_mode == "next_open":
        keep = ti + 1 + hmax < n
    elif entry_mode == "close":
        keep = ti + hmax < n
    elif entry_mode == "open":
        keep = ti + hmax < n
    else:
        raise ValueError(entry_mode)
    ti, si = ti[keep], si[keep]
    out = {"date": idx[ti], "symbol": np.array(cols, dtype=object)[si]}
    if entry_mode == "next_open":
        e = ti + 1
        px = O[e, si]
        out["entry_price"] = px
        out["ret_gap"] = px / Cn[ti, si] - 1.0  # 신호 종가 -> 진입 시가 (얼마나 갭으로 뛰어 시작하나)
        out["ret_oc"] = Cn[e, si] / px - 1.0
        out["mfe1"] = H[e, si] / px - 1.0
        out["mae1"] = L[e, si] / px - 1.0
        for h in horizons:
            out[f"ret_{h}d"] = Cn[e + h - 1, si] / px - 1.0
        out["mfe5"] = np.nanmax(np.stack([H[e + k, si] for k in range(5)]), axis=0) / px - 1.0
        out["mae5"] = np.nanmin(np.stack([L[e + k, si] for k in range(5)]), axis=0) / px - 1.0
    elif entry_mode == "close":
        px = Cn[ti, si]
        out["entry_price"] = px
        out["ret_on"] = O[ti + 1, si] / px - 1.0
        for h in horizons:
            out[f"ret_{h}d"] = Cn[ti + h, si] / px - 1.0
    else:  # open
        px = O[ti, si]
        out["entry_price"] = px
        out["ret_oc"] = Cn[ti, si] / px - 1.0
        out["mfe1"] = H[ti, si] / px - 1.0
        out["mae1"] = L[ti, si] / px - 1.0
        for h in horizons:
            out[f"ret_{h}d"] = Cn[ti + h - 1, si] / px - 1.0
    df = pd.DataFrame(out)
    if features is not None:
        names = feature_names or list(features)
        for name in names:
            if name in features:
                arr = features[name].reindex(index=idx, columns=cols).to_numpy(float)
                df[name] = arr[ti, si]
    df = df[np.isfinite(df["entry_price"]) & (df["entry_price"] > 0)]
    return df.reset_index(drop=True)


def stats(r: pd.Series, cost=0.0) -> dict:
    """cost: 스칼라 또는 r 과 같은 index 의 Series (왕복 비용, 소수)."""
    r = r.dropna()
    if len(r) == 0:
        return {"n": 0}
    if isinstance(cost, pd.Series):
        cost = cost.reindex(r.index).fillna(0.0)
    net = r - cost
    m, s = float(net.mean()), float(net.std(ddof=1)) if len(net) > 1 else float("nan")
    return {
        "n": int(len(r)),
        "mean": float(r.mean()),
        "mean_net": m,
        "median": float(r.median()),
        "win": float((net > 0).mean()),
        "p_gt10": float((r > 0.10).mean()),
        "p_lt10": float((r < -0.10).mean()),
        "tstat": m / (s / math.sqrt(len(net))) if s and s > 0 else float("nan"),
    }


def summarize_events(df: pd.DataFrame, cols: list[str], cost: float) -> pd.DataFrame:
    rows = {c: stats(df[c], cost) for c in cols if c in df}
    return pd.DataFrame(rows).T


def bucket_stats(df: pd.DataFrame, feature: str, bins, ret_col: str, cost="cost", labels=None) -> pd.DataFrame:
    """feature 를 bins 로 나눠 ret_col 통계. cost 는 스칼라 또는 df 의 컬럼명."""
    if feature not in df or df[feature].notna().sum() == 0:
        return pd.DataFrame()
    cat = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)
    rows = {}
    for k, g in df.groupby(cat, observed=True):
        c = g[cost] if isinstance(cost, str) and cost in g else cost
        rows[str(k)] = stats(g[ret_col], c)
    return pd.DataFrame(rows).T


def fmt_stats_table(tbl: pd.DataFrame, title_col: str = "구간") -> str:
    if tbl.empty:
        return "(이벤트 없음)"
    lines = [f"| {title_col} | n | 평균 | 비용후 평균 | 중앙값 | 승률(비용후) | P(>+10%) | P(<-10%) | t |", "|---|---|---|---|---|---|---|---|---|"]
    for k, r in tbl.iterrows():
        if r.get("n", 0) == 0:
            lines.append(f"| {k} | 0 | - | - | - | - | - | - | - |")
            continue
        lines.append(f"| {k} | {int(r['n'])} | {r['mean'] * 100:+.2f}% | {r['mean_net'] * 100:+.2f}% | {r['median'] * 100:+.2f}% | "
                     f"{r['win'] * 100:.1f}% | {r['p_gt10'] * 100:.1f}% | {r['p_lt10'] * 100:.1f}% | {r['tstat']:.1f} |")
    return "\n".join(lines)
