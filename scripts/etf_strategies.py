"""ETF 장기 검증 (2006~): 추세추종·듀얼 모멘텀·변동성 타게팅 레버리지 — '최대한 돈을 버는' 후보의 장기 성적.

  python scripts/etf_strategies.py --out reports/best_model
비용: 리밸런스 시 편도 0.25% 수수료 + 0.05% 슬리피지 (매매 비중 변화분에만 적용).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COST = 0.0025 + 0.0005


def stats(ret: pd.Series, label: str) -> dict:
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    years = len(ret) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    by_year = (1 + ret).groupby(ret.index.year).prod() - 1
    return {"전략": label, "기간": f"{ret.index[0].date()}~{ret.index[-1].date()}", "CAGR": cagr, "MDD": dd, "샤프": sharpe,
            "최근3년": (1 + ret[ret.index >= "2023-09-18"]).prod() - 1, "최근1년": (1 + ret[ret.index >= "2025-09-18"]).prod() - 1, "연도별": by_year}


def apply_costs(weights: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """weights: 각 날 '종가에 결정해 다음날부터 보유' 하는 비중. 수익률 = sum(w_{t-1} * r_t) - 비용(비중 변화)."""
    w = weights.shift(1).fillna(0.0)
    port = (w * rets).sum(axis=1)
    turnover = (weights - weights.shift(1)).abs().sum(axis=1).fillna(0.0)
    return port - turnover.shift(1).fillna(0.0) * COST


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/best_model")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    px = pd.read_parquet("data/cache/etf_close.parquet")
    rets = px.pct_change()
    cash = rets["BIL"].fillna(rets["SHY"]).fillna(0.0)  # 현금 대용
    results = []

    def run(label, weights):
        r = apply_costs(weights, rets.fillna(0.0)) + (1 - weights.shift(1).fillna(0.0).sum(axis=1)).clip(lower=0) * cash
        results.append(stats(r[weights.index[0]:], label))

    # 1) 바이앤홀드
    for t in ("SPY", "QQQ", "TQQQ"):
        w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        w.loc[px[t].notna(), t] = 1.0
        run(f"바이앤홀드 {t}", w[px[t].notna()])
    # 2) 200일선 추세 필터 (월말 판정 → 다음날 반영, 밴드 없음)
    for t in ("SPY", "QQQ", "TQQQ", "QLD"):
        s = px[t].dropna()
        sig = (s > s.rolling(200).mean()).astype(float)
        monthly = sig.resample("ME").last().reindex(s.index, method="ffill").fillna(0.0)
        w = pd.DataFrame(0.0, index=s.index, columns=px.columns)
        w[t] = monthly
        run(f"200일선 필터 {t} (월말 판정)", w[s.index >= s.index[200]])
    # 3) 듀얼 모멘텀 (Antonacci): 매월 SPY/QQQ/EFA 중 12개월 수익률 최고, 단 T-bill 보다 낮으면 TLT
    m = px[["SPY", "QQQ", "EFA", "TLT", "BIL"]].resample("ME").last()
    mom = m / m.shift(12) - 1
    w_m = pd.DataFrame(0.0, index=m.index, columns=px.columns)
    for d in m.index[12:]:
        risky = mom.loc[d, ["SPY", "QQQ", "EFA"]]
        best = risky.idxmax()
        if pd.notna(risky.max()) and risky.max() > (mom.loc[d, "BIL"] if pd.notna(mom.loc[d, "BIL"]) else 0):
            w_m.loc[d, best] = 1.0
        else:
            w_m.loc[d, "TLT"] = 1.0
    w = w_m.reindex(px.index, method="ffill").fillna(0.0)
    run("듀얼 모멘텀 (SPY/QQQ/EFA vs TLT, 월간)", w[px.index >= m.index[12]])
    # 4) 변동성 타게팅 레버리지: QQQ 20일 실현변동성으로 노출 = min(3, 목표/실현), 200일선 아래면 0. TQQQ 로 근사(일수익률 3배 - 연 1% 비용)
    q = px["QQQ"].dropna()
    qr = q.pct_change()
    rv = qr.rolling(20).std() * np.sqrt(252)
    for target in (0.20, 0.30):
        expo = (target / rv).clip(upper=3.0)
        expo = expo.where(q > q.rolling(200).mean(), 0.0)
        expo_w = expo.resample("W-FRI").last().reindex(q.index, method="ffill").fillna(0.0)  # 주간 조정
        lev_ret = 3 * qr - 0.01 / 252  # TQQQ 근사
        w = pd.DataFrame(0.0, index=q.index, columns=px.columns)
        w["QQQ"] = (expo_w / 3.0)  # TQQQ 비중으로 환산 (수익률은 아래서 3배 적용)
        r = (w["QQQ"].shift(1).fillna(0) * lev_ret).fillna(0.0) - (w["QQQ"] - w["QQQ"].shift(1)).abs().shift(1).fillna(0) * COST
        r = r + (1 - w["QQQ"].shift(1).fillna(0)).clip(lower=0) * cash.reindex(q.index).fillna(0)
        results.append(stats(r[q.index >= q.index[200]], f"변동성 타게팅 {int(target * 100)}% + 200일선 (TQQQ 근사, 주간)"))

    rows = ["| 전략 | 기간 | CAGR | MDD | 샤프 | 최근3년 | 최근1년 |", "|---|---|---|---|---|---|---|"]
    for r in results:
        rows.append(f"| {r['전략']} | {r['기간']} | {r['CAGR'] * 100:+.1f}% | {r['MDD'] * 100:.1f}% | {r['샤프']:.2f} | {r['최근3년'] * 100:+.1f}% | {r['최근1년'] * 100:+.1f}% |")
    years = sorted({y for r in results for y in r["연도별"].index})
    yr = ["| 전략 | " + " | ".join(str(y) for y in years) + " |", "|---|" + "---|" * len(years)]
    for r in results:
        yr.append(f"| {r['전략']} | " + " | ".join(f"{r['연도별'].get(y, np.nan) * 100:+.0f}%" if pd.notna(r['연도별'].get(y, np.nan)) else "-" for y in years) + " |")
    md = ["# ETF 장기 검증 (2006~2026, 수수료 편도 0.25% + 슬리피지 0.05%)\n", "\n".join(rows), "\n## 연도별 수익률\n", "\n".join(yr),
          "\n주의: TQQQ 근사(QQQ 일수익률×3 − 연 1%)는 실제 TQQQ(2010~)와 거의 같지만 2008년 같은 급락기의 변동성 손실을 과소평가할 수 있다. 레버리지 ETF 는 MDD 가 -70~-90% 까지 갈 수 있다.\n"]
    (out / "etf_longrun.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(rows))
    print("\n".join(yr))


if __name__ == "__main__":
    main()
