"""커뮤니티·GitHub·유튜브에서 공유되는 자동매매 모델을 규칙 그대로 재현해 같은 비용으로 비교한다 (ETF 기반).

포함 모델 (출처: docs/community_models.md)
  HFEA            : UPRO 55 / TMF 45, 분기 리밸런스 (Bogleheads Hedgefundie)
  9Sig            : TQQQ/AGG 60/40, 분기 신호선 9% 성장, 잉여 매도/부족 매수, 매수 상한 채권의 90%, 채권 하한 10%, 채권>30% 시 60/40 리셋, 30다운 룰 (Jason Kelly)
  TQQQ FTLT       : SPY>200일선 → (TQQQ RSI10>79 → UVXY, else TQQQ) / SPY≤200일선 → (TQQQ RSI10<31 → TECL, SPY RSI10<30 → UPRO, TQQQ>20일선 → TQQQ, else TLT) (Composer/Reddit 공유본 근사)
  200일선 스위칭   : SPY>200일선 이면 TQQQ, 아니면 현금 — 일간 판정 (Gayed 'Leverage for the Long Run' 류)
  듀얼 모멘텀 GEM  : 월간, SPY vs EFA 12개월 수익률 상위, 둘 다 T-bill 이하면 AGG (Antonacci; 강환국 소개)
  무한매수법 v2.2  : TQQQ 40분할, T=누적매수/1회분, 전반 절반 평단 LOC + 절반 평단×(1+(10−T/2)%) LOC, 후반 (10−T/2)% LOC, 매도 1/4 (10−T/2)% LOC + 3/4 +10% 지정가, 40회 소진 시 전량 매도 후 재시작 (라오어)
  밸류 리밸런싱 근사: TQQQ 목표가치 V 연 15% 성장, 2주마다 평가금이 V×(1±0.15) 밴드 밖이면 밴드 안으로 매수/매도, 현금 부족 시 가능한 만큼 (라오어 VR 의 공개 설명 기반 근사)
  변동성 돌파 K=0.5: 대상 ETF, 목표가 = 시가 + K×(전일 고가−저가) 를 고가가 넘으면 목표가 매수, 다음날 시가 매도 (Larry Williams; 국내 봇 표준)
  변동성 타게팅(우리): 비교 기준
비용: 매매 금액의 편도 0.25% + 슬리피지 0.05%.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COST = 0.0025 + 0.0005


def load():
    px = pd.read_parquet("data/cache/etf_close.parquet")
    ohlc = pd.read_parquet("data/cache/etf_ohlc.parquet")
    return px, ohlc


def rsi(s: pd.Series, n: int = 10) -> pd.Series:
    d = s.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + au / ad.replace(0, np.nan))


def stats(r: pd.Series, label: str) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / 252
    by_year = (1 + r).groupby(r.index.year).prod() - 1
    return {"모델": label, "기간": f"{r.index[0].date()}~{r.index[-1].date()}", "CAGR": eq.iloc[-1] ** (1 / years) - 1, "MDD": (eq / eq.cummax() - 1).min(),
            "샤프": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan, "최근3년": (1 + r[r.index >= "2023-09-18"]).prod() - 1,
            "2024~": (1 + r[r.index >= "2024-01-02"]).prod() - 1, "최근1년": (1 + r[r.index >= "2025-09-18"]).prod() - 1, "연도별": by_year}


def weights_to_returns(w: pd.DataFrame, rets: pd.DataFrame, cash: pd.Series) -> pd.Series:
    w = w.reindex(rets.index).ffill().fillna(0.0)
    lag = w.shift(1).fillna(0.0)
    port = (lag * rets.reindex(columns=w.columns).fillna(0.0)).sum(axis=1)
    turnover = (w - w.shift(1)).abs().sum(axis=1).fillna(0.0)
    return port - turnover.shift(1).fillna(0.0) * COST + (1 - lag.sum(axis=1)).clip(lower=0) * cash.reindex(rets.index).fillna(0.0)


def hfea(px, rets, cash):
    s = px[["UPRO", "TMF"]].dropna()
    q = s.resample("QE").last().index
    w = pd.DataFrame(np.nan, index=s.index, columns=["UPRO", "TMF"])
    # 분기말에 55/45 로 맞추고 그 사이엔 드리프트 (비중 = 가치 비율 추적)
    val = None
    rows = []
    hold = {"UPRO": 0.55, "TMF": 0.45}
    r = rets[["UPRO", "TMF"]].reindex(s.index).fillna(0.0)
    cur = pd.Series(hold)
    out = []
    for d in s.index:
        cur = cur * (1 + r.loc[d])
        cur = cur / cur.sum()
        if d in q:
            cur = pd.Series(hold)
        out.append(cur.values)
    w = pd.DataFrame(out, index=s.index, columns=["UPRO", "TMF"])
    return weights_to_returns(w, rets, cash), "HFEA (UPRO55/TMF45 분기)"


def gem(px, rets, cash):
    m = px[["SPY", "EFA", "AGG", "BIL"]].dropna().resample("ME").last()
    mom = m / m.shift(12) - 1
    w = pd.DataFrame(0.0, index=m.index, columns=["SPY", "EFA", "AGG"])
    for d in m.index[12:]:
        best = "SPY" if mom.loc[d, "SPY"] >= mom.loc[d, "EFA"] else "EFA"
        w.loc[d, best if mom.loc[d, best] > mom.loc[d, "BIL"] else "AGG"] = 1.0
    return weights_to_returns(w[w.index >= m.index[12]], rets, cash), "듀얼 모멘텀 GEM (SPY/EFA/AGG 월간)"


def sma200_switch(px, rets, cash, lev="TQQQ"):
    spy = px["SPY"]
    on = (spy > spy.rolling(200).mean()).astype(float)
    s = px[lev].dropna()
    w = pd.DataFrame({lev: on.reindex(s.index)}).fillna(0.0)
    return weights_to_returns(w[s.index >= s.index[1]], rets, cash), f"SPY 200일선 스위칭 → {lev} (일간)"


def tqqq_ftlt(px, rets, cash, hedge="UVXY"):
    spy, tq = px["SPY"], px["TQQQ"].dropna()
    idx = tq.index
    r_tq, r_spy = rsi(tq, 10), rsi(spy.reindex(idx), 10)
    above = (spy > spy.rolling(200).mean()).reindex(idx)
    tq20 = tq > tq.rolling(20).mean()
    cols = ["TQQQ", "UVXY", "TECL", "UPRO", "TLT"]
    w = pd.DataFrame(0.0, index=idx, columns=cols)
    for d in idx:
        if above.loc[d]:
            pick = hedge if r_tq.loc[d] > 79 else "TQQQ"
            if pick == hedge and (hedge == "BIL" or pd.isna(px.loc[d, hedge])):
                pick = None  # UVXY 상장 전 / BIL 변형은 현금
        else:
            if r_tq.loc[d] < 31:
                pick = "TECL"
            elif r_spy.loc[d] < 30:
                pick = "UPRO"
            elif tq20.loc[d]:
                pick = "TQQQ"
            else:
                pick = "TLT"
        if pick:
            w.loc[d, pick] = 1.0
    return weights_to_returns(w, rets, cash), f"TQQQ FTLT (Composer/Reddit 트리 근사, 과열 시 {hedge}, 일간)"


def sma_cross_etf(px, rets, cash, sym="QQQ", fast=50, slow=200):
    s = px[sym].dropna()
    on = (s.rolling(fast).mean() > s.rolling(slow).mean()).astype(float)
    w = pd.DataFrame({sym: on})
    return weights_to_returns(w[s.index >= s.index[slow]], rets, cash), f"SMA {fast}/{slow} 골든크로스 ({sym}, 일간)"


def nine_sig(px, rets, cash):
    s = px[["TQQQ", "AGG"]].dropna()
    r = rets[["TQQQ", "AGG"]].reindex(s.index).fillna(0.0)
    qends = set(s.resample("QE").last().index)
    stock, bond = 60.0, 40.0
    signal = stock
    skip_sells = 0
    hi8 = s["TQQQ"].rolling(252 * 2).max()
    out = []
    for d in s.index:
        stock *= 1 + r.loc[d, "TQQQ"]
        bond *= 1 + r.loc[d, "AGG"]
        cost = 0.0
        if d in qends:
            signal *= 1.09
            if s.loc[d, "TQQQ"] <= 0.7 * hi8.loc[d]:
                skip_sells = 2
            diff = stock - signal
            if diff > 0:
                if skip_sells > 0:
                    skip_sells -= 1
                else:
                    stock -= diff
                    bond += diff
                    cost = diff * COST * 2
                    if bond > 0.3 * (stock + bond):
                        tot = stock + bond
                        stock, bond, signal = 0.6 * tot, 0.4 * tot, 0.6 * tot
            elif diff < 0:
                buy = min(-diff, 0.9 * bond, max(0.0, bond - 0.1 * (stock + bond)))
                stock += buy
                bond -= buy
                cost = buy * COST * 2
            if s.loc[d, "TQQQ"] / s["TQQQ"].shift(63).loc[d] - 1 >= 1.0 if not pd.isna(s["TQQQ"].shift(63).loc[d]) else False:
                tot = stock + bond
                stock, bond, signal = 0.6 * tot, 0.4 * tot, 0.6 * tot
        tot = stock + bond
        stock -= cost * stock / tot
        bond -= cost * bond / tot
        out.append(stock + bond)
    eq = pd.Series(out, index=s.index) / 100.0
    return eq.pct_change().fillna(0.0), "9Sig (TQQQ/AGG 60/40, 분기 9% 신호선)"


def value_rebalancing(px, rets, cash, growth=0.15, band=0.15, every=10):
    s = px["TQQQ"].dropna()
    r = rets["TQQQ"].reindex(s.index).fillna(0.0)
    c = cash.reindex(s.index).fillna(0.0)
    V = 50.0
    stock, pool = 50.0, 50.0
    out = []
    daily_g = (1 + growth) ** (1 / 252) - 1
    for i, d in enumerate(s.index):
        stock *= 1 + r.loc[d]
        pool *= 1 + c.loc[d]
        V *= 1 + daily_g
        if i % every == 0 and i > 0:
            if stock > V * (1 + band):
                sell = stock - V
                stock -= sell
                pool += sell * (1 - 2 * COST)
            elif stock < V * (1 - band):
                buy = min(V - stock, pool)
                stock += buy * (1 - 2 * COST)
                pool -= buy
        out.append(stock + pool)
    eq = pd.Series(out, index=s.index) / 100.0
    return eq.pct_change().fillna(0.0), f"밸류 리밸런싱 근사 (TQQQ, V 연 {int(growth * 100)}%, ±{int(band * 100)}% 밴드, 2주)"


def infinite_buying(ohlc, symbol="TQQQ", n_split=40, target=0.10, reset_on_exhaust=True):
    d = ohlc[ohlc.symbol == symbol].set_index("date").sort_index()
    cash0 = 100.0
    unit = cash0 / n_split
    cash, qty, cost_basis = cash0, 0.0, 0.0
    out = []
    for date, row in d.iterrows():
        o, h, l, c = row.open, row.high, row.low, row.close
        avg = cost_basis / qty if qty > 0 else np.nan
        T = (cost_basis / unit) if unit > 0 else 0.0
        # 매도 (보유 시): 1/4 LOC at avg*(1+(10-T/2)%), 3/4 지정가 +10%
        if qty > 0:
            lvl1 = avg * (1 + max(0.0, (10 - T / 2)) / 100)
            if h >= avg * (1 + target):
                px = avg * (1 + target)
                cash += 0.75 * qty * px * (1 - COST)
                cost_basis -= 0.75 * cost_basis
                qty *= 0.25
            if c >= lvl1 and qty > 0:
                cash += qty * c * (1 - COST)  # 남은 1/4 (또는 전량) LOC 매도
                qty, cost_basis = 0.0, 0.0
        # 매수 (현금 있을 때)
        if cash >= unit * 0.5 and qty >= 0:
            avg = cost_basis / qty if qty > 0 else c
            T = cost_basis / unit
            spend = 0.0
            if T < 20:
                half = unit / 2
                if c <= avg:
                    spend += half
                lvl = avg * (1 + max(0.0, (10 - T / 2)) / 100)
                if c <= lvl:
                    spend += half
            else:
                lvl = avg * (1 + max(0.0, (10 - T / 2)) / 100)
                if c <= lvl:
                    spend += unit
            spend = min(spend, cash)
            if spend > 0:
                qty += spend * (1 - COST) / c
                cost_basis += spend
                cash -= spend
        if reset_on_exhaust and cash < unit * 0.5 and qty > 0 and cost_basis >= unit * (n_split - 0.5):
            # 40회차 소진: 전량 매도 후 재시작
            cash += qty * c * (1 - COST)
            qty, cost_basis = 0.0, 0.0
            unit = cash / n_split
        if qty == 0 and cost_basis == 0 and cash > 0:
            unit = cash / n_split  # 사이클 종료 시 원금 갱신
        out.append(cash + qty * c)
    eq = pd.Series(out, index=d.index) / cash0
    return eq.pct_change().fillna(0.0), f"무한매수법 v2.2 ({symbol}, 40분할, +{int(target * 100)}%)"


def vol_breakout(ohlc, symbol="TQQQ", k=0.5):
    d = ohlc[ohlc.symbol == symbol].set_index("date").sort_index()
    rng = (d.high - d.low).shift(1)
    target = d.open + k * rng
    hit = d.high >= target
    nxt_open = d.open.shift(-1)
    r = np.where(hit, nxt_open / target - 1 - 2 * COST, 0.0)
    return pd.Series(r, index=d.index).fillna(0.0), f"변동성 돌파 K={k} ({symbol}, 다음날 시가 매도)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/community_compare")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    px, ohlc = load()
    rets = px.pct_change()
    cash = rets["BIL"].fillna(rets["SHY"]).fillna(0.0)
    results = []
    for fn in (hfea, gem, sma200_switch, tqqq_ftlt, nine_sig, value_rebalancing):
        r, label = fn(px, rets, cash)
        results.append(stats(r, label))
    r, label = tqqq_ftlt(px, rets, cash, hedge="BIL")
    results.append(stats(r, label))
    for sym in ("QQQ", "TQQQ"):
        r, label = sma_cross_etf(px, rets, cash, sym)
        results.append(stats(r, label))
    for sym in ("TQQQ", "SOXL"):
        r, label = infinite_buying(ohlc, sym)
        results.append(stats(r, label))
    for sym, k in (("TQQQ", 0.5), ("QQQ", 0.5), ("SPY", 0.5)):
        r, label = vol_breakout(ohlc, sym, k)
        results.append(stats(r, label))
    for t in ("SPY", "QQQ", "TQQQ"):
        results.append(stats(rets[t].dropna(), f"바이앤홀드 {t}"))
    rows = ["| 모델 | 기간 | CAGR | MDD | 샤프 | 최근3년 | 2024~ | 최근1년 |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        rows.append(f"| {r['모델']} | {r['기간']} | {r['CAGR'] * 100:+.1f}% | {r['MDD'] * 100:.1f}% | {r['샤프']:.2f} | {r['최근3년'] * 100:+.1f}% | {r['2024~'] * 100:+.1f}% | {r['최근1년'] * 100:+.1f}% |")
    years = list(range(2010, 2027))
    yr = ["| 모델 | " + " | ".join(str(y) for y in years) + " |", "|---|" + "---|" * len(years)]
    for r in results:
        yr.append(f"| {r['모델']} | " + " | ".join(f"{r['연도별'].get(y, np.nan) * 100:+.0f}%" if pd.notna(r['연도별'].get(y, np.nan)) else "-" for y in years) + " |")
    md = ["# 커뮤니티 ETF 모델 재현 (비용: 편도 0.25% + 슬리피지 0.05%)\n", "\n".join(rows), "\n## 연도별\n", "\n".join(yr)]
    (out / "etf_models.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k != "연도별"} for r in results]).to_csv(out / "etf_models.csv", index=False)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
