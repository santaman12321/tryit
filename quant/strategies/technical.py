"""국내외 자동매매 봇에서 가장 흔한 기술적 지표 전략들 (알고랩 '자동매매 전략 30가지', je-suis-tm/quant-trading, freqtrade 샘플 전략 기준).

SMACrossStrategy   : 이동평균 골든/데드크로스 (기본 20/60)
MACDStrategy       : MACD(12,26,9) 신호선 상향/하향 교차
BollingerRSIStrategy: 볼린저(20, 2σ) 하단 이탈 + RSI(14) < 30 매수, 중심선 회복 시 매도
RSIStrategy        : RSI(14) < 30 매수, > 70 매도
SupertrendStrategy : Supertrend(10, 3) 상향 전환 매수, 하향 전환 매도
EMARSIStrategy     : EMA(9) > EMA(21) 이고 RSI(14) < 70 이면 매수, EMA 데드크로스 매도
모두 유동성 필터(가격 ≥ $5, 20일 평균 거래대금 ≥ $20M) 를 두고, score = 63일 수익률(슬롯 초과 시 우선순위).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import atr, dollar_volume, ema, pct_change, rsi, sma
from .base import Signals, Strategy, liquidity_mask, nan_like


def _common(panels, p):
    c = panels["close"]
    liq = liquidity_mask(panels, p["min_price"], p["min_dollar_volume"])
    a = atr(panels["high"], panels["low"], c, 14)
    return c, liq, a, pct_change(c, 63), dollar_volume(c, panels["volume"])


def _sig(entry, exit_, liq, score, adv, p, a):
    return Signals(entry=(entry & liq).fillna(False).astype(bool), exit=exit_.fillna(False).astype(bool), score=score, adv=adv,
                   stop_dist=p["stop_atr_mult"] * a if p.get("stop_atr_mult") else None)


class SMACrossStrategy(Strategy):
    name = "sma_cross"
    hold_mode = "swing"
    defaults = dict(fast=20, slow=60, min_price=5.0, min_dollar_volume=20_000_000, stop_atr_mult=None, max_hold_days=None)

    def generate(self, panels, bench) -> Signals:
        p = self.params
        c, liq, a, score, adv = _common(panels, p)
        f, s = sma(c, p["fast"]), sma(c, p["slow"])
        up, down = (f > s), (f < s)
        return _sig(up & ~up.shift(1).fillna(False), down & ~down.shift(1).fillna(False), liq, score, adv, p, a)


class MACDStrategy(Strategy):
    name = "macd"
    hold_mode = "swing"
    defaults = dict(fast=12, slow=26, signal=9, min_price=5.0, min_dollar_volume=20_000_000, stop_atr_mult=None, max_hold_days=None)

    def generate(self, panels, bench) -> Signals:
        p = self.params
        c, liq, a, score, adv = _common(panels, p)
        macd = ema(c, p["fast"]) - ema(c, p["slow"])
        sig = macd.ewm(span=p["signal"], adjust=False).mean()
        up, down = (macd > sig), (macd < sig)
        return _sig(up & ~up.shift(1).fillna(False), down & ~down.shift(1).fillna(False), liq, score, adv, p, a)


class BollingerRSIStrategy(Strategy):
    name = "bollinger_rsi"
    hold_mode = "swing"
    defaults = dict(n=20, k=2.0, rsi_max=30.0, min_price=5.0, min_dollar_volume=20_000_000, stop_atr_mult=None, max_hold_days=20)

    def generate(self, panels, bench) -> Signals:
        p = self.params
        c, liq, a, score, adv = _common(panels, p)
        mid = sma(c, p["n"])
        sd = c.rolling(p["n"]).std()
        lower = mid - p["k"] * sd
        r = rsi(c, 14)
        return _sig((c < lower) & (r < p["rsi_max"]), c >= mid, liq, -r, adv, p, a)


class RSIStrategy(Strategy):
    name = "rsi_1430"
    hold_mode = "swing"
    defaults = dict(n=14, buy=30.0, sell=70.0, min_price=5.0, min_dollar_volume=20_000_000, stop_atr_mult=None, max_hold_days=30)

    def generate(self, panels, bench) -> Signals:
        p = self.params
        c, liq, a, score, adv = _common(panels, p)
        r = rsi(c, p["n"])
        return _sig((r < p["buy"]) & ~(r.shift(1) < p["buy"]).fillna(False), r > p["sell"], liq, -r, adv, p, a)


def supertrend(panels, n=10, mult=3.0) -> pd.DataFrame:
    """True = 상승 추세. 벡터화가 어려워 열 단위 루프(심볼 수 x 일수)."""
    h, l, c = panels["high"], panels["low"], panels["close"]
    a = atr(h, l, c, n)
    hl2 = (h + l) / 2
    upper, lower = (hl2 + mult * a).to_numpy(), (hl2 - mult * a).to_numpy()
    cl = c.to_numpy()
    T, N = cl.shape
    trend = np.zeros((T, N), dtype=bool)
    fu, fl = upper.copy(), lower.copy()
    for t in range(1, T):
        prev_c = cl[t - 1]
        # 직전 밴드가 NaN(ATR 워밍업)이면 현재 밴드로 초기화
        pu, pl = np.where(np.isnan(fu[t - 1]), upper[t], fu[t - 1]), np.where(np.isnan(fl[t - 1]), lower[t], fl[t - 1])
        fu[t] = np.where((upper[t] < pu) | (prev_c > pu), upper[t], pu)
        fl[t] = np.where((lower[t] > pl) | (prev_c < pl), lower[t], pl)
        trend[t] = np.where(trend[t - 1], cl[t] > fl[t], cl[t] > fu[t])
    return pd.DataFrame(trend, index=c.index, columns=c.columns)


class SupertrendStrategy(Strategy):
    name = "supertrend"
    hold_mode = "swing"
    defaults = dict(n=10, mult=3.0, min_price=5.0, min_dollar_volume=20_000_000, stop_atr_mult=None, max_hold_days=None)

    def generate(self, panels, bench) -> Signals:
        p = self.params
        c, liq, a, score, adv = _common(panels, p)
        up = supertrend(panels, p["n"], p["mult"])
        return _sig(up & ~up.shift(1).fillna(False), ~up & up.shift(1).fillna(False), liq, score, adv, p, a)


class EMARSIStrategy(Strategy):
    name = "ema_rsi"
    hold_mode = "swing"
    defaults = dict(fast=9, slow=21, rsi_max=70.0, min_price=5.0, min_dollar_volume=20_000_000, stop_atr_mult=None, max_hold_days=None)

    def generate(self, panels, bench) -> Signals:
        p = self.params
        c, liq, a, score, adv = _common(panels, p)
        f, s = ema(c, p["fast"]), ema(c, p["slow"])
        r = rsi(c, 14)
        up = (f > s)
        return _sig(up & ~up.shift(1).fillna(False) & (r < p["rsi_max"]), f < s, liq, score, adv, p, a)
