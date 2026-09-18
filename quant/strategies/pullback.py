"""눌림목 평균회귀 (Pullback / mean reversion in uptrend) — 스윙.

아이디어: 장기 상승추세(종가 > 200일선)에 있는 유동성 좋은 종목이 단기 과매도(RSI(2) < 10)
상태가 되면 매수하고, 반등(RSI(2) > 70 또는 종가 > 5일선)하거나 N일이 지나면 판다.
Larry Connors 계열의 고전적 규칙으로, 문헌상 승률이 높은(60~75%) 대신 건당 수익은 작다.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import atr, dollar_volume, pct_change, rsi, sma
from .base import Signals, Strategy, broadcast, liquidity_mask, nan_like, regime_series


class PullbackStrategy(Strategy):
    name = "pullback"
    hold_mode = "swing"
    defaults = dict(
        rsi_len=2,
        rsi_entry=10.0,
        rsi_exit=70.0,
        trend_sma=200,
        exit_sma=5,
        max_hold_days=10,
        stop_atr_mult=None,  # None: 손절 없음(시간 손절만). 예) 3.0 -> 3*ATR(14) 재난 손절
        atr_len=14,
        min_price=5.0,
        min_dollar_volume=10_000_000,
        max_mcap=None,  # 근사 시총 상한(USD). 예) 2e9 = 소형주만
        min_mcap=None,
        use_regime_filter=False,
        rank_by="rsi",  # "rsi": 가장 과매도 우선 / "rs": 6개월 상대강도 우선
    )

    def generate(self, panels, bench) -> Signals:
        p = self.params
        c = panels["close"]
        r = rsi(c, p["rsi_len"])
        trend = c > sma(c, p["trend_sma"])
        liq = liquidity_mask(panels, p["min_price"], p["min_dollar_volume"], max_mcap=p["max_mcap"], min_mcap=p["min_mcap"])
        entry = trend & liq & (r < p["rsi_entry"])
        if p["use_regime_filter"]:
            entry &= broadcast(regime_series(bench["close"]), c).fillna(False).astype(bool)
        exit_ = (r > p["rsi_exit"]) | (c > sma(c, p["exit_sma"]))
        score = -r if p["rank_by"] == "rsi" else pct_change(c, 126)
        if p["stop_atr_mult"]:
            stop_dist = p["stop_atr_mult"] * atr(panels["high"], panels["low"], c, p["atr_len"])
        else:
            stop_dist = nan_like(c)
        return Signals(entry=entry.fillna(False), exit=exit_.fillna(False), score=score, stop_dist=stop_dist,
                       adv=dollar_volume(c, panels["volume"]))
