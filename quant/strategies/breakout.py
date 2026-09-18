"""돌파 추세추종 (Breakout momentum) — 스윙.

아이디어: 상승추세(종가 > 50일선 > 200일선) 종목이 거래량을 동반(평균의 1.5배 이상)해
직전 20일 고가를 종가로 돌파하면 다음날 시가에 산다. 2*ATR 초기 손절, 3*ATR 트레일링 스탑,
직전 10일 저가 이탈 시 청산. 승률은 낮지만(30~45%) 손익비로 버는 구조.
"""
from __future__ import annotations

from ..indicators import atr, dollar_volume, pct_change, rolling_max, rolling_min, sma
from .base import Signals, Strategy, broadcast, liquidity_mask, regime_series


class BreakoutStrategy(Strategy):
    name = "breakout"
    hold_mode = "swing"
    defaults = dict(
        lookback=20,
        exit_lookback=10,
        vol_mult=1.5,
        vol_len=20,
        trend_sma_fast=50,
        trend_sma_slow=200,
        stop_atr_mult=2.0,
        trail_atr_mult=3.0,
        atr_len=14,
        max_hold_days=60,
        min_price=5.0,
        min_dollar_volume=10_000_000,
        max_mcap=None,  # 근사 시총 상한(USD). 예) 2e9 = 소형주만
        min_mcap=None,
        use_regime_filter=True,
        rank_by="rs",  # "rs": 63일 수익률 / "vol": 거래량 배수
    )

    def generate(self, panels, bench) -> Signals:
        p = self.params
        c, h, l, v = panels["close"], panels["high"], panels["low"], panels["volume"]
        prior_high = rolling_max(h, p["lookback"]).shift(1)
        vol_ratio = v / sma(v, p["vol_len"]).shift(1)
        fast, slow = sma(c, p["trend_sma_fast"]), sma(c, p["trend_sma_slow"])
        trend = (c > fast) & (fast > slow)
        liq = liquidity_mask(panels, p["min_price"], p["min_dollar_volume"], max_mcap=p["max_mcap"], min_mcap=p["min_mcap"])
        entry = (c > prior_high) & (vol_ratio >= p["vol_mult"]) & trend & liq
        if p["use_regime_filter"]:
            entry &= broadcast(regime_series(bench["close"]), c).fillna(False).astype(bool)
        exit_ = c < rolling_min(l, p["exit_lookback"]).shift(1)
        score = pct_change(c, 63) if p["rank_by"] == "rs" else vol_ratio
        a = atr(h, l, c, p["atr_len"])
        return Signals(
            entry=entry.fillna(False),
            exit=exit_.fillna(False),
            score=score,
            stop_dist=p["stop_atr_mult"] * a if p["stop_atr_mult"] else None,
            trail_dist=p["trail_atr_mult"] * a if p["trail_atr_mult"] else None,
            adv=dollar_volume(c, v),
        )
