"""갭상승 단타 (Gap-up day trade) — 데이 트레이딩 근사.

아이디어: 전일 종가 대비 +4% 이상 갭 상승으로 시작한 상승추세 종목을 시가에 사서 당일 종가에 판다.
장중 손절(시가 대비 -3%)은 일봉의 저가로 판정한다(저가 <= 손절가면 손절 체결로 간주).
개인들이 흔히 하는 '급등주 따라붙기' 를 일봉 데이터로 검증하기 위한 전략이다.

주의: 일봉으로는 장중 고가/저가의 순서를 알 수 없으므로 보수적으로 손절이 먼저 맞았다고 가정한다.
"""
from __future__ import annotations

from ..indicators import dollar_volume, sma
from .base import Signals, Strategy, broadcast, liquidity_mask, regime_series


class GapDayStrategy(Strategy):
    name = "gap_day"
    hold_mode = "intraday"
    defaults = dict(
        gap_pct=0.04,
        max_gap_pct=0.30,
        trend_sma=200,
        stop_pct=0.03,
        min_price=5.0,
        min_dollar_volume=10_000_000,
        max_mcap=None,  # 근사 시총 상한(USD). 예) 2e9 = 소형주만
        min_mcap=None,
        use_regime_filter=False,
    )

    def generate(self, panels, bench) -> Signals:
        p = self.params
        o, c = panels["open"], panels["close"]
        prev_close = c.shift(1)
        gap = o / prev_close - 1.0
        # 당일 시가 시점에 알 수 있는 정보 = 전일까지의 종가/거래량 -> shift(1)
        trend = prev_close > sma(c, p["trend_sma"]).shift(1)
        liq = liquidity_mask(panels, p["min_price"], p["min_dollar_volume"], max_mcap=p["max_mcap"], min_mcap=p["min_mcap"]).shift(1).fillna(False).astype(bool)
        entry = (gap >= p["gap_pct"]) & (gap <= p["max_gap_pct"]) & trend & liq
        if p["use_regime_filter"]:
            entry &= broadcast(regime_series(bench["close"]).shift(1), c).fillna(False).astype(bool)
        stop_dist = p["stop_pct"] * o if p["stop_pct"] else None
        return Signals(entry=entry.fillna(False), exit=entry.fillna(False), score=gap, stop_dist=stop_dist,
                       adv=dollar_volume(c, panels["volume"]).shift(1))
