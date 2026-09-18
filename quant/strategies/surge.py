"""급등주 전략 2종 — 시총 작은 종목이 거래량을 터뜨리며 급등한 뒤의 흐름을 검증한다.

'급등일' 정의(공통): 당일 수익률 >= surge_pct(기본 15%), 거래량 >= 직전 20일 평균의 vol_mult(기본 5)배,
                  당일 거래대금 >= min_day_dollar_volume(기본 $5M, 실제로 살 수 있는 종목만),
                  근사 시총 <= max_mcap(기본 $2B), 가격 >= min_price(기본 $1).

SurgeChaseStrategy   : 급등일 종가 확인 후 **다음날 시가에 추격 매수**, N일 보유 / 손절 / (선택) 익절.
SurgePullbackStrategy: 급등 후 window일 안에 급등 종가 대비 pullback_pct 이상 **눌림**이 오면 매수
                       (단, 급등 전 종가는 지켜야 함), 급등 종가 회복 / RSI(2)>70 / N일 경과 시 청산.
"""
from __future__ import annotations

from ..indicators import dollar_volume, rsi
from .base import Signals, Strategy, broadcast, liquidity_mask, nan_like, regime_series, surge_mask


class SurgeChaseStrategy(Strategy):
    name = "surge_chase"
    hold_mode = "swing"
    defaults = dict(
        surge_pct=0.15,
        vol_mult=5.0,
        vol_len=20,
        min_close_pos=0.6,  # 종가가 당일 범위 상단 40% 안 (강한 마감)
        max_hold_days=3,
        stop_pct=0.10,
        target_pct=None,  # 예) 0.15
        min_price=1.0,
        min_dollar_volume=1_000_000,  # 20일 평균 거래대금
        min_day_dollar_volume=5_000_000,  # 급등일 당일 거래대금
        max_mcap=2_000_000_000,
        min_mcap=None,
        use_regime_filter=False,
        rank_by="vol",  # "vol": 거래량 배수 / "ret": 급등률
    )

    def generate(self, panels, bench) -> Signals:
        p = self.params
        c, v = panels["close"], panels["volume"]
        ret, vol_ratio, close_pos, surge = surge_mask(panels, p["surge_pct"], p["vol_mult"], p["vol_len"], p["min_close_pos"])
        liq = liquidity_mask(panels, p["min_price"], p["min_dollar_volume"], max_mcap=p["max_mcap"], min_mcap=p["min_mcap"])
        entry = surge & liq & ((c * v) >= p["min_day_dollar_volume"])
        if p["use_regime_filter"]:
            entry &= broadcast(regime_series(bench["close"]), c).fillna(False).astype(bool)
        exit_ = nan_like(c).fillna(False).astype(bool)  # 시간/손절/익절로만 청산
        score = vol_ratio if p["rank_by"] == "vol" else ret
        stop_dist = p["stop_pct"] * c if p["stop_pct"] else None
        return Signals(entry=entry.fillna(False), exit=exit_, score=score, stop_dist=stop_dist,
                       target_pct=p["target_pct"], adv=dollar_volume(c, v))


class SurgePullbackStrategy(Strategy):
    name = "surge_pullback"
    hold_mode = "swing"
    defaults = dict(
        surge_pct=0.15,
        vol_mult=5.0,
        vol_len=20,
        min_close_pos=0.0,
        window=10,  # 급등 후 며칠 안의 눌림만 본다
        pullback_pct=0.08,  # 급등일 종가 대비 최소 눌림 폭
        max_hold_days=5,
        rsi_exit=70.0,
        stop_pct=0.10,
        min_price=1.0,
        min_dollar_volume=1_000_000,
        min_day_dollar_volume=5_000_000,
        max_mcap=2_000_000_000,
        min_mcap=None,
        use_regime_filter=False,
    )

    def generate(self, panels, bench) -> Signals:
        p = self.params
        c, v = panels["close"], panels["volume"]
        ret, vol_ratio, close_pos, surge = surge_mask(panels, p["surge_pct"], p["vol_mult"], p["vol_len"], p["min_close_pos"])
        liq = liquidity_mask(panels, p["min_price"], p["min_dollar_volume"], max_mcap=p["max_mcap"], min_mcap=p["min_mcap"])
        surge = surge & liq & ((c * v) >= p["min_day_dollar_volume"])
        # 최근 window일 안의 마지막 급등일 정보를 앞으로 채운다 (급등일 당일 포함 -> 당일은 눌림 조건에 안 걸림)
        surge_close = c.where(surge).ffill(limit=p["window"])
        pre_close = c.shift(1).where(surge).ffill(limit=p["window"])
        surge_vol = vol_ratio.where(surge).ffill(limit=p["window"])
        pulled = c <= surge_close * (1.0 - p["pullback_pct"])
        held = c > pre_close  # 급등 전 종가는 지키고 있어야 (완전 되돌림 제외)
        entry = pulled & held & surge_close.notna()
        if p["use_regime_filter"]:
            entry &= broadcast(regime_series(bench["close"]), c).fillna(False).astype(bool)
        r2 = rsi(c, 2)
        exit_ = (r2 > p["rsi_exit"]) | (c >= surge_close) | (c <= pre_close)
        score = surge_vol
        stop_dist = p["stop_pct"] * c if p["stop_pct"] else None
        return Signals(entry=entry.fillna(False), exit=exit_.fillna(False), score=score, stop_dist=stop_dist,
                       adv=dollar_volume(c, v))
