"""저회전율 스윙 전략 3종 — 국내 증권사 수수료(편도 0.25%) 구조에서 살아남으려면 보유 기간이 길어야 한다.

XSMomentumStrategy : 횡단면 모멘텀 (Jegadeesh & Titman 12-1개월). 리밸런스일에 상위 N 진입, 순위가 2N 밖으로 밀리면 청산. SPY 200일선 레짐 필터.
PEADStrategy       : 실적발표 후 드리프트 (Bernard & Thomas). 실적발표일(±1일)에 +5%↑·거래량 2배↑ 급등한 종목을 다음날 시가 매수, N일 보유 + ATR 트레일.
HighBreakoutStrategy: 52주 신고가 돌파 + 거래량 + 정배열 (O'Neil/Minervini 계열). 3~5 ATR 트레일링, 긴 보유.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import atr, dollar_volume, pct_change, rolling_max, sma
from ..screens import earnings_mask
from .base import Signals, Strategy, broadcast, liquidity_mask, nan_like, regime_series


class XSMomentumStrategy(Strategy):
    name = "xs_momentum"
    hold_mode = "swing"
    defaults = dict(
        lookback=252, skip=21, top_n=20, exit_rank=40, rebalance_days=21,
        min_price=5.0, min_dollar_volume=20_000_000, max_mcap=None, min_mcap=None,
        use_regime_filter=True, trail_atr_mult=None, atr_len=14, max_hold_days=None, stop_atr_mult=None,
        rank_by="mom",  # "mom": 12-1 수익률 / "mom_vol": 수익률/변동성
    )

    def generate(self, panels, bench) -> Signals:
        p = self.params
        c = panels["close"]
        mom = c.shift(p["skip"]) / c.shift(p["lookback"]) - 1.0
        if p["rank_by"] == "mom_vol":
            vol = np.log(c / c.shift(1)).rolling(p["lookback"] - p["skip"]).std().shift(p["skip"])
            mom = mom / vol.replace(0, np.nan)
        liq = liquidity_mask(panels, p["min_price"], p["min_dollar_volume"], max_mcap=p["max_mcap"], min_mcap=p["min_mcap"])
        score = mom.where(liq)
        rank = score.rank(axis=1, ascending=False)
        reb = pd.Series(False, index=c.index)
        reb.iloc[::p["rebalance_days"]] = True
        entry = (rank <= p["top_n"]) & broadcast(reb, c).astype(bool)
        exit_ = (rank > p["exit_rank"]) & broadcast(reb, c).astype(bool)
        if p["use_regime_filter"]:
            reg = broadcast(regime_series(bench["close"]), c).fillna(False).astype(bool)
            entry &= reg
            exit_ |= ~reg
        a = atr(panels["high"], panels["low"], c, p["atr_len"])
        return Signals(entry=entry.fillna(False), exit=exit_.fillna(False), score=score, adv=dollar_volume(c, panels["volume"]),
                       stop_dist=p["stop_atr_mult"] * a if p["stop_atr_mult"] else None, trail_dist=p["trail_atr_mult"] * a if p["trail_atr_mult"] else None)


class PEADStrategy(Strategy):
    name = "pead"
    hold_mode = "swing"
    defaults = dict(
        min_ret=0.05, min_rvol=2.0, max_hold_days=40, trail_atr_mult=3.0, stop_atr_mult=2.0, atr_len=14,
        min_price=5.0, min_dollar_volume=10_000_000, max_mcap=None, min_mcap=None, use_regime_filter=True, require_trend=True,
    )

    def generate(self, panels, bench) -> Signals:
        p = self.params
        c, v = panels["close"], panels["volume"]
        em = earnings_mask(panels)
        if em is None:
            raise RuntimeError("earnings_dates.parquet 가 필요합니다 (scripts/fetch_earnings.py)")
        ret1 = c / c.shift(1) - 1.0
        rvol = v / sma(v, 20).shift(1)
        liq = liquidity_mask(panels, p["min_price"], p["min_dollar_volume"], max_mcap=p["max_mcap"], min_mcap=p["min_mcap"])
        entry = em & (ret1 >= p["min_ret"]) & (rvol >= p["min_rvol"]) & liq
        if p["require_trend"]:
            entry &= c > sma(c, 50)
        if p["use_regime_filter"]:
            entry &= broadcast(regime_series(bench["close"]), c).fillna(False).astype(bool)
        a = atr(panels["high"], panels["low"], c, p["atr_len"])
        return Signals(entry=entry.fillna(False).astype(bool), exit=nan_like(c).fillna(False).astype(bool), score=ret1 * rvol,
                       stop_dist=p["stop_atr_mult"] * a if p["stop_atr_mult"] else None, trail_dist=p["trail_atr_mult"] * a if p["trail_atr_mult"] else None,
                       adv=dollar_volume(c, v))


class HighBreakoutStrategy(Strategy):
    name = "high_breakout"
    hold_mode = "swing"
    defaults = dict(
        lookback=252, vol_mult=1.5, vol_len=50, trail_atr_mult=4.0, stop_atr_mult=2.5, atr_len=14, max_hold_days=250,
        min_price=5.0, min_dollar_volume=20_000_000, max_mcap=None, min_mcap=None, use_regime_filter=True, rank_by="rs",
        exit_below_sma=50,
    )

    def generate(self, panels, bench) -> Signals:
        p = self.params
        c, h, v = panels["close"], panels["high"], panels["volume"]
        prior_high = rolling_max(h, p["lookback"]).shift(1)
        vol_ratio = v / sma(v, p["vol_len"]).shift(1)
        trend = (c > sma(c, 50)) & (sma(c, 50) > sma(c, 200))
        liq = liquidity_mask(panels, p["min_price"], p["min_dollar_volume"], max_mcap=p["max_mcap"], min_mcap=p["min_mcap"])
        entry = (c > prior_high) & (vol_ratio >= p["vol_mult"]) & trend & liq
        if p["use_regime_filter"]:
            entry &= broadcast(regime_series(bench["close"]), c).fillna(False).astype(bool)
        exit_ = c < sma(c, p["exit_below_sma"]) if p["exit_below_sma"] else nan_like(c).fillna(False)
        score = pct_change(c, 126) if p["rank_by"] == "rs" else vol_ratio
        a = atr(h, panels["low"], c, p["atr_len"])
        return Signals(entry=entry.fillna(False).astype(bool), exit=exit_.fillna(False).astype(bool), score=score,
                       stop_dist=p["stop_atr_mult"] * a if p["stop_atr_mult"] else None, trail_dist=p["trail_atr_mult"] * a if p["trail_atr_mult"] else None,
                       adv=dollar_volume(c, v))


class MLRankStrategy(Strategy):
    """ML 랭커(20일 선행 수익률 순위, scripts/ml_rank_train.py) 상위 N 을 주기적으로 리밸런스."""

    name = "ml_rank"
    hold_mode = "swing"
    defaults = dict(top_n=20, exit_rank=40, rebalance_days=5, use_regime_filter=True, trail_atr_mult=None, stop_atr_mult=None, atr_len=14,
                    max_hold_days=None, min_price=5.0, min_dollar_volume=5_000_000, max_mcap=None, min_mcap=None, pred_file=None)

    def generate(self, panels, bench) -> Signals:
        from ..data import CACHE_DIR

        p = self.params
        c = panels["close"]
        pred = pd.read_parquet(p["pred_file"] or (CACHE_DIR / "ml_rank_pred.parquet"))
        pred.index = pd.to_datetime(pred.index)
        pred = pred.reindex(index=c.index, columns=c.columns)
        liq = liquidity_mask(panels, p["min_price"], p["min_dollar_volume"], max_mcap=p["max_mcap"], min_mcap=p["min_mcap"])
        score = pred.where(liq)
        rank = score.rank(axis=1, ascending=False)
        reb = pd.Series(False, index=c.index)
        reb.iloc[::p["rebalance_days"]] = True
        entry = (rank <= p["top_n"]) & broadcast(reb, c).astype(bool)
        exit_ = (rank > p["exit_rank"]) & broadcast(reb, c).astype(bool)
        if p["use_regime_filter"]:
            reg = broadcast(regime_series(bench["close"]), c).fillna(False).astype(bool)
            entry &= reg
            exit_ |= ~reg
        a = atr(panels["high"], panels["low"], c, p["atr_len"])
        return Signals(entry=entry.fillna(False), exit=exit_.fillna(False), score=score, adv=dollar_volume(c, panels["volume"]),
                       stop_dist=p["stop_atr_mult"] * a if p["stop_atr_mult"] else None, trail_dist=p["trail_atr_mult"] * a if p["trail_atr_mult"] else None)
