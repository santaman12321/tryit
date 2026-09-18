"""전략 인터페이스.

전략은 가격 패널(dict[str, DataFrame]) 과 벤치마크(SPY) 일봉을 받아 Signals 를 돌려준다.

Signals 의 각 DataFrame 은 패널과 같은 index(거래일)/columns(심볼) 을 갖는다.

시간 규약 (look-ahead 방지)
--------------------------
hold_mode == "swing":
    entry[t] / exit[t] 는 t 일 **종가까지의 정보** 로 계산되고, 체결은 t+1 일 **시가** 에 이뤄진다.
hold_mode == "intraday":
    entry[t] 는 t-1 일 종가까지의 정보 + t 일 **시가** 만 사용할 수 있다.
    체결은 t 일 시가, 청산은 t 일 종가(또는 손절가).
stop_dist[t] : t 일 신호로 진입할 때 체결가에서 뺄 손절 거리(가격 단위). NaN -> 손절 없음.
trail_dist[t]: t 일 종가 기준 트레일링 스탑 거리. stop = max(stop, close[t] - trail_dist[t]). None -> 사용 안 함.
score[t]     : 슬롯보다 신호가 많을 때 우선순위(클수록 우선).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators import dollar_volume, sma


@dataclass
class Signals:
    entry: pd.DataFrame
    exit: pd.DataFrame
    score: pd.DataFrame
    stop_dist: pd.DataFrame | None = None
    trail_dist: pd.DataFrame | None = None
    target_pct: float | None = None
    adv: pd.DataFrame | None = None  # 20일 평균 거래대금(USD). 엔진이 포지션 크기를 adv * max_adv_pct 로 제한하는 데 사용


class Strategy:
    name: str = "base"
    hold_mode: str = "swing"  # "swing" | "intraday" | "overnight"
    direction: int = 1  # +1 롱 / -1 숏 (엔진이 모든 진입을 공매도로 처리)
    defaults: dict = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {sorted(unknown)}; allowed {sorted(self.defaults)}")
        self.params = {**self.defaults, **params}

    @property
    def max_hold_days(self) -> int | None:
        return self.params.get("max_hold_days")

    @property
    def label(self) -> str:
        """리포트/파일명에 쓰는 이름."""
        return self.name

    def generate(self, panels: dict[str, pd.DataFrame], bench: pd.DataFrame) -> Signals:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name}({self.params})"


# ----------------------------------------------------------------- 공용 필터
def liquidity_mask(panels, min_price: float, min_dollar_volume: float, n: int = 20,
                   max_mcap: float | None = None, min_mcap: float | None = None) -> pd.DataFrame:
    """가격 >= min_price, n일 평균 거래대금 >= min_dollar_volume, (있으면) 근사 시총 범위 안인 종목/일자.

    mcap 패널은 현재 발행주식수 * 당시 종가의 근사값(data.load_panels(universe=...) 참고).
    max_mcap/min_mcap 이 주어졌는데 mcap 패널이 없으면 ValueError.
    """
    close, volume = panels["close"], panels["volume"]
    dv = dollar_volume(close, volume, n)
    mask = (close >= min_price) & (dv >= min_dollar_volume)
    if max_mcap is not None or min_mcap is not None:
        if "mcap" not in panels:
            raise ValueError("mcap 패널이 없습니다: data.load_panels(universe=...) 로 로드하세요")
        mcap = panels["mcap"]
        if max_mcap is not None:
            mask &= mcap <= max_mcap
        if min_mcap is not None:
            mask &= mcap >= min_mcap
    return mask


def surge_mask(panels, surge_pct: float, vol_mult: float, vol_len: int = 20, min_close_pos: float = 0.0):
    """'급등일' 판정: 당일 수익률 >= surge_pct, 거래량 >= 직전 vol_len일 평균의 vol_mult 배,
    종가가 당일 범위의 min_close_pos 이상(1=고가 마감).  (ret, vol_ratio, close_pos, mask) 반환."""
    c, h, l, v = panels["close"], panels["high"], panels["low"], panels["volume"]
    ret = c / c.shift(1) - 1.0
    vol_ratio = v / sma(v, vol_len).shift(1)
    rng = (h - l)
    close_pos = ((c - l) / rng.where(rng > 0)).fillna(1.0)
    mask = (ret >= surge_pct) & (vol_ratio >= vol_mult) & (close_pos >= min_close_pos)
    return ret, vol_ratio, close_pos, mask.fillna(False)


def regime_series(bench_close: pd.Series, n: int = 200) -> pd.Series:
    """시장 레짐: 벤치마크 종가 > n일 이동평균."""
    return bench_close > sma(bench_close, n)


def broadcast(series: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    """일자 Series 를 패널 모양(일자 x 심볼)으로 확장."""
    s = series.reindex(like.index)
    return pd.DataFrame(np.repeat(s.to_numpy()[:, None], like.shape[1], axis=1), index=like.index, columns=like.columns)


def nan_like(like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=like.index, columns=like.columns)
