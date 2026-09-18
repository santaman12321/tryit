"""블로그/유튜브/해외 데이트레이딩 커뮤니티에서 공유되는 '급등주 찾기 공식' 을 일봉 규칙으로 정량화한 스크린 모음.

각 스크린은 (panels, feats, **params) -> bool DataFrame(일자 x 심볼) 을 돌려준다.
  * "close 이벤트": t 일 종가 시점에 알 수 있는 정보만 사용 -> 다음날 시가 진입(추격/눌림/스윙) 또는 당일 종가 진입(오버나잇)
  * "open 이벤트" (gap 계열): t-1 일까지의 정보 + t 일 시가만 사용 -> t 일 시가 진입, 당일 종가 청산

출처와 원문 조건은 docs/surge_formulas.md 참고.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr, ema, rolling_max, rolling_min, rsi, sma


_FEATURE_CACHE: dict[tuple, dict] = {}


def compute_features(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """스크린들이 공통으로 쓰는 특징 패널. 모두 시점 t 까지의 정보만 사용. (같은 패널 객체면 캐시)"""
    key = (id(panels["close"]), panels["close"].shape, "mcap" in panels)
    if key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key]
    o, h, l, c, v = (panels[k] for k in ("open", "high", "low", "close", "volume"))
    f: dict[str, pd.DataFrame] = {}
    prev_c = c.shift(1)
    f["ret1"] = c / prev_c - 1.0
    f["gap"] = o / prev_c - 1.0
    f["body"] = c / o - 1.0
    rng = h - l
    f["close_pos"] = ((c - l) / rng.where(rng > 0)).fillna(1.0)
    f["vol_avg20_prev"] = sma(v, 20).shift(1)
    f["vol_avg30_prev"] = sma(v, 30).shift(1)
    f["vol_ratio"] = v / f["vol_avg20_prev"]
    f["vol_ratio30"] = v / f["vol_avg30_prev"]
    f["vol_ratio_prev"] = v / v.shift(1)
    f["vol_max20_prev"] = rolling_max(v, 20).shift(1)
    f["vol_min20_prev"] = rolling_min(v, 20).shift(1)
    f["dv"] = c * v
    f["adv20"] = (c * v).rolling(20, min_periods=20).mean()
    f["adv20_prev"] = f["adv20"].shift(1)
    f["dv_max30"] = rolling_max(c * v, 30)
    for n in (5, 10, 20, 50, 60, 200):
        f[f"sma{n}"] = sma(c, n)
    f["ema9"], f["ema20"] = ema(c, 9), ema(c, 20)
    f["rsi14"] = rsi(c, 14)
    f["rsi2"] = rsi(c, 2)
    f["high20_prev"] = rolling_max(h, 20).shift(1)
    f["high52_prev"] = rolling_max(h, 252).shift(1)
    f["close_max25"] = rolling_max(c, 25)
    f["close_max60_prev"] = rolling_max(c, 60).shift(1)
    f["close_min30_prev"] = rolling_min(c, 30).shift(1)
    f["ret5"], f["ret20"] = c / c.shift(5) - 1.0, c / c.shift(20) - 1.0
    f["runup30"] = c / f["close_min30_prev"] - 1.0  # 30거래일 저점 대비 상승률
    f["range40_prev"] = (rolling_max(c, 40) / rolling_min(c, 40) - 1.0).shift(1)  # 직전 40일 박스 폭 (횡보 판정)
    f["atr14"] = atr(h, l, c, 14)
    f["atr_pct"] = f["atr14"] / c
    up = (c > prev_c).to_numpy()
    consec = np.zeros(up.shape, dtype=float)
    for t in range(1, up.shape[0]):
        consec[t] = np.where(up[t], consec[t - 1] + 1.0, 0.0)
    f["consec_up"] = pd.DataFrame(consec, index=c.index, columns=c.columns)
    f["price"] = c
    if "mcap" in panels:
        f["mcap"] = panels["mcap"]
        f["shares"] = panels["mcap"] / c  # 발행주식수(≈float 상한) 근사
        f["turnover"] = v / f["shares"]  # 당일 거래량/발행주식수 (float rotation 근사, 1.0 = 발행주식 전량 손바뀜)
    f["days_from_high52"] = c / f["high52_prev"] - 1.0
    _FEATURE_CACHE.clear()
    _FEATURE_CACHE[key] = f
    return f


_EARN_CACHE: dict = {}


def earnings_mask(panels: dict[str, pd.DataFrame], days_after: int = 1) -> pd.DataFrame | None:
    """실적발표일(및 다음 days_after 거래일) 이면 True 인 패널. data/cache/earnings_dates.parquet 가 없으면 None."""
    from .data import CACHE_DIR

    f = CACHE_DIR / "earnings_dates.parquet"
    if not f.exists():
        return None
    c = panels["close"]
    key = (id(c), c.shape, days_after)
    if key in _EARN_CACHE:
        return _EARN_CACHE[key]
    e = pd.read_parquet(f)
    e["date"] = pd.to_datetime(e["date"])
    e = e[e["symbol"].isin(c.columns)]
    m = pd.DataFrame(False, index=c.index, columns=c.columns)
    pos = c.index.searchsorted(e["date"].to_numpy())
    cols = m.columns.get_indexer(e["symbol"])
    arr = m.to_numpy()
    for k, j in zip(pos, cols):
        for d in range(days_after + 1):
            if 0 <= k + d < len(c.index):
                arr[k + d, j] = True
    m = pd.DataFrame(arr, index=c.index, columns=c.columns)
    _EARN_CACHE.clear()
    _EARN_CACHE[key] = m
    return m


def _true_like(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=df.index, columns=df.columns)


def _mcap_ok(f, max_mcap=None, min_mcap=None, max_shares=None):
    m = _true_like(f["ret1"])
    if max_mcap is not None and "mcap" in f:
        m &= f["mcap"] <= max_mcap
    if min_mcap is not None and "mcap" in f:
        m &= f["mcap"] >= min_mcap
    if max_shares is not None and "shares" in f:
        m &= f["shares"] <= max_shares
    return m


def _fin(x: pd.DataFrame) -> pd.DataFrame:
    return x.fillna(False).astype(bool)


# ----------------------------------------------------------------------------- 1. 급등일 정의 계열 (추격 매수 검증)
def ross_5pillars(panels, f, min_ret=0.10, min_rvol=5.0, min_price=1.0, max_price=20.0, max_shares=20e6, min_dv=1e6, **_):
    """Ross Cameron 5 Pillars: RVOL>=5x(30일), 당일 +10%, $1~$20, 저유통(float<10~20M). 뉴스 조건은 검증 불가."""
    return _fin((f["ret1"] >= min_ret) & (f["vol_ratio30"] >= min_rvol) & (f["price"] >= min_price) & (f["price"] <= max_price)
                & (f["dv"] >= min_dv) & _mcap_ok(f, max_shares=max_shares))


def warrior_scanner(panels, f, min_ret=0.05, min_rvol=1.5, min_price=1.0, max_price=30.0, min_vol=200_000, min_close_pos=0.7, **_):
    """Warrior Trading 모멘텀 스캐너(ToS 버전): +5%, RVOL 1.5, $1~$30, 20만주, 윗꼬리 짧은 양봉, 9EMA>20EMA."""
    return _fin((f["ret1"] >= min_ret) & (f["vol_ratio"] >= min_rvol) & (f["price"] >= min_price) & (f["price"] <= max_price)
                & (panels["volume"] >= min_vol) & (f["close_pos"] >= min_close_pos) & (f["body"] > 0) & (f["ema9"] > f["ema20"]))


def finviz_top_gainer(panels, f, min_ret=0.20, min_avg_vol=100_000, min_price=5.0, min_vol=2_000_000, min_rsi=60.0, **_):
    """Finviz 급등주 스크리너(국내 블로그 소개): +20%, 평균거래량 10만+, $5+, 거래량 200만+, >SMA20/50, RSI>60."""
    return _fin((f["ret1"] >= min_ret) & (f["vol_avg20_prev"] >= min_avg_vol) & (f["price"] >= min_price) & (panels["volume"] >= min_vol)
                & (f["price"] > f["sma20"]) & (f["price"] > f["sma50"]) & (f["rsi14"] >= min_rsi))


def kr_screener_basic(panels, f, min_ret=0.05, min_vol=500_000, min_rvol=2.0, max_rsi=70.0, min_price=1.0, **_):
    """국내 블로그 '자동 스크리너 조건': $1+, +5%, 50만주+, RVOL 2+, RSI 70 이하."""
    return _fin((f["ret1"] >= min_ret) & (panels["volume"] >= min_vol) & (f["vol_ratio"] >= min_rvol) & (f["rsi14"] <= max_rsi) & (f["price"] >= min_price))


def kr_newhigh_alignment(panels, f, max_mcap=1e9, min_vol=1_000_000, min_dv=2e6, near_max=0.97, min_vol_ratio_prev=2.0, **_):
    """국내 HTS 급등주 검색식 예시: 시총 상한, 거래량 100만+, 거래대금, 25일 최고종가 -3% 이내, 신고가, 정배열, 전일比 거래량 200%."""
    aligned = (f["sma5"] > f["sma20"]) & (f["sma20"] > f["sma60"])
    return _fin(_mcap_ok(f, max_mcap=max_mcap) & (panels["volume"] >= min_vol) & (f["dv"] >= min_dv) & (f["price"] >= near_max * f["close_max25"])
                & (f["price"] >= f["high52_prev"]) & aligned & (f["vol_ratio_prev"] >= min_vol_ratio_prev))


def big_candle_maxvol(panels, f, min_ret=0.20, min_close_pos=0.6, min_price=1.0, max_mcap=2e9, min_dv=2e6, **_):
    """장대양봉(+20%) + 최근 20일 최대 거래량 + 고가권 마감 (국내 카페 '대시세 첫날')."""
    return _fin((f["ret1"] >= min_ret) & (panels["volume"] >= f["vol_max20_prev"]) & (f["close_pos"] >= min_close_pos)
                & (f["price"] >= min_price) & (f["dv"] >= min_dv) & _mcap_ok(f, max_mcap=max_mcap))


def volume_3x_after_base(panels, f, min_rvol=3.0, max_range40=0.25, min_ret=0.05, min_price=1.0, max_mcap=2e9, min_dv=1e6, **_):
    """'2개월 이상 횡보 후 거래량 3배 급증' (국내 카페 급등주 발굴법)."""
    return _fin((f["vol_ratio"] >= min_rvol) & (f["range40_prev"] <= max_range40) & (f["ret1"] >= min_ret) & (f["price"] >= min_price)
                & (f["dv"] >= min_dv) & _mcap_ok(f, max_mcap=max_mcap))


# ----------------------------------------------------------------------------- 2. 눌림목 계열
def _last_surge(panels, f, surge_pct, vol_mult, window, min_dv, max_mcap):
    c, v = panels["close"], panels["volume"]
    surge = _fin((f["ret1"] >= surge_pct) & (f["vol_ratio"] >= vol_mult) & (f["dv"] >= min_dv) & _mcap_ok(f, max_mcap=max_mcap))
    s_close = c.where(surge).ffill(limit=window)
    s_vol = v.where(surge).ffill(limit=window)
    s_pre = c.shift(1).where(surge).ffill(limit=window)
    return surge, s_close, s_vol, s_pre


def pullback_lowvol(panels, f, surge_pct=0.15, vol_mult=5.0, window=5, vol_frac=0.2, hold_pct=0.9, min_dv=2e6, max_mcap=2e9, **_):
    """'장대양봉 후 3~5봉 안에 거래량이 장대양봉의 10~20% 이내로 마르고 주가는 버티면 급등 임박' (국내 카페/스레드)."""
    surge, s_close, s_vol, s_pre = _last_surge(panels, f, surge_pct, vol_mult, window, min_dv, max_mcap)
    c, v = panels["close"], panels["volume"]
    return _fin(~surge & s_close.notna() & (v <= vol_frac * s_vol) & (c >= hold_pct * s_close) & (c < s_close) & (c > s_pre))


def pullback_ma(panels, f, surge_pct=0.15, vol_mult=5.0, window=10, ma="sma10", vol_frac=0.5, min_dv=2e6, max_mcap=2e9, **_):
    """5/10일선 눌림: 급등 후 window일 안에 저가가 이평선에 닿고 종가는 이평선 위, 거래량은 급등일의 절반 이하."""
    surge, s_close, s_vol, s_pre = _last_surge(panels, f, surge_pct, vol_mult, window, min_dv, max_mcap)
    c, l, v = panels["close"], panels["low"], panels["volume"]
    m = f[ma]
    return _fin(~surge & s_close.notna() & (l <= m) & (c >= m) & (c < s_close) & (v <= vol_frac * s_vol) & (c > s_pre))


def threads_us_pullback(panels, f, max_mcap=1.8e9, min_price=0.001, max_price=30.0, min_runup30=0.9, min_dv_max30=65e6, **_):
    """스레드(@sec_stock) 미국 급등주 눌림 검색식: 시총 ≤ ₩2.5조, $0.001~$30, 30거래일 내 +90%, 30일 내 거래대금 ≥ ₩900억, 종가 5일선 상향돌파."""
    c = panels["close"]
    cross = (c > f["sma5"]) & (c.shift(1) <= f["sma5"].shift(1))
    return _fin(_mcap_ok(f, max_mcap=max_mcap) & (c >= min_price) & (c <= max_price) & (f["runup30"] >= min_runup30)
                & (f["dv_max30"] >= min_dv_max30) & cross)


# ----------------------------------------------------------------------------- 3. 매집 / 거래량 바닥 계열 (급등 '전' 포착)
def quiet_volume_spike(panels, f, max_abs_ret=0.03, min_rvol=3.0, min_price=1.0, max_mcap=2e9, min_dv=1e6, **_):
    """'주가 변동폭은 거의 없는데 말도 안 되는 거래량' = 세력 손바뀜/매집 흔적 (스레드 @jtdj_official2)."""
    return _fin((f["ret1"].abs() <= max_abs_ret) & (f["vol_ratio"] >= min_rvol) & (f["price"] >= min_price) & (f["dv"] >= min_dv)
                & _mcap_ok(f, max_mcap=max_mcap))


def dryup_first_spike(panels, f, dry_frac=0.5, min_rvol=3.0, min_ret=0.03, min_close_pos=0.6, min_price=1.0, max_mcap=2e9, min_dv=1e6, **_):
    """거래량 바닥(직전 20일 최저 거래량이 60일 평균의 절반 이하) 뒤 첫 거래량 급증 양봉 ('거래량 바닥 + 첫 급증')."""
    v = panels["volume"]
    dry = f["vol_min20_prev"] <= dry_frac * sma(v, 60).shift(1)
    return _fin(dry & (f["vol_ratio"] >= min_rvol) & (f["ret1"] >= min_ret) & (f["close_pos"] >= min_close_pos) & (f["price"] >= min_price)
                & (f["dv"] >= min_dv) & _mcap_ok(f, max_mcap=max_mcap))


# ----------------------------------------------------------------------------- 4. 오버나잇 / 멀티데이 계열
def first_green_day(panels, f, min_ret=0.15, min_rvol=5.0, min_close_pos=0.7, beaten=0.7, min_price=0.5, max_price=20.0, min_dv=2e6, **_):
    """Tim Sykes First Green Day: 하락해 있던 종목이 대량 거래로 급등, 고가 근처 마감 -> 종가 매수, 다음날 갭업에 매도."""
    prior_low = panels["close"].shift(1) <= beaten * f["close_max60_prev"]
    return _fin(prior_low & (f["ret1"] >= min_ret) & (f["vol_ratio"] >= min_rvol) & (f["close_pos"] >= min_close_pos)
                & (f["price"] >= min_price) & (f["price"] <= max_price) & (f["dv"] >= min_dv))


def multiday_runner_day2(panels, f, day1_ret=0.30, day1_rvol=10.0, min_dv=5e6, max_mcap=2e9, **_):
    """멀티데이 러너 'day-2 효과': 1일차 대급등(+30%, RVOL 10) 다음날 종가가 1일차 종가 이상(안 빠짐) -> 3일차부터 스윙."""
    c = panels["close"]
    day1 = _fin((f["ret1"] >= day1_ret) & (f["vol_ratio"] >= day1_rvol) & (f["dv"] >= min_dv) & _mcap_ok(f, max_mcap=max_mcap))
    return _fin(day1.shift(1) & (c >= c.shift(1)))


# ----------------------------------------------------------------------------- 5. 갭 계열 (open 이벤트: 당일 시가 진입)
def gap_and_go(panels, f, min_gap=0.10, max_gap=1.0, min_price=2.0, max_price=20.0, min_adv=1e6, max_shares=None, **_):
    """Gap & Go: 프리마켓 갭 ≥10%(완화판 4~5%), $2~$20, 거래량 동반. 프리마켓 고가 돌파 조건은 일봉으로 검증 불가 -> 시가 매수/종가 매도로 근사."""
    o = panels["open"]
    m = (f["gap"] >= min_gap) & (f["gap"] <= max_gap) & (o >= min_price) & (o <= max_price) & (f["adv20_prev"] >= min_adv)
    if max_shares is not None and "shares" in f:
        m &= f["shares"] <= max_shares
    return _fin(m)


SCREENS = {
    # name: (function, entry_mode)  entry_mode: next_open | close | open
    "ross_5pillars": (ross_5pillars, "next_open"),
    "warrior_scanner": (warrior_scanner, "next_open"),
    "finviz_top_gainer": (finviz_top_gainer, "next_open"),
    "kr_screener_basic": (kr_screener_basic, "next_open"),
    "kr_newhigh_alignment": (kr_newhigh_alignment, "next_open"),
    "big_candle_maxvol": (big_candle_maxvol, "next_open"),
    "volume_3x_after_base": (volume_3x_after_base, "next_open"),
    "pullback_lowvol": (pullback_lowvol, "next_open"),
    "pullback_ma": (pullback_ma, "next_open"),
    "threads_us_pullback": (threads_us_pullback, "next_open"),
    "quiet_volume_spike": (quiet_volume_spike, "next_open"),
    "dryup_first_spike": (dryup_first_spike, "next_open"),
    "first_green_day": (first_green_day, "close"),
    "multiday_runner_day2": (multiday_runner_day2, "next_open"),
    "gap_and_go": (gap_and_go, "open"),
}


# ----------------------------------------------------------------------------- 7. 커뮤니티 '개인 매매법' — 토스증권 미국주식이야기 "제 매매타이밍(방식) 공유" (거래량 기반 단기 패턴)
def _first_spike(panels, f, spike_mult, base_days, base_range, base_adv, min_abs_ret, max_mcap):
    """'첫 거래량' 날: 직전 base_days 동안 조용(박스폭 <= base_range, 평균 거래대금 <= base_adv)하던 종목에
    base_days 내 최대 거래량이면서 직전 20일 평균의 spike_mult 배 이상 거래량이 터지고 |당일 수익률| >= min_abs_ret."""
    c, v = panels["close"], panels["volume"]
    base_rng = (rolling_max(c, base_days) / rolling_min(c, base_days) - 1.0).shift(1)
    base_advv = (c * v).rolling(base_days, min_periods=base_days).mean().shift(1)
    quiet = (base_rng <= base_range) & (base_advv <= base_adv)
    spike = (f["vol_ratio"] >= spike_mult) & (v >= rolling_max(v, base_days).shift(1)) & (f["ret1"].abs() >= min_abs_ret)
    return _fin(quiet & spike & _mcap_ok(f, max_mcap=max_mcap))


def _days_since(event: pd.DataFrame, window: int) -> pd.DataFrame:
    """각 (일자, 종목) 에서 마지막 event 이후 지난 거래일 수 (window 초과면 NaN)."""
    pos = np.arange(len(event.index), dtype=float)[:, None]
    last = pd.DataFrame(np.where(event.to_numpy(), pos, np.nan), index=event.index, columns=event.columns).ffill(limit=window)
    return pd.DataFrame(pos - last.to_numpy(), index=event.index, columns=event.columns)


def toss_second_volume(panels, f, spike_mult=5.0, base_days=60, base_range=0.5, base_adv=2e6, min_abs_ret=0.05, max_mcap=2e9,
                       window=30, min_gap=2, second_mult=3.0, min_price=0.5, min_dv=1e6, **_):
    """'첫 거래량은 패스, 두 번째 거래량이 기회': 첫 급등(거래량 폭발)일로부터 min_gap~window 거래일 뒤,
    급등 전 20일 평균 대비 second_mult 배 이상 거래량이 다시 터지는 날 = 진입 신호(다음날 시가)."""
    c, v = panels["close"], panels["volume"]
    first = _first_spike(panels, f, spike_mult, base_days, base_range, base_adv, min_abs_ret, max_mcap)
    base_vol = f["vol_avg20_prev"].where(first).ffill(limit=window)  # 첫 급등 '전' 평균 거래량
    first_vol = v.where(first).ffill(limit=window)
    days_since = _days_since(first, window)
    second = (v >= second_mult * base_vol) & (v < first_vol) & (days_since >= min_gap) & (days_since <= window)
    return _fin(second & (c >= min_price) & (f["dv"] >= min_dv) & ~first)


def toss_pullback_after_green(panels, f, spike_mult=5.0, base_days=60, base_range=0.5, base_adv=2e6, min_ret=0.10, max_mcap=2e9,
                              day2_max_ret=-0.03, day2_vol_frac=0.5, min_price=0.5, **_):
    """첫날 '양봉 폭등' 다음날 '강한 음봉 + 거래량 급감(첫날의 절반 이하)' → 눌림목 진입(그 다음날 시가). 글쓴이의 주력 시나리오."""
    c, v = panels["close"], panels["volume"]
    first = _first_spike(panels, f, spike_mult, base_days, base_range, base_adv, min_ret, max_mcap) & (f["ret1"] >= min_ret)
    day2 = first.shift(1).fillna(False).astype(bool) & (f["ret1"] <= day2_max_ret) & (v <= day2_vol_frac * v.shift(1))
    return _fin(day2 & (c >= min_price))


def toss_dryup_after_red(panels, f, spike_mult=5.0, base_days=60, base_range=0.5, base_adv=2e6, min_abs_ret=0.05, max_mcap=2e9,
                         window=20, dry_frac=0.2, drop_from_first=0.4, revol_mult=2.0, min_price=0.5, **_):
    """첫날이 '음봉 마감' 급등(거래량 폭발 + 종가<시가) → 이후 거래량이 첫날의 dry_frac 이하로 소멸하며 음봉 지속 →
    (a) 첫 음봉 종가 대비 -drop_from_first 까지 눌리거나 (b) 거래량이 다시 붙는(전일 대비 revol_mult 배) 날 진입."""
    c, o, v = panels["close"], panels["open"], panels["volume"]
    first = _first_spike(panels, f, spike_mult, base_days, base_range, base_adv, min_abs_ret, max_mcap) & (c < o)
    first_close = c.where(first).ffill(limit=window)
    first_vol = v.where(first).ffill(limit=window)
    active = first_close.notna() & ~first
    dry_prev = (v.shift(1) <= dry_frac * first_vol) & (c.shift(1) < c.shift(2))  # 전일까지 거래량 소멸 + 하락 지속
    cond_a = c <= (1.0 - drop_from_first) * first_close
    cond_b = v >= revol_mult * v.shift(1)
    return _fin(active & dry_prev & (cond_a | cond_b) & (c >= min_price))


def toss_weak_green_rising_vol(panels, f, spike_mult=5.0, base_days=60, base_range=0.5, base_adv=2e6, min_abs_ret=0.05, max_mcap=2e9,
                               window=10, max_ret=0.05, min_price=0.5, **_):
    """첫날 급등 이후 window일 안에 '거래량 증가 + 약한 양봉'(0 < 수익률 <= max_ret, 거래량 > 전일) → 진입 후 보유
    (청산: 음봉 전환 또는 거래량이 전일의 2배 — FormulaStrategy exit_rule='toss' 로 구현)."""
    c, v = panels["close"], panels["volume"]
    first = _first_spike(panels, f, spike_mult, base_days, base_range, base_adv, min_abs_ret, max_mcap)
    first_close = c.where(first).ffill(limit=window)
    active = first_close.notna() & ~first
    return _fin(active & (f["ret1"] > 0) & (f["ret1"] <= max_ret) & (v > v.shift(1)) & (c >= min_price))


SCREENS.update({
    "toss_second_volume": (toss_second_volume, "next_open"),
    "toss_pullback_after_green": (toss_pullback_after_green, "next_open"),
    "toss_dryup_after_red": (toss_dryup_after_red, "next_open"),
    "toss_weak_green_rising_vol": (toss_weak_green_rising_vol, "next_open"),
})
