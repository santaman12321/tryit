"""합성 데이터로 백테스터의 체결 규칙을 검증한다."""
import math

import numpy as np
import pandas as pd
import pytest

from quant.backtest import BacktestConfig, run_backtest
from quant.strategies.base import Signals, Strategy


def make_panels(prices: dict[str, list[float]], spread: float = 0.0):
    """종가 리스트로 패널 생성. open=전일 종가(첫날은 종가), high/low = close ± spread."""
    idx = pd.bdate_range("2025-01-01", periods=len(next(iter(prices.values()))))
    close = pd.DataFrame(prices, index=idx, dtype=float)
    open_ = close.shift(1).fillna(close)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = pd.DataFrame(1_000_000.0, index=idx, columns=close.columns)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": vol}


class Fixed(Strategy):
    name = "fixed"
    defaults = {"max_hold_days": None}

    def __init__(self, signals: Signals, hold_mode="swing", **params):
        super().__init__(**params)
        self._signals = signals
        self.hold_mode = hold_mode

    def generate(self, panels, bench):
        return self._signals


def bool_frame(like, true_at: dict[str, list[int]]):
    df = pd.DataFrame(False, index=like.index, columns=like.columns)
    for sym, rows in true_at.items():
        for r in rows:
            df.iloc[r, df.columns.get_loc(sym)] = True
    return df


def cfg(**kw):
    base = dict(initial_cash=10_000.0, max_positions=2, slippage_bps=0.0, penny_slippage_bps=0.0, commission=0.0, commission_pct=0.0)
    base.update(kw)
    return BacktestConfig(**base)


def test_swing_entry_fills_next_open_and_exit_next_open():
    p = make_panels({"A": [10, 11, 12, 13, 14, 15]})
    c = p["close"]
    sig = Signals(entry=bool_frame(c, {"A": [1]}), exit=bool_frame(c, {"A": [3]}), score=c * 0 + 1)
    res = run_backtest(p, sig, Fixed(sig), cfg())
    t = res.trades.iloc[0]
    # 신호일(1) 다음날(2) 시가 = 전일 종가 11 에 진입, 청산 신호(3) 다음날(4) 시가 = 13 에 청산
    assert t["entry_price"] == pytest.approx(11.0)
    assert t["exit_price"] == pytest.approx(13.0)
    assert t["entry_date"] == c.index[2] and t["exit_date"] == c.index[4]
    assert t["reason"] == "signal"
    # 5000 배분 -> 454주
    assert t["shares"] == 454
    assert res.equity.iloc[-1] == pytest.approx(10_000 + 454 * 2.0)


def test_no_lookahead_signal_on_last_day_never_fills():
    p = make_panels({"A": [10, 11, 12]})
    c = p["close"]
    sig = Signals(entry=bool_frame(c, {"A": [2]}), exit=bool_frame(c, {}), score=c * 0 + 1)
    res = run_backtest(p, sig, Fixed(sig), cfg())
    assert res.trades.empty


def test_stop_fills_at_stop_or_gap_open():
    # 진입 후 저가가 손절가에 닿으면 손절가 체결
    p = make_panels({"A": [10, 10, 10, 10, 10]}, spread=0.0)
    p["low"].iloc[3, 0] = 9.0  # 3일차 저가 9
    c = p["close"]
    stop_dist = c * 0 + 0.5  # 손절 거리 0.5 -> 진입가 10 -> 손절가 9.5
    sig = Signals(entry=bool_frame(c, {"A": [1]}), exit=bool_frame(c, {}), score=c * 0 + 1, stop_dist=stop_dist)
    res = run_backtest(p, sig, Fixed(sig), cfg())
    t = res.trades.iloc[0]
    assert t["reason"] == "stop" and t["exit_price"] == pytest.approx(9.5)
    # 갭하락: 시가가 손절가 아래면 시가 체결
    p2 = make_panels({"A": [10, 10, 10, 8, 8]}, spread=0.0)
    p2["open"].iloc[3, 0] = 8.0
    p2["low"].iloc[3, 0] = 7.9
    res2 = run_backtest(p2, sig, Fixed(sig), cfg())
    assert res2.trades.iloc[0]["exit_price"] == pytest.approx(8.0)


def test_max_positions_and_score_ranking():
    p = make_panels({"A": [10] * 6, "B": [10] * 6, "C": [10] * 6})
    c = p["close"]
    score = pd.DataFrame({"A": 1.0, "B": 3.0, "C": 2.0}, index=c.index)
    sig = Signals(entry=bool_frame(c, {"A": [1], "B": [1], "C": [1]}), exit=bool_frame(c, {}), score=score)
    res = run_backtest(p, sig, Fixed(sig), cfg(max_positions=2))
    assert sorted(res.trades["symbol"]) == ["B", "C"]
    assert res.n_positions.max() == 2


def test_time_exit():
    p = make_panels({"A": [10] * 10})
    c = p["close"]
    sig = Signals(entry=bool_frame(c, {"A": [0]}), exit=bool_frame(c, {}), score=c * 0 + 1)
    res = run_backtest(p, sig, Fixed(sig, max_hold_days=3), cfg())
    t = res.trades.iloc[0]
    # 1일 시가 진입, 종가 3번(1,2,3일) 후 예약 -> 4일 시가 청산 => hold_days = 3
    assert t["reason"] == "time" and t["hold_days"] == 3


def test_trailing_stop_ratchets_up():
    p = make_panels({"A": [10, 10, 12, 14, 16, 13, 13]}, spread=0.0)
    c = p["close"]
    trail = c * 0 + 1.0  # stop = max(stop, close-1)
    sig = Signals(entry=bool_frame(c, {"A": [0]}), exit=bool_frame(c, {}), score=c * 0 + 1, trail_dist=trail)
    res = run_backtest(p, sig, Fixed(sig), cfg())
    t = res.trades.iloc[0]
    # 4일 종가 16 -> stop 15; 5일 시가 16, 저가 13 -> 15 에 손절
    assert t["reason"] == "stop" and t["exit_price"] == pytest.approx(15.0)


def test_intraday_mode_enters_open_exits_close():
    p = make_panels({"A": [10, 11, 12]}, spread=0.0)
    p["open"].iloc[1, 0] = 10.5
    c = p["close"]
    sig = Signals(entry=bool_frame(c, {"A": [1]}), exit=bool_frame(c, {}), score=c * 0 + 1)
    res = run_backtest(p, sig, Fixed(sig, hold_mode="intraday"), cfg())
    t = res.trades.iloc[0]
    assert t["entry_price"] == pytest.approx(10.5) and t["exit_price"] == pytest.approx(11.0)
    assert t["hold_days"] == 0 and t["reason"] == "close"
    assert res.n_positions.iloc[1] == 1 and res.n_positions.iloc[2] == 0


def test_slippage_and_commission_accounting():
    p = make_panels({"A": [10, 10, 10, 10]})
    c = p["close"]
    sig = Signals(entry=bool_frame(c, {"A": [0]}), exit=bool_frame(c, {"A": [1]}), score=c * 0 + 1)
    res = run_backtest(p, sig, Fixed(sig), cfg(slippage_bps=100, commission=1.0))
    t = res.trades.iloc[0]
    assert t["entry_price"] == pytest.approx(10.1) and t["exit_price"] == pytest.approx(9.9)
    shares = int((5000 - 1.0) // 10.1)
    assert t["shares"] == shares
    assert res.equity.iloc[-1] == pytest.approx(10_000 + shares * (9.9 - 10.1) - 2.0)


def test_cash_never_negative_and_equity_consistent():
    rng = np.random.default_rng(0)
    n, m = 120, 8
    closes = {f"S{i}": (100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))).tolist() for i in range(m)}
    p = make_panels(closes, spread=0.5)
    c = p["close"]
    entry = pd.DataFrame(rng.random((n, m)) < 0.1, index=c.index, columns=c.columns)
    exit_ = pd.DataFrame(rng.random((n, m)) < 0.15, index=c.index, columns=c.columns)
    sig = Signals(entry=entry, exit=exit_, score=c.pct_change(5), stop_dist=c * 0.05, trail_dist=c * 0.08)
    res = run_backtest(p, sig, Fixed(sig, max_hold_days=7), cfg(max_positions=3, slippage_bps=5))
    assert (res.cash >= -1e-6).all()
    assert res.n_positions.max() <= 3
    assert len(res.trades) > 5
    # 최종 자산 = 초기자금 + 모든 거래 손익
    assert res.equity.iloc[-1] == pytest.approx(10_000 + res.trades["pnl"].sum(), rel=1e-9)


def test_adv_cap_limits_position_size():
    p = make_panels({"A": [10] * 5})
    c = p["close"]
    adv = c * 0 + 20_000.0  # 평균 거래대금 $20k -> 1% = $200 -> 20주
    sig = Signals(entry=bool_frame(c, {"A": [0]}), exit=bool_frame(c, {}), score=c * 0 + 1, adv=adv)
    res = run_backtest(p, sig, Fixed(sig), cfg(max_adv_pct=0.01))
    assert res.trades.iloc[0]["shares"] == 20
    res2 = run_backtest(p, sig, Fixed(sig), cfg(max_adv_pct=None))
    assert res2.trades.iloc[0]["shares"] == 500


def test_surge_strategies_run_on_synthetic_data():
    from quant.strategies import make_strategy

    rng = np.random.default_rng(1)
    n = 80
    closes = {f"S{i}": (5 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))).tolist() for i in range(4)}
    closes["S0"][40] = closes["S0"][39] * 1.30  # 급등일
    closes["S0"][41:45] = [closes["S0"][40] * 0.9] * 4  # 눌림
    p = make_panels(closes, spread=0.1)
    p["volume"].iloc[40, 0] = 50_000_000
    p["volume"] = p["volume"].where(p["volume"] > 0, 1_000_000)
    p["mcap"] = p["close"] * 10_000_000
    bench = pd.DataFrame({"close": p["close"]["S1"]})
    for name in ["surge_chase", "surge_pullback"]:
        strat = make_strategy(name, min_dollar_volume=0, min_day_dollar_volume=0)
        sig = strat.generate(p, bench)
        assert sig.entry["S0"].any(), name
        res = run_backtest(p, sig, strat, cfg(slippage_bps=0))
        assert len(res.trades) >= 1, name
    # 시총 상한을 낮추면 신호 없음
    strat = make_strategy("surge_chase", min_dollar_volume=0, min_day_dollar_volume=0, max_mcap=1.0)
    assert not strat.generate(p, bench).entry.any().any()


def test_overnight_mode_enters_close_exits_next_open():
    p = make_panels({"A": [10, 11, 12, 13]}, spread=0.0)
    p["open"].iloc[2, 0] = 11.5
    c = p["close"]
    sig = Signals(entry=bool_frame(c, {"A": [1]}), exit=bool_frame(c, {}), score=c * 0 + 1)
    res = run_backtest(p, sig, Fixed(sig, hold_mode="overnight"), cfg())
    t = res.trades.iloc[0]
    assert t["entry_price"] == pytest.approx(11.0) and t["exit_price"] == pytest.approx(11.5)
    assert t["hold_days"] == 1 and t["reason"] == "open"
    assert len(res.trades) == 1


def test_penny_slippage_applies_below_threshold():
    p = make_panels({"A": [3, 3, 3, 3], "B": [30, 30, 30, 30]})
    c = p["close"]
    sig = Signals(entry=bool_frame(c, {"A": [0], "B": [0]}), exit=bool_frame(c, {"A": [1], "B": [1]}), score=c * 0 + 1)
    res = run_backtest(p, sig, Fixed(sig), cfg(slippage_bps=10, penny_slippage_bps=100, penny_price=5.0))
    a = res.trades[res.trades.symbol == "A"].iloc[0]
    b = res.trades[res.trades.symbol == "B"].iloc[0]
    assert a["entry_price"] == pytest.approx(3 * 1.01) and a["exit_price"] == pytest.approx(3 * 0.99)
    assert b["entry_price"] == pytest.approx(30 * 1.001) and b["exit_price"] == pytest.approx(30 * 0.999)


def test_short_direction_pnl_stop_and_borrow():
    # 숏: 10 에 공매도 -> 8 에 환매 = +20%
    p = make_panels({"A": [10, 10, 8, 8, 8]}, spread=0.0)
    c = p["close"]
    sig = Signals(entry=bool_frame(c, {"A": [0]}), exit=bool_frame(c, {"A": [2]}), score=c * 0 + 1)
    strat = Fixed(sig)
    strat.direction = -1
    res = run_backtest(p, sig, strat, cfg())
    t = res.trades.iloc[0]
    assert t["entry_price"] == pytest.approx(10.0) and t["exit_price"] == pytest.approx(8.0)
    assert t["shares"] == -500 and t["pnl"] == pytest.approx(500 * 2.0) and t["ret_pct"] == pytest.approx(0.2)
    assert res.equity.iloc[-1] == pytest.approx(11_000.0)
    assert (res.cash >= 0).all()
    # 숏 손절: 고가가 손절가(진입가+dist) 이상이면 손절가에 환매
    p2 = make_panels({"A": [10, 10, 10, 10, 10]}, spread=0.0)
    p2["high"].iloc[2, 0] = 11.5
    sig2 = Signals(entry=bool_frame(c, {"A": [0]}), exit=bool_frame(c, {}), score=c * 0 + 1, stop_dist=c * 0 + 1.0)
    res2 = run_backtest(p2, sig2, strat, cfg())
    t2 = res2.trades.iloc[0]
    assert t2["reason"] == "stop" and t2["exit_price"] == pytest.approx(11.0) and t2["ret_pct"] == pytest.approx(-0.1)
    # 대차비용: 연 252% = 하루 1% -> 2일 보유 시 2% 차감
    res3 = run_backtest(p, sig, strat, cfg(borrow_rate_annual=2.52))
    t3 = res3.trades.iloc[0]
    assert t3["hold_days"] == 2 and t3["ret_pct"] == pytest.approx(0.2 - 0.02)
    assert res3.equity.iloc[-1] == pytest.approx(10_000 + 500 * 2.0 - 500 * 10 * 0.02)


def test_percentage_commission_korean_broker():
    p = make_panels({"A": [10, 10, 10, 10]})
    c = p["close"]
    sig = Signals(entry=bool_frame(c, {"A": [0]}), exit=bool_frame(c, {"A": [1]}), score=c * 0 + 1)
    res = run_backtest(p, sig, Fixed(sig), cfg(commission_pct=0.0025))
    t = res.trades.iloc[0]
    shares = int(5000 // (10 * 1.0025))
    assert t["shares"] == shares
    fees = 2 * 0.0025 * shares * 10
    assert t["pnl"] == pytest.approx(-fees) and t["ret_pct"] == pytest.approx(-0.005)
    assert res.equity.iloc[-1] == pytest.approx(10_000 - fees)
