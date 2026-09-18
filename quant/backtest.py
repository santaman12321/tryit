"""포트폴리오 수준 일봉 백테스터 (이벤트 루프, look-ahead 없음).

하루(t)의 처리 순서
  1. 전일 종가 신호로 예약된 청산 -> t 시가 체결
  2. 전일 종가 신호로 예약된 진입 -> t 시가 체결 (intraday 전략은 t 시가 신호로 즉시 진입)
  3. 장중 손절/익절 판정: 저가 <= 손절가 -> min(시가, 손절가) 체결, 고가 >= 익절가 -> max(시가, 익절가) 체결
     (같은 날 둘 다 맞으면 보수적으로 손절 우선)
  4. 종가: intraday 전략은 전량 종가 청산. swing 전략은 트레일링 스탑 갱신, 청산 신호/보유일 초과 예약,
     다음날 진입 후보 선정(score 내림차순, 빈 슬롯만큼)
  5. 종가 기준 평가(mark-to-market)

hold_mode == "overnight": t 종가(마감 직전) 진입 -> t+1 시가 청산 (First Green Day 류 오버나잇 검증용).
공매도: strategy.direction == -1 이면 모든 진입이 공매도(다음날 시가 매도, 청산은 환매). 손절은 고가 기준, 익절은 저가 기준.
       대차비용 borrow_rate_annual 을 보유일수만큼 일할 차감. 공매도 대금은 담보로 묶여 현금 불변(레버리지 없음).
비용: 체결마다 slippage_bps 만큼 불리하게(매수 +, 매도 -), 체결가 < penny_price 면 penny_slippage_bps 적용,
      체결당 정액 commission USD + 정률 commission_pct(편도, 국내 증권사 해외주식 기본 0.25%). trades.ret_pct 는 모든 비용 차감 후.
사이징: "equal" = 직전 종가 자산 / max_positions, "risk" = 그 값과 (risk_per_trade*자산/손절거리) 중 작은 쪽.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategies.base import Signals, Strategy


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    max_positions: int = 10
    slippage_bps: float = 10.0
    penny_slippage_bps: float = 50.0  # 체결가 < penny_price 인 종목에 적용 (동전주 스프레드 반영)
    penny_price: float = 5.0
    commission: float = 0.0  # 체결당 정액 수수료 (USD)
    commission_pct: float = 0.0025  # 체결당 정률 수수료 (편도). 국내 증권사 해외주식 기본 0.25%, 이벤트 계좌 0.07~0.1%
    sizing: str = "equal"  # "equal" | "risk"
    risk_per_trade: float = 0.01
    max_adv_pct: float | None = 0.01  # 포지션 <= 20일 평균 거래대금 * 이 비율 (유동성 현실화). None=제한 없음
    borrow_rate_annual: float = 0.0  # 공매도 대차비용 (연율, 예: 1.0 = 100%/년). 보유일수 x 일할로 차감
    start: str | None = None
    end: str | None = None


@dataclass
class Position:
    col: int
    symbol: str
    shares: int
    entry_price: float
    entry_idx: int
    score: float
    stop: float | None = None
    target: float | None = None
    hold_days: int = 0
    last_price: float = 0.0
    direction: int = 1  # +1 롱, -1 숏


@dataclass
class BacktestResult:
    strategy: str
    params: dict
    config: BacktestConfig
    equity: pd.Series
    n_positions: pd.Series
    cash: pd.Series
    trades: pd.DataFrame
    extra: dict = field(default_factory=dict)


TRADE_COLUMNS = [
    "symbol", "entry_date", "exit_date", "entry_price", "exit_price", "shares",
    "pnl", "ret_pct", "hold_days", "reason", "score",
]


def _aligned(df: pd.DataFrame | None, idx, cols, fill=np.nan, dtype=float):
    if df is None:
        return None
    out = df.reindex(index=idx, columns=cols)
    if dtype is bool:
        return out.fillna(False).to_numpy(dtype=bool)
    return out.to_numpy(dtype=float)


def run_backtest(panels: dict[str, pd.DataFrame], signals: Signals, strategy: Strategy, config: BacktestConfig) -> BacktestResult:
    C = panels["close"]
    idx, cols = C.index, list(C.columns)
    O = panels["open"].reindex(index=idx, columns=cols).to_numpy(float)
    H = panels["high"].reindex(index=idx, columns=cols).to_numpy(float)
    L = panels["low"].reindex(index=idx, columns=cols).to_numpy(float)
    Cn = C.to_numpy(float)
    entry = _aligned(signals.entry, idx, cols, dtype=bool)
    exit_ = _aligned(signals.exit, idx, cols, dtype=bool)
    score = _aligned(signals.score, idx, cols)
    stop_dist = _aligned(signals.stop_dist, idx, cols)
    trail_dist = _aligned(signals.trail_dist, idx, cols)
    adv = _aligned(signals.adv, idx, cols)
    target_pct = signals.target_pct

    start_i = int(idx.searchsorted(pd.Timestamp(config.start))) if config.start else 0
    end_i = int(idx.searchsorted(pd.Timestamp(config.end), side="right")) if config.end else len(idx)
    if end_i <= start_i:
        raise ValueError("empty backtest window")

    slip_base = config.slippage_bps / 1e4
    slip_penny = config.penny_slippage_bps / 1e4
    penny_price = config.penny_price

    def slip_for(price: float) -> float:
        return slip_penny if price < penny_price else slip_base

    comm = config.commission
    cpct = config.commission_pct
    max_pos = config.max_positions
    intraday = strategy.hold_mode == "intraday"
    overnight = strategy.hold_mode == "overnight"
    direction = -1 if getattr(strategy, "direction", 1) == -1 else 1
    max_hold = strategy.max_hold_days
    borrow_daily = config.borrow_rate_annual / 252.0

    cash = float(config.initial_cash)
    last_equity = cash
    positions: dict[int, Position] = {}
    pending_entries: list[tuple[int, float, float, float]] = []  # (col, score, stop_dist, adv)
    pending_exits: dict[int, str] = {}
    trades: list[dict] = []
    eq_hist = np.empty(end_i - start_i)
    npos_hist = np.empty(end_i - start_i, dtype=int)
    cash_hist = np.empty(end_i - start_i)

    def close_position(pos: Position, raw_price: float, i: int, reason: str, costs: bool = True) -> None:
        nonlocal cash
        d = pos.direction
        # 롱 청산은 매도(가격 -slip), 숏 청산(환매)은 매수(가격 +slip)
        px = raw_price * (1.0 - d * slip_for(raw_price)) if costs else raw_price
        fee = (comm + cpct * pos.shares * px) if costs else 0.0
        entry_fee = comm + cpct * pos.shares * pos.entry_price
        borrow = pos.shares * pos.entry_price * borrow_daily * max(i - pos.entry_idx, 1) if d == -1 else 0.0
        pnl = d * pos.shares * (px - pos.entry_price) - entry_fee - fee - borrow
        if d == 1:
            cash += pos.shares * px - fee
        else:
            # 진입 때 공매도 대금은 담보로 묶어 두었으므로(cash 미변동) 여기서 손익만 반영
            cash += pnl + entry_fee  # 진입 수수료는 진입 시 이미 차감됨
        trades.append(
            {
                "symbol": pos.symbol,
                "entry_date": idx[pos.entry_idx],
                "exit_date": idx[i],
                "entry_price": pos.entry_price,
                "exit_price": px,
                "shares": pos.shares * d,
                "pnl": pnl,
                "ret_pct": pnl / (pos.shares * pos.entry_price),  # 슬리피지·수수료·대차비용 모두 차감한 순수익률
                "hold_days": i - pos.entry_idx,
                "reason": reason,
                "score": pos.score,
            }
        )
        del positions[pos.col]

    def open_position(col: int, sc: float, sd: float, adv_i: float, raw_price: float, i: int) -> None:
        nonlocal cash
        if len(positions) >= max_pos or col in positions or math.isnan(raw_price):
            return
        px = raw_price * (1.0 + direction * slip_for(raw_price))
        alloc = last_equity / max_pos
        has_stop = sd is not None and not math.isnan(sd) and sd > 0
        if config.sizing == "risk" and has_stop:
            alloc = min(alloc, config.risk_per_trade * last_equity / sd * px)
        if config.max_adv_pct is not None and adv_i is not None and not math.isnan(adv_i):
            alloc = min(alloc, config.max_adv_pct * adv_i)
        budget = min(alloc, cash) - comm
        shares = int(budget // (px * (1.0 + cpct)))
        if shares < 1:
            return
        entry_fee = comm + cpct * shares * px
        if direction == 1:
            cash -= shares * px + entry_fee
        else:
            cash -= entry_fee  # 공매도: 대금은 담보로 묶임(현금 불변), 손익은 청산 시 반영. 레버리지 없음(alloc <= 현금)
        positions[col] = Position(
            col=col,
            symbol=cols[col],
            shares=shares,
            entry_price=px,
            entry_idx=i,
            score=sc,
            stop=(px - direction * sd) if has_stop else None,
            target=(px * (1.0 + direction * target_pct)) if target_pct else None,
            last_price=px,
            direction=direction,
        )

    for k, i in enumerate(range(start_i, end_i)):
        o, h, l, c = O[i], H[i], L[i], Cn[i]

        # 1. 예약된 청산 (시가) — overnight 모드는 전일 종가 진입분을 여기서 전량 청산
        if overnight:
            for col in list(positions):
                pending_exits.setdefault(col, "open")
        for col, reason in list(pending_exits.items()):
            pos = positions.get(col)
            if pos is None:
                del pending_exits[col]
            elif not math.isnan(o[col]):
                close_position(pos, o[col], i, reason)
                del pending_exits[col]

        # 2. 진입 (시가)
        if intraday:
            cands = [(score[i, col] if not math.isnan(score[i, col]) else -math.inf, col) for col in np.flatnonzero(entry[i])]
            cands.sort(key=lambda t: -t[0])
            pending_entries = [(col, sc, stop_dist[i, col] if stop_dist is not None else math.nan,
                                adv[i, col] if adv is not None else math.nan) for sc, col in cands]
        for col, sc, sd, adv_i in pending_entries:
            if len(positions) >= max_pos:
                break
            open_position(col, sc, sd, adv_i, o[col], i)
        pending_entries = []
        n_pos_today = len(positions)

        # 3. 장중 손절 / 익절
        for col, pos in list(positions.items()):
            if math.isnan(l[col]) or math.isnan(h[col]):
                continue
            if pos.direction == 1:
                if pos.stop is not None and l[col] <= pos.stop:
                    fill = min(o[col], pos.stop) if not math.isnan(o[col]) else pos.stop
                    close_position(pos, fill, i, "stop")
                elif pos.target is not None and h[col] >= pos.target:
                    fill = max(o[col], pos.target) if not math.isnan(o[col]) else pos.target
                    close_position(pos, fill, i, "target")
            else:  # 숏: 고가가 손절가 이상이면 손절, 저가가 목표가 이하면 익절
                if pos.stop is not None and h[col] >= pos.stop:
                    fill = max(o[col], pos.stop) if not math.isnan(o[col]) else pos.stop
                    close_position(pos, fill, i, "stop")
                elif pos.target is not None and l[col] <= pos.target:
                    fill = min(o[col], pos.target) if not math.isnan(o[col]) else pos.target
                    close_position(pos, fill, i, "target")

        # 4. 종가 처리
        if intraday:
            for col, pos in list(positions.items()):
                if not math.isnan(c[col]):
                    close_position(pos, c[col], i, "close")
        elif overnight:
            # 당일 종가 신호 -> 당일 종가(마감 직전) 진입, 다음날 시가 청산
            cands = [(score[i, col] if not math.isnan(score[i, col]) else -math.inf, col) for col in np.flatnonzero(entry[i]) if col not in positions]
            cands.sort(key=lambda t: -t[0])
            for sc, col in cands[: max_pos - len(positions)]:
                open_position(col, sc, math.nan, adv[i, col] if adv is not None else math.nan, c[col], i)
            n_pos_today = len(positions)
        else:
            for col, pos in positions.items():
                pos.hold_days += 1
                cc = c[col]
                if trail_dist is not None and not math.isnan(cc):
                    td = trail_dist[i, col]
                    if not math.isnan(td):
                        if pos.direction == 1:
                            ts = cc - td
                            pos.stop = ts if pos.stop is None else max(pos.stop, ts)
                        else:
                            ts = cc + td
                            pos.stop = ts if pos.stop is None else min(pos.stop, ts)
                if col in pending_exits:
                    continue
                if exit_[i, col]:
                    pending_exits[col] = "signal"
                elif max_hold is not None and pos.hold_days >= max_hold:
                    pending_exits[col] = "time"
            free = max_pos - len(positions) + len(pending_exits)
            if free > 0:
                cands = []
                for col in np.flatnonzero(entry[i]):
                    if col in positions:
                        continue
                    sc = score[i, col]
                    cands.append((sc if not math.isnan(sc) else -math.inf, col))
                cands.sort(key=lambda t: -t[0])
                pending_entries = [(col, sc, stop_dist[i, col] if stop_dist is not None else math.nan,
                                    adv[i, col] if adv is not None else math.nan) for sc, col in cands[:free]]

        # 5. 평가
        mv = 0.0
        for col, pos in positions.items():
            if not math.isnan(c[col]):
                pos.last_price = c[col]
            if pos.direction == 1:
                mv += pos.shares * pos.last_price
            else:
                mv += pos.shares * (pos.entry_price - pos.last_price) - pos.shares * pos.entry_price * borrow_daily * (i - pos.entry_idx)
        last_equity = cash + mv
        eq_hist[k] = last_equity
        npos_hist[k] = n_pos_today
        cash_hist[k] = cash

    # 종료: 잔여 포지션을 마지막 평가가로 청산 (평가 목적이므로 비용 없음 -> 자산곡선과 거래손익 합이 일치)
    for col, pos in list(positions.items()):
        close_position(pos, pos.last_price, end_i - 1, "end", costs=False)

    window = idx[start_i:end_i]
    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return BacktestResult(
        strategy=strategy.name,
        params=dict(strategy.params),
        config=config,
        equity=pd.Series(eq_hist, index=window, name="equity"),
        n_positions=pd.Series(npos_hist, index=window, name="n_positions"),
        cash=pd.Series(cash_hist, index=window, name="cash"),
        trades=trades_df,
    )
