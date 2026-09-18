"""일일 실행 로직: 장 마감 후 신호 계산 -> 다음날 시가 부근 지정가 주문 계획.

백테스터와 같은 Strategy.generate() 를 쓰므로 백테스트와 실거래 로직이 어긋나지 않는다.
공매도 없음(국내 증권사 해외주식). 손절은 브로커 스탑 주문 대신 매 실행 시 종가 기준으로 판정해 다음날 매도한다
(장중 손절이 필요하면 15분봉 감시 프로세스를 별도로 두어야 한다 — README 참고).

상태 파일(state/<strategy>.json)
  positions: {symbol: {qty, entry_date, entry_price, stop, hold_days}}, cash, equity (dryrun 에서만 의미)
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from .backtest import BacktestConfig
from .broker import Broker, Order
from .strategies.base import Strategy

log = logging.getLogger(__name__)


@dataclass
class Plan:
    as_of: str
    exits: list[Order] = field(default_factory=list)
    entries: list[Order] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def load_state(path: Path, initial_cash: float) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"positions": {}, "cash": initial_cash, "equity": initial_cash}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str))


def plan_orders(strategy: Strategy, stocks: dict, bench: pd.DataFrame, broker: Broker, state: dict, config: BacktestConfig,
                limit_buffer: float = 0.01, exchange_of: dict[str, str] | None = None) -> Plan:
    """마지막 봉(오늘 종가)의 신호로 내일 주문을 만든다. swing 전략 전용. 지정가 = 종가 ± limit_buffer."""
    if strategy.hold_mode != "swing":
        raise ValueError("intraday/overnight 전략은 장중 실행이 필요해 일일 스케줄러로 돌릴 수 없습니다")
    if getattr(strategy, "direction", 1) == -1:
        raise ValueError("공매도 전략은 국내 증권사 해외주식 API 에서 실행할 수 없습니다")
    sig = strategy.generate(stocks, bench)
    close = stocks["close"]
    last = close.index[-1]
    entry, exit_, score = sig.entry.loc[last], sig.exit.loc[last], sig.score.loc[last]
    stop_dist = sig.stop_dist.loc[last] if sig.stop_dist is not None else None
    trail = sig.trail_dist.loc[last] if sig.trail_dist is not None else None
    adv = sig.adv.loc[last] if sig.adv is not None else None
    exchange_of = exchange_of or {}

    acct = broker.account()
    held = broker.positions()
    st_pos = state.setdefault("positions", {})
    plan = Plan(as_of=str(last.date()))

    # 브로커 포지션과 상태 동기화 (수동 매매/체결 반영)
    for sym, p in held.items():
        if sym not in st_pos:
            st_pos[sym] = {"qty": p["qty"], "entry_date": plan.as_of, "entry_price": p["avg_price"], "stop": None, "hold_days": 0}
            plan.notes.append(f"{sym}: 브로커 포지션을 상태에 새로 등록")
        else:
            st_pos[sym]["qty"] = p["qty"]
    if broker.name != "dryrun":
        for sym in list(st_pos):
            if sym not in held:
                plan.notes.append(f"{sym}: 브로커에 없음 -> 상태에서 제거")
                del st_pos[sym]

    # 1) 청산: 신호 / 보유일 초과 / 종가가 손절가 이하
    for sym, p in st_pos.items():
        p["hold_days"] = int(p.get("hold_days", 0)) + 1
        c = float(close[sym].iloc[-1]) if sym in close.columns else float("nan")
        if trail is not None and sym in trail.index and not math.isnan(float(trail[sym])) and not math.isnan(c):
            ts = c - float(trail[sym])
            p["stop"] = ts if p.get("stop") is None else max(float(p["stop"]), ts)
        reason = None
        if sym in exit_.index and bool(exit_[sym]):
            reason = "signal"
        elif strategy.max_hold_days is not None and p["hold_days"] >= strategy.max_hold_days:
            reason = "time"
        elif p.get("stop") is not None and not math.isnan(c) and c <= float(p["stop"]):
            reason = "stop(close)"
        if reason:
            plan.exits.append(Order(sym, int(p["qty"]), "sell", round(c * (1 - limit_buffer), 2), exchange_of.get(sym, "NASD"), reason))
            p["pending_exit"] = reason

    # 2) 진입: 빈 슬롯만큼 score 순
    n_open = len([s for s, p in st_pos.items() if not p.get("pending_exit")])
    free = config.max_positions - n_open
    cands = [(float(score[s]) if not math.isnan(float(score[s])) else -math.inf, s) for s in entry.index[entry.to_numpy(bool)] if s not in st_pos]
    cands.sort(key=lambda t: -t[0])
    alloc_base = acct["equity"] / config.max_positions
    for sc, sym in cands[: max(free, 0)]:
        px = float(close[sym].iloc[-1])
        alloc = alloc_base
        if config.max_adv_pct is not None and adv is not None and not math.isnan(float(adv[sym])):
            alloc = min(alloc, config.max_adv_pct * float(adv[sym]))
        limit = round(px * (1 + limit_buffer), 2)
        qty = int(alloc // (limit * (1 + config.commission_pct)))
        if qty < 1:
            continue
        sd = float(stop_dist[sym]) if stop_dist is not None and not math.isnan(float(stop_dist[sym])) else None
        plan.entries.append(Order(sym, qty, "buy", limit, exchange_of.get(sym, "NASD"), f"entry score={sc:.2f}"))
        state.setdefault("pending_entries", {})[sym] = {"qty": qty, "ref_price": px, "stop_dist": sd, "signal_date": plan.as_of}
    return plan


def apply_plan(plan: Plan, broker: Broker, state: dict, stocks: dict, dry_run: bool) -> None:
    """주문 제출 + 상태 갱신. dryrun 브로커는 다음날 시가 대신 오늘 종가로 체결됐다고 가정한다."""
    close = stocks["close"]
    for o in plan.exits + plan.entries:
        if dry_run:
            log.info("[PLAN] %s %d %s @ %.2f (%s)", o.side.upper(), o.qty, o.symbol, o.limit_price, o.reason)
        else:
            broker.submit(o)
    st_pos = state["positions"]
    if broker.name == "dryrun":
        for o in plan.exits:
            p = st_pos.pop(o.symbol, None)
            if p:
                state["cash"] = state.get("cash", 0.0) + p["qty"] * float(close[o.symbol].iloc[-1])
        for o in plan.entries:
            pe = state.get("pending_entries", {}).pop(o.symbol, None)
            px = float(close[o.symbol].iloc[-1])
            state["cash"] = state.get("cash", 0.0) - o.qty * px
            st_pos[o.symbol] = {"qty": o.qty, "entry_date": plan.as_of, "entry_price": px,
                                "stop": (px - pe["stop_dist"]) if pe and pe.get("stop_dist") else None, "hold_days": 0}
        mv = sum(p["qty"] * float(close[s].iloc[-1]) for s, p in st_pos.items() if s in close.columns)
        state["equity"] = state.get("cash", 0.0) + mv
    else:
        for o in plan.exits:
            st_pos.pop(o.symbol, None)
        for o in plan.entries:
            pe = state.get("pending_entries", {}).get(o.symbol)
            st_pos[o.symbol] = {"qty": o.qty, "entry_date": plan.as_of, "entry_price": None,
                                "stop": (pe["ref_price"] - pe["stop_dist"]) if pe and pe.get("stop_dist") else None, "hold_days": 0}
    state["pending_entries"] = {}
    state["last_run"] = str(date.today())
