"""장중(1시간봉/15분봉) 이벤트 스터디용 유틸.

Yahoo 봉은 09:30 부터 시작(1h: 09:30,10:30,...,15:30 = 7개 / 15m: 26개). 거래가 없는 슬롯은 종가로 채운다(고가=저가=종가).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOT_MINUTES = {"1h": 60, "15m": 15, "5m": 5, "30m": 30}


def slots_for(interval: str) -> list[str]:
    step = SLOT_MINUTES[interval]
    n = int(390 / step) if 390 % step == 0 else int(390 / step) + 1
    times = []
    for k in range(n):
        m = 9 * 60 + 30 + k * step
        times.append(f"{m // 60:02d}:{m % 60:02d}")
    return times


def build_day_arrays(bars: pd.DataFrame, interval: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """long 봉 데이터 -> (키 테이블[symbol, date], {field: (n_days x n_slots)}).

    누락 슬롯은 직전 종가로 채움. 첫 슬롯(09:30)이 없는 날은 제외.
    """
    slots = slots_for(interval)
    bars = bars.copy()
    bars["date"] = bars["datetime"].dt.normalize()
    bars["slot"] = bars["datetime"].dt.strftime("%H:%M")
    bars = bars[bars["slot"].isin(slots)]
    slot_idx = {s: i for i, s in enumerate(slots)}
    bars["k"] = bars["slot"].map(slot_idx).astype(int)
    key = bars[["symbol", "date"]].drop_duplicates().reset_index(drop=True)
    key["row"] = np.arange(len(key))
    bars = bars.merge(key, on=["symbol", "date"], how="left")
    n, m = len(key), len(slots)
    arr = {f: np.full((n, m), np.nan) for f in ("open", "high", "low", "close", "volume")}
    for f in arr:
        arr[f][bars["row"].to_numpy(), bars["k"].to_numpy()] = bars[f].to_numpy(float)
    # 첫 슬롯 없는 날 제외
    ok = ~np.isnan(arr["close"][:, 0])
    key = key[ok].reset_index(drop=True)
    for f in arr:
        arr[f] = arr[f][ok]
    # 누락 슬롯 채우기: close ffill, open/high/low = 그 close, volume 0
    c = arr["close"]
    for k in range(1, m):
        miss = np.isnan(c[:, k])
        c[miss, k] = c[miss, k - 1]
        for f in ("open", "high", "low"):
            arr[f][miss, k] = c[miss, k]
        arr["volume"][miss, k] = 0.0
    key["row"] = np.arange(len(key))
    return key, arr


def lookup_rows(key: pd.DataFrame, symbols: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """(symbol, date) -> row index (없으면 -1)."""
    idx = pd.MultiIndex.from_frame(key[["symbol", "date"]])
    q = pd.MultiIndex.from_arrays([symbols, pd.to_datetime(dates)])
    pos = idx.get_indexer(q)
    return pos


def simulate_exits(o, h, l, c, entry_px, entry_k, exit_k, stop_pct=None, target_pct=None, trail_pct=None):
    """봉 단위 경로 시뮬레이션 (벡터). entry_k 봉 '종가' 에 진입했다고 보고 entry_k+1 봉부터 손절/익절/트레일 판정, exit_k 봉 종가 청산.

    같은 봉에서 손절·익절 모두 걸리면 손절 우선(보수적). 반환: (exit_price, exit_reason_code)  0=time 1=stop 2=target 3=trail
    """
    n, m = c.shape
    ex_px = c[np.arange(n), exit_k].copy()
    reason = np.zeros(n, dtype=int)
    done = np.zeros(n, dtype=bool)
    hi_run = entry_px.copy()
    for k in range(m):
        active = (~done) & (k > entry_k) & (k <= exit_k)
        if not active.any():
            continue
        lo_k, hi_k = l[:, k], h[:, k]
        if stop_pct is not None:
            stop_lvl = entry_px * (1 - stop_pct)
            hit = active & (lo_k <= stop_lvl)
            ex_px[hit] = np.minimum(o[hit, k], stop_lvl[hit])
            reason[hit] = 1
            done |= hit
            active &= ~hit
        if trail_pct is not None:
            trail_lvl = hi_run * (1 - trail_pct)
            hit = active & (lo_k <= trail_lvl)
            ex_px[hit] = np.minimum(o[hit, k], trail_lvl[hit])
            reason[hit] = 3
            done |= hit
            active &= ~hit
        if target_pct is not None:
            tgt = entry_px * (1 + target_pct)
            hit = active & (hi_k >= tgt)
            ex_px[hit] = np.maximum(o[hit, k], tgt[hit])
            reason[hit] = 2
            done |= hit
            active &= ~hit
        hi_run = np.where(active, np.maximum(hi_run, hi_k), hi_run)
    return ex_px, reason
