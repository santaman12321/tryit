"""기술적 지표. Series / wide DataFrame 모두 지원 (열 단위 벡터 연산).

모든 지표는 시점 t 의 값이 t 까지의 데이터만 사용하도록 작성되어 있다(look-ahead 없음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x, n: int):
    return x.rolling(n, min_periods=n).mean()


def ema(x, n: int):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close, n: int = 14):
    """Wilder RSI. 값이 없으면 NaN."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_up = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # 하락이 전혀 없으면 RSI=100, 상승/하락 모두 0 이면 50
    out = out.where(avg_down != 0.0, 100.0)
    out = out.where(~((avg_down == 0.0) & (avg_up == 0.0)), 50.0)
    out = out.where(avg_up.notna() & avg_down.notna())
    return out


def true_range(high, low, close):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return np.fmax(np.fmax(tr1, tr2), tr3)


def atr(high, low, close, n: int = 14):
    """Wilder ATR."""
    return true_range(high, low, close).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rolling_max(x, n: int):
    return x.rolling(n, min_periods=n).max()


def rolling_min(x, n: int):
    return x.rolling(n, min_periods=n).min()


def pct_change(x, n: int):
    return x / x.shift(n) - 1.0


def dollar_volume(close, volume, n: int = 20):
    return (close * volume).rolling(n, min_periods=n).mean()
