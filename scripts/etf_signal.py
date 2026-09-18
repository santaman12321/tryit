"""ETF 레그 오늘의 목표 비중: QQQ 20일 실현변동성 기반 변동성 타게팅(목표 30%) + 200일선 필터, TQQQ 로 실행.

  python scripts/etf_signal.py            -> 목표 TQQQ 비중(%) 출력 (주 1회 금요일 종가 기준으로 조정 권장)
"""
from __future__ import annotations

import argparse

import numpy as np
import yfinance as yf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.30, help="목표 연변동성 (0.20 보수적 / 0.30 공격적)")
    args = ap.parse_args()
    q = yf.download("QQQ", period="2y", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    rv = float(q.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
    sma200 = float(q.rolling(200).mean().iloc[-1])
    last = float(q.iloc[-1])
    expo = min(3.0, args.target / rv) if last > sma200 else 0.0
    print(f"QQQ 종가 {last:.2f} / 200일선 {sma200:.2f} -> {'추세 ON' if last > sma200 else '추세 OFF(현금)'}")
    print(f"20일 실현변동성 {rv * 100:.1f}% -> 목표 QQQ 노출 {expo * 100:.0f}% = TQQQ 비중 {expo / 3 * 100:.0f}% (나머지 현금/단기채)")


if __name__ == "__main__":
    main()
