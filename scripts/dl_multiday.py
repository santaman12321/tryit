"""저장된 예측(pred_<decision>.parquet)으로 다일 보유 성과를 점검한다: 유니버스 베이스라인, 십분위, 반기, 시총 구간, 이상치 의존도, 중첩 보정 누적.

  python scripts/dl_multiday.py --decision open --out reports/dl_daytrading
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def top(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df.sort_values(["date", "p_ens"], ascending=[True, False]).groupby("date").head(k)


def line(t: pd.DataFrame, col: str, H: int) -> str:
    t = t.dropna(subset=[col])
    net = t[col] - t["cost"]
    daily = net.groupby(t["date"]).mean()
    cum = float((1 + daily / H).prod() - 1)
    pnl = net.sort_values(ascending=False)
    top10_share = float(pnl.head(10).sum() / pnl.sum()) if pnl.sum() > 0 else float("nan")
    wo_top = float(net[net < pnl.head(int(len(net) * 0.02)).min()].mean()) if len(net) > 50 else float("nan")
    return (f"{len(t)} | {(net > 0).mean() * 100:.1f}% | {t[col].mean() * 100:+.3f}% | {net.mean() * 100:+.3f}% | {net.median() * 100:+.3f}% | "
            f"{cum * 100:+.1f}% | {top10_share * 100:.0f}% | {wo_top * 100:+.3f}%")


HEAD = "| 구분 | 거래 | 승률 | 총평균 | 비용후 평균 | 중앙값 | 누적(중첩 보정) | 상위10건 손익 비중 | 상위 2% 제외 평균 |\n|---|---|---|---|---|---|---|---|---|"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", default="open")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="reports/dl_daytrading")
    args = ap.parse_args()
    out = Path(args.out)
    pr = pd.read_parquet(out / f"pred_{args.decision}.parquet")
    pr["cost"] = 2 * (np.where(pr["dec_px"] < 5.0, 0.005, 0.001) + 0.0025)
    te = pr[pr.split == "test"].copy()
    va = pr[pr.split == "val"].copy()
    md = [f"# 같은 예측으로 다일 보유 — 점검 (결정 {args.decision}, 앙상블 상위 {args.k})\n",
          "시간봉 종가 기준 라벨(배당 미조정 기준 통일). 비용후 = 왕복 0.7%(≥$5)/1.5%(동전주). 누적(중첩 보정) = 일 수익률/보유일수 로 근사. "
          "상위10건 손익 비중 = 상위 10건 순손익 / 전체 순손익(100% 초과면 나머지 거래 합이 손실).\n"]
    for col, H, label in (("y_rod", 1, "당일 종가"), ("y_nd", 1, "1일 보유"), ("y_3d", 3, "3세션 보유"), ("y_5d", 5, "5세션 보유")):
        md.append(f"## {label} ({col})\n\n{HEAD}")
        md.append(f"| 테스트 전체 | {line(top(te, args.k), col, H)} |")
        md.append(f"| 테스트 전반(~2026-03-14) | {line(top(te[te.date < '2026-03-15'], args.k), col, H)} |")
        md.append(f"| 테스트 후반(2026-03-15~) | {line(top(te[te.date >= '2026-03-15'], args.k), col, H)} |")
        md.append(f"| 검증(2025-07~09) | {line(top(va, args.k), col, H)} |")
        mc = np.exp(te["log_mcap"])
        for lab, lo, hi in (("<$300M", 0, 3e8), ("$300M~$2B", 3e8, 2e9), ("$2B~$10B", 2e9, 1e10), (">$10B", 1e10, 1e15)):
            sub = te[(mc >= lo) & (mc < hi)]
            if len(sub) > 2000:
                md.append(f"| 테스트 시총 {lab} | {line(top(sub, args.k), col, H)} |")
        base = te[col].mean()
        md.append(f"\n유니버스 평균(테스트, 베이스라인): {base * 100:+.3f}% · 십분위별 평균: " + ", ".join(
            f"{d}:{v * 100:+.2f}%" for d, v in te.assign(dec=te.groupby('date')['p_ens'].rank(pct=True).mul(10).clip(upper=9.999).astype(int)).groupby('dec')[col].mean().items()) + "\n")
    (out / f"multiday_{args.decision}.md").write_text("\n".join(md), encoding="utf-8")
    print("->", out / f"multiday_{args.decision}.md")


if __name__ == "__main__":
    main()
