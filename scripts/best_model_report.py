"""'최대한 돈을 버는 모델' 후보 리더보드: 종목 전략 격자(연도별) + ETF 장기 결과를 모아 정리한다.

선택 기준(과최적화 방지): 각 전략의 격자 중 '2024·2025·2026 세 해 모두 수익 > 0' 인 설정만 후보로 두고, 그중 전체 기간(2023-11~2026-09) 수익률이 가장 높은 설정을 대표로 삼는다.
  python scripts/best_model_report.py --out reports/best_model
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

WINDOWS = ["2024", "2025", "2026", "full"]


def pick(df: pd.DataFrame):
    keys = [c for c in df.columns if not any(c.startswith(w + "_") for w in WINDOWS)]
    ok = df[(df["2024_total_return"] > 0) & (df["2025_total_return"] > 0) & (df["2026_total_return"] > 0)]
    cand = ok if not ok.empty else df
    best = cand.sort_values("full_total_return", ascending=False).iloc[0]
    return best, keys, len(ok), len(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/best_model")
    args = ap.parse_args()
    out = Path(args.out)
    lines = ["# 수익 극대화 모델 후보 리더보드 (국내 증권사 비용: 수수료 편도 0.25% + 슬리피지)\n",
             "## 1. 종목 전략 (미국 전 종목 일봉, 최대 10~20종목 동일비중, 포지션 ≤ 20일 평균거래대금 1%)\n",
             "각 전략의 파라미터 격자 중 **2024·2025·2026 모두 수익 > 0** 인 설정만 후보로 두고, 전체 기간 수익률이 가장 높은 설정을 대표로 뽑았다(연도별 일관성 우선).\n",
             "| 전략 | 세 해 모두 + / 격자 | 대표 설정 | 2024 | 2025 | 2026 YTD | 전체(2023-11~) | 전체 MDD | 전체 승률 | 거래 |", "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for f in sorted(glob.glob(str(out / "grid_*.csv"))):
        df = pd.read_csv(f)
        if "full_total_return" not in df:
            continue
        name = Path(f).stem.replace("grid_", "")
        best, keys, n_ok, n_all = pick(df)
        setting = ", ".join(f"{k}={best[k]}" for k in keys)
        lines.append(f"| {name} | {n_ok} / {n_all} | `{setting}` | {best['2024_total_return'] * 100:+.1f}% | {best['2025_total_return'] * 100:+.1f}% | "
                     f"{best['2026_total_return'] * 100:+.1f}% | {best['full_total_return'] * 100:+.1f}% | {best['full_max_drawdown'] * 100:.1f}% | {best['full_win_rate'] * 100:.1f}% | {int(best['full_n_trades'])} |")
        rows.append({"strategy": name, "setting": setting, **{c: best[c] for c in df.columns if c.startswith(tuple(w + "_" for w in WINDOWS))}, "n_ok": n_ok, "n_all": n_all})
    pd.DataFrame(rows).to_csv(out / "leaderboard_stocks.csv", index=False)
    etf = out / "etf_longrun.md"
    if etf.exists():
        lines.append("\n## 2. ETF 장기 검증 (2005~2026)\n")
        lines.append(etf.read_text(encoding="utf-8").split("\n", 2)[2])
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
