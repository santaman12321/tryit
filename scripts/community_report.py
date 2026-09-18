"""우리 모델 vs 커뮤니티/GitHub/유튜브 모델 비교표 생성 → reports/community_compare/README.md"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

OUT = Path("reports/community_compare")
WINDOWS = ["2024", "2025", "2026", "full"]


def tech_rows():
    rows = []
    for f in sorted(glob.glob(str(OUT / "grid_*.csv"))):
        df = pd.read_csv(f)
        if "full_total_return" not in df:
            continue
        name = Path(f).stem.replace("grid_", "")
        keys = [c for c in df.columns if not any(c.startswith(w + "_") for w in WINDOWS)]
        best = df.sort_values("full_total_return", ascending=False).iloc[0]
        ok = int(((df["2024_total_return"] > 0) & (df["2025_total_return"] > 0) & (df["2026_total_return"] > 0)).sum())
        rows.append(f"| 기술적 봇: {name} | S&P 1500, `{', '.join(f'{k}={best[k]}' for k in keys)}` | {best['2024_total_return'] * 100:+.1f}% | {best['2025_total_return'] * 100:+.1f}% | "
                    f"{best['2026_total_return'] * 100:+.1f}% | {best['full_total_return'] * 100:+.1f}% | {best['full_max_drawdown'] * 100:.1f}% | {best['full_win_rate'] * 100:.0f}% | {ok}/{len(df)} 세 해 + |")
    return rows


def main() -> None:
    fin = pd.read_csv("reports/best_model/finalists.csv")
    etf = pd.read_csv(OUT / "etf_models.csv")
    lines = ["# 우리 모델 vs 공개 자동매매 모델 비교 (같은 비용: 수수료 편도 0.25% + 슬리피지)\n",
             "출처·규칙: `docs/community_models.md`. 공개된 '주장 수치'는 인샘플·비용 전인 경우가 많아 별도로 적었다.\n",
             "## 1. 2024-01~2026-09 (최근 2.7년, 우리 종목 모델과 같은 구간)\n",
             "| 모델 | 설명 | 2024 | 2025 | 2026 YTD | 총수익 | MDD | 승률 | 비고 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in fin.itertuples():
        lines.append(f"| **우리: {r.name}** | 이 저장소 | {getattr(r, '_6', float('nan')) * 100 if False else fin.loc[r.Index, '2024'] * 100:+.1f}% | {fin.loc[r.Index, '2025'] * 100:+.1f}% | {fin.loc[r.Index, '2026'] * 100:+.1f}% | "
                     f"{r.total * 100:+.1f}% | {r.mdd * 100:.1f}% | {'-' if pd.isna(r.win_rate) else f'{r.win_rate * 100:.0f}%'} | |")
    # ETF 커뮤니티 모델: 2024~ 구간 수익과 연도별은 etf_models.md 의 연도표에서
    md_etf = (OUT / "etf_models.md").read_text(encoding="utf-8")
    yr_tbl = md_etf.split("## 연도별")[1].strip().split("\n")
    yr_head = [h.strip() for h in yr_tbl[0].strip("|").split("|")]
    yr_map = {}
    for ln in yr_tbl[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        yr_map[cells[0]] = dict(zip(yr_head[1:], cells[1:]))
    for r in etf.itertuples():
        y = yr_map.get(r.모델, {})
        lines.append(f"| 커뮤니티: {r.모델} | ETF | {y.get('2024', '-')} | {y.get('2025', '-')} | {y.get('2026', '-')} | {getattr(r, '_7') * 100:+.1f}% | (장기 MDD {r.MDD * 100:.0f}%) | - | 2024~ 누적 |")
    lines += tech_rows()
    lines += ["", "## 2. 장기 (가능한 최장 기간, ETF 모델)\n", md_etf.split("\n", 2)[2].split("## 연도별")[0],
              "## 3. 공개된 성과 주장 vs 재현\n",
              "| 모델 | 공개 주장 | 우리 재현(같은 비용) | 차이의 이유 |", "|---|---|---|---|",
              "| TQQQ FTLT (Composer) | 2021-09~ 연 160%, 샤프 2.09 | 위 표 | Composer 표시치는 규칙을 그 기간에 맞춰 만든 인샘플이며, 공유본 트리마다 세부가 다름 |",
              "| 9Sig | 2010~2026 CAGR 39% (BestFolio 재현) | 위 표 | 우리는 신규 납입 없는 폐쇄형, 비용 포함 |",
              "| FinRL DRL 앙상블 | 연 25.9~52.6%, 샤프 1.5~2.8 | 재현 안 함 | 비용 전·특정 기간(2016~2020), FinRL-Meta 재현에서 절반으로 감소 |",
              "| Qlib LightGBM Alpha158 | IC 0.04~0.045, 연 초과 9~13% (중국 CSI300) | 우리 ML 랭커 IC 0.05 수준, 연 15~26% | 시장·비용 다름, 방향은 일치 |",
              "| freqtrade 공개 전략 | 사이트별 수백 % 표시 | 재현 안 함(암호화폐) | freqtrade 문서 자체가 '공개 전략 백테스트 믿지 말 것' 경고 |"]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
