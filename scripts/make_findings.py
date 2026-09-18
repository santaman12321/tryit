"""검증 결과를 한 문서로 모은다 -> reports/surge_study/FINDINGS.md

입력: reports/surge_study/overview.csv, events/*.csv.gz, grid_*.csv, intraday_*/combos.csv, reports/sp1500_*/summary.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R = Path("reports/surge_study")
TRAIN, TEST = "2023-11-01~2025-09-17", "2025-09-18~2026-09-17"


def pct(x, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x * 100:+.2f}%" if sign else f"{x * 100:.1f}%"


def sec_overview() -> str:
    ov = pd.read_csv(R / "overview.csv")
    piv = ov.pivot(index="screen", columns="window", values=["n", "key_win", "key_mean", "d3_win", "d3_mean"])
    order = [s for s in ov["screen"].drop_duplicates()]
    lines = ["| 공식(스크린) | 학습 n | 학습 단타 승률/평균 | 학습 3일 승률/평균 | 검증 n | 검증 단타 승률/평균 | 검증 3일 승률/평균 |", "|---|---|---|---|---|---|---|"]
    for s in order:
        r = piv.loc[s]
        lines.append(f"| {s} | {int(r[('n', 'train')])} | {pct(r[('key_win', 'train')], False)} / {pct(r[('key_mean', 'train')])} | "
                     f"{pct(r[('d3_win', 'train')], False)} / {pct(r[('d3_mean', 'train')])} | {int(r[('n', 'test')])} | "
                     f"{pct(r[('key_win', 'test')], False)} / {pct(r[('key_mean', 'test')])} | {pct(r[('d3_win', 'test')], False)} / {pct(r[('d3_mean', 'test')])} |")
    both_pos = int(((piv[("key_mean", "train")] > 0) & (piv[("key_mean", "test")] > 0)).sum())
    both_pos3 = int(((piv[("d3_mean", "train")] > 0) & (piv[("d3_mean", "test")] > 0)).sum())
    return "\n".join(lines) + f"\n\n두 기간 모두 비용후 평균 > 0: 단타 {both_pos}개 / 3일 보유 {both_pos3}개 (전체 {len(order)}개 공식). 최고 승률(검증): {piv[('key_win', 'test')].max() * 100:.1f}%.\n"


def sec_buckets() -> str:
    tr = pd.read_csv(R / "events/surge_base_train.csv.gz")
    te = pd.read_csv(R / "events/surge_base_test.csv.gz")
    for d in (tr, te):
        for c in ["ret_oc", "ret_3d", "ret_5d", "ret_10d"]:
            d["net_" + c[4:]] = d[c] - d["cost"]
    feats = {
        "ret1": ([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 100], "당일 상승률"), "vol_ratio": ([2, 3, 5, 10, 20, 50, 1e9], "거래량 배수"),
        "close_pos": ([0, 0.4, 0.7, 0.9, 1.0], "종가 위치"), "price": ([0, 1, 2, 5, 10, 20, 1e9], "주가($)"), "mcap": ([0, 5e7, 3e8, 2e9], "근사 시총"),
        "shares": ([0, 1e7, 2e7, 5e7, 1e8, 1e12], "발행주식수"), "turnover": ([0, 0.05, 0.2, 0.5, 1.0, 100], "회전율"),
        "consec_up": ([0, 1, 2, 3, 100], "연속 상승일"), "runup30": ([-1, 0.2, 0.5, 1.0, 2.0, 100], "30일 상승폭"),
        "days_from_high52": ([-1, -0.5, -0.2, -0.05, 100], "52주 고가 대비"), "gap": ([-1, 0, 0.05, 0.1, 0.2, 0.5, 100], "다음날 갭"),
        "spy_above200": ([-0.5, 0.5, 1.5], "SPY>200일선"), "iwm_above50": ([-0.5, 0.5, 1.5], "IWM>50일선"),
        "earnings": ([-0.5, 0.5, 1.5], "실적발표 재료"), "china": ([-0.5, 0.5, 1.5], "중국/홍콩 기업"), "weekday": ([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], "요일(0=월)"),
    }
    lines = ["| 특징 | 구간 | 학습 n | 학습 단타 승률/평균 | 학습 3일 평균 | 검증 n | 검증 단타 승률/평균 | 검증 3일 평균 | 두 기간 모두 + ? |", "|---|---|---|---|---|---|---|---|---|"]
    n_pos = 0
    n_all = 0
    for f, (bins, label) in feats.items():
        if f not in tr or f not in te:
            continue
        ct = pd.cut(tr[f], bins, include_lowest=True)
        cv = pd.cut(te[f], bins, include_lowest=True)
        for b in ct.cat.categories:
            a, z = tr[ct == b], te[cv == b]
            if len(a) < 100 or len(z) < 100:
                continue
            both = (a.net_oc.mean() > 0 and z.net_oc.mean() > 0) or (a.net_3d.mean() > 0 and z.net_3d.mean() > 0)
            n_all += 1
            n_pos += int(both)
            lines.append(f"| {label} | {b} | {len(a)} | {pct((a.net_oc > 0).mean(), False)} / {pct(a.net_oc.mean())} | {pct(a.net_3d.mean())} | "
                         f"{len(z)} | {pct((z.net_oc > 0).mean(), False)} / {pct(z.net_oc.mean())} | {pct(z.net_3d.mean())} | {'**예**' if both else '아니오'} |")
    return "\n".join(lines) + f"\n\n두 기간 모두 양(+)인 구간(단타 또는 3일): {n_pos} / {n_all}.\n"


def sec_penny() -> str:
    out = []
    for w in ("train", "test"):
        d = pd.read_csv(R / f"events/surge_base_{w}.csv.gz")
        s = d[d.price <= 1.0].copy()
        s["net_10d"] = s.ret_10d - s.cost
        top = s.nlargest(3, "ret_10d")
        trimmed = s[s.ret_10d < s.ret_10d.quantile(0.98)].net_10d.mean()
        out.append(f"- {w}: n={len(s)}, 10일 보유 평균 {pct(s.net_10d.mean())}, 중앙값 {pct(s.net_10d.median())}, 상위 2% 제외 평균 {pct(trimmed)}, "
                   f"승률 {pct((s.net_10d > 0).mean(), False)}, 최대 3건: " + ", ".join(f"{r.symbol} {r.ret_10d * 100:+.0f}%" for r in top.itertuples()))
    return "\n".join(out)


def sec_grids() -> str:
    lines = ["| 스크린 | 조합 수 | 두 기간 모두 수익 + | 검증 최고 승률 | 검증 최고 수익률(조합) | 그 조합의 학습 수익률 |", "|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(str(R / "grid_*.csv"))):
        df = pd.read_csv(f)
        name = Path(f).stem.replace("grid_", "")
        if "test_total_return" not in df:
            continue
        both = int(((df.train_total_return > 0) & (df.test_total_return > 0)).sum())
        best = df.loc[df.test_total_return.idxmax()]
        keys = [c for c in df.columns if not c.startswith(("train_", "test_"))]
        combo = ", ".join(f"{k}={best[k]}" for k in keys)
        lines.append(f"| {name} | {len(df)} | {both} | {df.test_win_rate.max() * 100:.1f}% | {best.test_total_return * 100:+.1f}% ({combo}) | {best.train_total_return * 100:+.1f}% |")
    return "\n".join(lines)


def sec_intraday() -> str:
    out = []
    for iv, label in [("1h", "1시간봉 (2023-10~2026-09, 학습/검증 = 2025-09-18 기준)"), ("15m", "15분봉 (2026-06-24~09-17, 전반/후반 = 2026-08-06 기준)")]:
        f = R / f"intraday_{iv}/combos.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        piv = df.pivot_table(index=["entry", "exit", "risk"], columns="window", values=["n", "win", "mean", "pf"])
        both = piv[(piv[("n", "train")] >= 100) & (piv[("n", "test")] >= 100)]
        n_pos = int(((both[("mean", "train")] > 0) & (both[("mean", "test")] > 0)).sum())
        maxwin = both[[("win", "train"), ("win", "test")]].min(axis=1).max()
        out.append(f"- **{label}**: 조합 {len(both)}개 중 두 기간 모두 비용후 평균 > 0 인 조합 **{n_pos}개**, '두 기간 중 낮은 쪽 승률' 최고 {maxwin * 100:.1f}%.")
    return "\n".join(out)


def sec_tournament() -> str:
    f = R / "tournament/leaderboard.csv"
    if not f.exists():
        return "(토너먼트 결과 없음)"
    lb = pd.read_csv(f)
    runs = pd.read_csv(R / "tournament/runs_all.csv") if (R / "tournament/runs_all.csv").exists() else None
    lines = ["| 순위 | 모델 | 재료제외 | 학습 거래/승률/기대/수익률/MDD | 검증 거래/승률/기대/수익률/MDD | 둘 다 + | 검증 최고(사후) 기대 |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(lb.itertuples(), 1):
        lines.append(f"| {i} | {r.model} | {'예' if r.exclude_earnings else '-'} | "
                     f"{int(r.train_n_trades)} / {r.train_win_rate * 100:.1f}% / {r.train_expectancy * 100:+.2f}% / {r.train_total_return * 100:+.1f}% / {r.train_max_drawdown * 100:.1f}% | "
                     f"{int(r.test_n_trades)} / {r.test_win_rate * 100:.1f}% / {r.test_expectancy * 100:+.2f}% / {r.test_total_return * 100:+.1f}% / {r.test_max_drawdown * 100:.1f}% | "
                     f"{'**예**' if r.both_positive else '아니오'} | {r.oracle_test_expectancy * 100:+.2f}% |")
    n_both = int(lb.both_positive.sum())
    extra = f"\n\n{len(lb)}개 모델 · {len(runs) if runs is not None else '?'}개 설정. 학습에서 고른 설정이 검증에서도 기대수익 > 0: **{n_both}개**. "
    extra += f"검증 승률 최고(선택 설정): {lb.test_win_rate.max() * 100:.1f}%. 상세(파라미터·청산 규칙): `tournament/leaderboard.md`.\n"
    if runs is not None:
        both_any = runs[(runs.train_n_trades >= 60) & (runs.test_n_trades >= 60) & (runs.train_expectancy > 0) & (runs.test_expectancy > 0)]
        extra += f"\n모든 설정 {len(runs)}개 중 두 기간 모두(거래 ≥ 60) 기대수익 > 0 인 설정: {len(both_any)}개"
        if len(both_any):
            top = both_any.sort_values("test_expectancy", ascending=False).head(5)
            extra += " — 예: " + "; ".join(f"{r.model} {r.params} {r.exit} (학습 {r.train_expectancy * 100:+.2f}%/검증 {r.test_expectancy * 100:+.2f}%, 승률 {r.train_win_rate * 100:.0f}/{r.test_win_rate * 100:.0f}%)" for r in top.itertuples())
        extra += ".\n"
    return "\n".join(lines) + extra


def sec_sp1500() -> str:
    lines = ["| 전략 | 기간 | 거래 | 승률 | 수익률 | MDD | PF | SPY |", "|---|---|---|---|---|---|---|---|"]
    for d in sorted(glob.glob("reports/sp1500_*")):
        f = Path(d) / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        for name, v in s.items():
            lines.append(f"| {name} | {v['start']}~{v['end']} | {v['n_trades']} | {v['win_rate'] * 100:.1f}% | {v['total_return'] * 100:+.1f}% | {v['max_drawdown'] * 100:.1f}% | {v['profit_factor']:.2f} | {v.get('bench_total_return', float('nan')) * 100:+.1f}% |")
    return "\n".join(lines)


def main() -> None:
    md = f"""# 급등주(동전주·소형주) 공식 검증 결과 — 한국 투자자 기준

기준: 국내 증권사 해외주식 계좌(공매도 불가, 매매수수료 편도 0.25%), 슬리피지 편도 10bp(주가 ≥ $5)/50bp(동전주), 롱(매수)만.
데이터: 미국 상장 보통주 4,987종목 일봉(2023-06~2026-09-17) + 급등 이력 종목 1,995개의 1시간봉(2023-10~) + 2,104종목 15분봉(최근 60일).
학습 구간 {TRAIN} 에서 규칙·값을 고르고 검증 구간 {TEST} (최근 1년) 에서 재확인했다.
공식 출처와 정량화 규칙: `docs/surge_formulas.md`. 상세 표: `README.md`, `intraday_1h/`, `intraday_15m/`, `grid_*.md`.

## 1. 결론

1. **블로그·유튜브·커뮤니티의 급등주 공식을 그대로 '다음날 매수' 로 쓰면, 어떤 것도 비용 차감 후 두 기간 모두 기대값이 양(+)이 아니다.** 승률은 30~45% 대에 머문다.
2. 상승률·거래량 배수·주가·시총·발행주식수·회전율·마감 위치·연속 상승·52주 고가 대비·시장 레짐·실적 재료 유무·중국 기업 여부·요일로 구간을 쪼개도, **두 기간 모두 양(+)인 구간은 사실상 없다**(아래 표). 1달러 미만 구간의 큰 '평균' 은 상위 2% 로또 종목이 만든 착시다.
3. 장중(1시간봉·15분봉)에서 시가·첫봉 확인·ORB·첫 눌림·VWAP 진입 × 시간/손절/익절/트레일 청산 조합 150개씩을 돌려도 **두 기간 모두 양(+)인 조합은 0개**. 급등 다음날 고가의 절반 이상이 개장 후 첫 봉에서 찍히고, 시가보다 낮게 마감하는 날이 과반이다.
4. '승률 높은 거래만 골라서 손절을 잘 하면 된다' 는 접근은 데이터와 맞지 않았다: **손절을 타이트하게 할수록 승률이 15~30% 로 떨어진다**(급등주는 하루 변동폭이 커서 손절이 먼저 걸린다). 손절을 넓히면 승률은 40% 대로 오르지만 한 번의 손실이 커서 총수익은 여전히 음수다.
5. 24개 모델을 파라미터 탐색 후 경쟁시킨 토너먼트(2절) 결과가 최종 판정이다: 학습 구간에서 고른 최적 설정을 검증 구간에 적용했을 때 어떤 모델이 살아남는지 표에 그대로 적었다. 토스 커뮤니티 글의 '첫 거래량 패스, 두 번째 거래량/눌림 진입' 4종도 같은 조건으로 경쟁했다.
6. 비교 기준으로 S&P 1500 유동주 대상 추세추종(돌파)은 올해 +81% 였지만 작년 +3%, 승률 32~40% 로 '고승률' 과 거리가 멀고 올해의 반도체 테마 장세에 의존한다.

**실현 가능성 판단**: 국내 증권사 계좌(수수료 0.25%×2)로 미국 동전주·소형주 급등주를 추격·눌림·오버나잇으로 사는 자동매매는, 이 데이터가 커버하는 3년 동안 어떤 파라미터로도 양(+)의 기대값을 만들지 못했다. 자동화할 대상이 아니라는 것이 검증 결과다.

## 2. 모델 토너먼트 — 모든 모델을 같은 조건에서 파라미터 탐색 후 경쟁

각 모델(공식 15종 + 커뮤니티 글 4종 + 전략 클래스 5종)의 파라미터 격자를 **학습 구간에서만** 돌려 기대수익/거래(모든 비용 차감)가 가장 좋은 설정을 고르고,
그 설정을 **검증 구간(최근 1년)** 에서 평가했다. 포트폴리오: 최대 10종목 동일비중, 포지션 ≤ 20일 평균 거래대금 1%, 수수료 편도 0.25%, 슬리피지.

{sec_tournament()}

## 3. 공식 15종 + 커뮤니티 4종 이벤트 스터디 (비용후, 기본 파라미터)

단타 = 다음날 시가 매수·당일 종가 매도(first_green_day 는 종가 매수·다음날 시가 매도, gap_and_go 는 당일 시가 매수·종가 매도), 3일 = 다음날 시가 매수·3일째 종가 매도.

{sec_overview()}

## 4. 적절값 탐색 — 급등일(전일 +5%↑, 거래량 2배↑, 거래대금 $1M↑, 시총 $2B↓) 이벤트를 특징 구간별로

{sec_buckets()}

### 1달러 미만 동전주 구간의 '평균' 이 왜 착시인가

{sec_penny()}

(현재 상장 종목만 있는 유니버스라, 급등 뒤 상장폐지된 동전주는 빠져 있다 → 실제는 이보다 나쁘다.)

## 5. 포트폴리오 격자 (최대 10종목 동일비중, 수수료·슬리피지·유동성 제한 포함)

각 스크린을 손절(5/10/20%/없음) × 익절(없음/10~20/30~50%) × 보유일(1~10) 격자로 두 기간에 돌린 결과.

{sec_grids()}

## 6. 장중 스터디 (진입 5~7종 × 청산 3종 × 리스크 10종)

{sec_intraday()}

봉별 고가/저가 분포와 시간대별 수익률은 `intraday_1h/README.md`, `intraday_15m/README.md` 참고.

## 7. 비교: S&P 1500 유동주 전략 (같은 비용)

{sec_sp1500()}

## 8. 한계와 다음 단계

- 생존 편향(상장폐지 종목 누락), 프리마켓·뉴스·공시 데이터 부재, float 대신 발행주식수, 15분봉은 60일뿐.
- 1분·15분봉 1년치가 필요하면 국내 증권사 API(한국투자증권 해외주식 분봉)로 매일 축적하거나 유료 데이터(Polygon 등)를 붙여야 한다. `scripts/fetch_intraday.py` 와 `scripts/intraday_study.py` 는 봉 간격만 바꾸면 그대로 쓸 수 있다.
- 자동매매 실행층(`scripts/run_daily.py`, `quant/broker.py` KIS 어댑터)은 어떤 전략이든 같은 인터페이스로 돌릴 수 있게 남겨 두었다. 검증에서 살아남는 규칙이 나오기 전까지는 dry-run/모의투자로만 쓰는 것을 권한다.
"""
    (R / "FINDINGS.md").write_text(md, encoding="utf-8")
    print("->", R / "FINDINGS.md")


if __name__ == "__main__":
    main()
