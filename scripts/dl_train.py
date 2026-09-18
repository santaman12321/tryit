"""딥러닝/ML 데이트레이딩 모델 학습·평가.

  python scripts/dl_train.py --decision h1 --out reports/dl_daytrading [--max-symbols 300 --epochs 2]

분할(시간 기준, 엠바고 5거래일): train < 2025-07-01, val 2025-07-01~2025-09-30, test 2025-10-01~ (최근 1년 아웃오브샘플)
모델: LightGBM(맥락+시퀀스 요약), GRU(시퀀스+맥락), 앙상블(순위 평균), 문헌 규칙(장중 모멘텀, stocks-in-play, 갭 페이드, 단기 반전, 20일 모멘텀)
평가: 일별 순위 IC, 십분위 스프레드, 상위 K 매수 포트폴리오(비용후: 수수료 0.25%x2 + 슬리피지 0.1%x2, 동전주 0.5%x2)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.dl.dataset import CTX_FEATS, build_dataset  # noqa: E402
from quant.dl.models import SEQ_SUMMARY_NAMES, cs_rank_target, daily_ic, decile_spread, predict_gru, seq_summary, topk_portfolio, train_gru, train_lgbm  # noqa: E402

TRAIN_END, VAL_START, VAL_END, TEST_START = "2025-06-23", "2025-07-01", "2025-09-23", "2025-10-01"


def fmt_port(p: dict) -> str:
    return f"{p['n_trades']} / {p['win'] * 100:.1f}% / {p['gross_mean'] * 100:+.3f}% / {p['net_mean'] * 100:+.3f}% / {p['total_return'] * 100:+.1f}% / {p['sharpe']:.2f} / {p['max_dd'] * 100:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", default="h1", choices=["h1", "open"])
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--commission-pct", type=float, default=0.0025)
    ap.add_argument("--out", default="reports/dl_daytrading")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(args.decision, max_symbols=args.max_symbols)
    key, seq, ctx = ds.key, ds.seq, ds.ctx
    y = cs_rank_target(key, "y_rod")
    d = key["date"]
    tr = (d <= TRAIN_END).to_numpy()
    va = ((d >= VAL_START) & (d <= VAL_END)).to_numpy()
    te = (d >= TEST_START).to_numpy()
    logging.info("split: train %d, val %d, test %d", tr.sum(), va.sum(), te.sum())
    slip = np.where(key["dec_px"].to_numpy() < 5.0, 0.005, 0.001)
    cost_rt = 2 * (slip + args.commission_pct)

    # ---- LightGBM
    X = np.concatenate([ctx, seq_summary(seq)], axis=1)
    names = CTX_FEATS + SEQ_SUMMARY_NAMES
    lgbm = train_lgbm(X[tr], y[tr], X[va], y[va], names)
    p_lgb = lgbm.predict(X, num_iteration=lgbm.best_iteration)
    imp = pd.Series(lgbm.feature_importance("gain"), index=names).sort_values(ascending=False)
    # ---- GRU
    gru = train_gru(seq[tr], ctx[tr], y[tr], seq[va], ctx[va], y[va], key[va].reset_index(drop=True), epochs=args.epochs)
    p_gru = predict_gru(gru, seq, ctx)
    # ---- 앙상블 (일자별 순위 평균)
    tmp = pd.DataFrame({"date": key["date"].to_numpy(), "a": p_lgb, "b": p_gru})
    p_ens = (tmp.groupby("date")["a"].rank(pct=True) + tmp.groupby("date")["b"].rank(pct=True)).to_numpy() / 2
    # ---- 규칙 베이스라인
    rules = {
        "장중 모멘텀(첫1h 수익률↑)": key["h1_ret"].to_numpy(),
        "stocks-in-play(첫1h 상대거래량↑, 양봉만)": np.where(key["h1_ret"].to_numpy() > 0, key["h1_rvol"].to_numpy(), -1.0),
        "갭 페이드(갭↓)": -key["gap"].to_numpy(),
        "단기 반전(전일 수익률↓)": -key["prev_ret"].to_numpy(),
        "20일 모멘텀(↑)": key["ret20"].to_numpy(),
        "52주 고가 근접(↑)": key["from_high52"].to_numpy(),
    }
    if args.decision == "open":
        rules.pop("장중 모멘텀(첫1h 수익률↑)")
        rules.pop("stocks-in-play(첫1h 상대거래량↑, 양봉만)")
    models = {"LightGBM": p_lgb, "GRU": p_gru, "앙상블(LGBM+GRU)": p_ens, **rules}

    md = [f"# 딥러닝/ML 데이트레이딩 모델 — 결정 시점 {'첫 1시간봉 마감(10:30)' if args.decision == 'h1' else '시가'}, 청산 당일 종가\n",
          f"샘플 {len(key):,}개 ({key['symbol'].nunique()}종목, {key['date'].min().date()}~{key['date'].max().date()}), 유동성 필터 20일 평균 거래대금 ≥ $2M·주가 ≥ $2. "
          f"분할: 학습 ≤ {TRAIN_END} ({tr.sum():,}), 검증 {VAL_START}~{VAL_END} ({va.sum():,}), 테스트 ≥ {TEST_START} ({te.sum():,}). "
          f"비용후 = 수수료 편도 {args.commission_pct * 100:.2f}%x2 + 슬리피지 편도 0.1%(주가 ≥ $5)/0.5%(동전주) x2.\n",
          "목표값 = 일자별 횡단면 순위(결정 시점→종가 수익률). 평가 = 일별 스피어만 IC, 십분위 스프레드, 매일 상위 K 종목 동일비중 매수.\n",
          "## 결과 (테스트 = 최근 1년 아웃오브샘플)\n",
          f"| 모델 | IC 평균 | IC t | IC>0 일 비율 | 십분위 스프레드(상-하) | 상위{args.k} 거래/승률/총평균/비용후평균/누적수익/샤프/MDD |", "|---|---|---|---|---|---|"]
    port_curves = {}
    rows = []
    for split_name, mask in (("검증", va), ("테스트", te)):
        md.append(f"| **{split_name}** | | | | | |")
        k_ = key[mask].reset_index(drop=True)
        y_ = y[mask]
        for name, pred in models.items():
            p_ = pred[mask]
            ic = daily_ic(k_, p_, y_)
            dec = decile_spread(k_, p_, "y_rod")
            port = topk_portfolio(k_, p_, "y_rod", args.k, cost_rt[mask])
            md.append(f"| {name} | {ic['ic_mean']:+.4f} | {ic['ic_t']:+.1f} | {ic['ic_pos'] * 100:.0f}% | {dec['spread'] * 100:+.2f}% ({dec['top'] * 100:+.2f}/{dec['bottom'] * 100:+.2f}) | {fmt_port(port)} |")
            rows.append({"split": split_name, "model": name, **{k: v for k, v in ic.items()}, "decile_spread": dec["spread"], "top_decile": dec["top"],
                         **{k: v for k, v in port.items() if k not in ("daily", "trades")}})
            if split_name == "테스트":
                port_curves[name] = port["daily"]
    pd.DataFrame(rows).to_csv(out / f"results_{args.decision}.csv", index=False)

    # ---- 시총 구간별 (테스트, LightGBM/앙상블)
    md.append("\n## 시총 구간별 (테스트, 앙상블 상위 K)\n\n| 시총 구간 | 거래 | 승률 | 총평균 | 비용후 평균 | 누적수익 |\n|---|---|---|---|---|---|")
    k_te = key[te].reset_index(drop=True)
    mc = np.exp(k_te["log_mcap"].to_numpy())
    for label, lo, hi in (("<$300M", 0, 3e8), ("$300M~$2B", 3e8, 2e9), ("$2B~$10B", 2e9, 1e10), (">$10B", 1e10, 1e15)):
        m = (mc >= lo) & (mc < hi)
        if m.sum() < 500:
            continue
        port = topk_portfolio(k_te[m].reset_index(drop=True), p_ens[te][m], "y_rod", args.k, cost_rt[te][m])
        md.append(f"| {label} | {port['n_trades']} | {port['win'] * 100:.1f}% | {port['gross_mean'] * 100:+.3f}% | {port['net_mean'] * 100:+.3f}% | {port['total_return'] * 100:+.1f}% |")

    # ---- 다른 보유기간 (같은 예측으로 오버나잇/1일 보유 시)
    md.append("\n## 같은 예측을 다른 청산에 적용 (테스트, 앙상블 상위 K)\n\n| 청산 | 거래 | 승률 | 총평균 | 비용후 평균 | 누적수익 |\n|---|---|---|---|---|---|")
    for label, col in (("당일 종가(데이트레이딩)", "y_rod"), ("다음날 같은 시점(1일 보유)", "y_nd")):
        port = topk_portfolio(k_te, p_ens[te], col, args.k, cost_rt[te])
        md.append(f"| {label} | {port['n_trades']} | {port['win'] * 100:.1f}% | {port['gross_mean'] * 100:+.3f}% | {port['net_mean'] * 100:+.3f}% | {port['total_return'] * 100:+.1f}% |")

    # ---- 손익분기 비용
    ens_te = topk_portfolio(k_te, p_ens[te], "y_rod", args.k, np.zeros(te.sum()))
    md.append(f"\n**손익분기 왕복 비용**(앙상블 상위 {args.k}, 테스트): 총평균 {ens_te['gross_mean'] * 100:+.3f}%/거래 — 이보다 왕복 비용이 크면 손실. "
              f"국내 증권사 기본 수수료(0.25%x2=0.5%)+슬리피지(0.2%)=0.7%.\n")
    md.append("## LightGBM 특징 중요도 (gain 상위 15)\n\n| 특징 | gain |\n|---|---|")
    for n_, v in imp.head(15).items():
        md.append(f"| {n_} | {v:,.0f} |")

    # ---- 차트
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, daily in port_curves.items():
        if name in ("LightGBM", "GRU", "앙상블(LGBM+GRU)", "장중 모멘텀(첫1h 수익률↑)", "stocks-in-play(첫1h 상대거래량↑, 양봉만)"):
            eq = (1 + daily["net"]).cumprod()
            ax.plot(eq.index, eq.values, label=name)
    ax.set_title(f"Top-{args.k} long, net of costs, test period ({args.decision})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"equity_{args.decision}.png", dpi=110)
    (out / f"README_{args.decision}.md").write_text("\n".join(md), encoding="utf-8")
    logging.info("-> %s", out / f"README_{args.decision}.md")


if __name__ == "__main__":
    main()
