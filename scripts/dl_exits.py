"""ML 예측 상위 K 종목에 장중 청산 규칙(손절/익절/트레일, 1시간봉 판정)을 적용했을 때의 성과.

Zarattini et al.(2024) 의 핵심이 '방향 예측' 보다 'stocks-in-play 선별 + 손절·종가 청산 리스크 관리' 였다는 점을 반영해,
같은 예측(LightGBM, 빠르게 재학습)에 대해 청산 규칙별 비용후 성과를 비교한다.

  python scripts/dl_exits.py --decision h1 --out reports/dl_daytrading
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.data import CACHE_DIR  # noqa: E402
from quant.dl.dataset import CTX_FEATS, build_dataset  # noqa: E402
from quant.dl.models import SEQ_SUMMARY_NAMES, cs_rank_target, seq_summary, train_lgbm  # noqa: E402
from quant.intraday import build_day_arrays, lookup_rows, simulate_exits  # noqa: E402
from scripts.dl_train import TEST_START, TRAIN_END, VAL_END, VAL_START  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", default="h1", choices=["h1", "open"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--commission-pct", type=float, default=0.0025)
    ap.add_argument("--out", default="reports/dl_daytrading")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(args.decision)
    key, seq, ctx = ds.key, ds.seq, ds.ctx
    y = cs_rank_target(key, "y_rod")
    d = key["date"]
    tr, va, te = (d <= TRAIN_END).to_numpy(), ((d >= VAL_START) & (d <= VAL_END)).to_numpy(), (d >= TEST_START).to_numpy()
    X = np.concatenate([ctx, seq_summary(seq)], axis=1)
    lgbm = train_lgbm(X[tr], y[tr], X[va], y[va], CTX_FEATS + SEQ_SUMMARY_NAMES)
    pred = lgbm.predict(X, num_iteration=lgbm.best_iteration)
    k_te = key[te].reset_index(drop=True).copy()
    k_te["p"] = pred[te]
    top = k_te.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(args.k).reset_index(drop=True)
    logging.info("test top-%d picks: %d", args.k, len(top))

    bars = pd.read_parquet(CACHE_DIR / "intraday_1h.parquet")
    bars = bars[bars["symbol"].isin(set(top["symbol"]))]
    dkey, arr = build_day_arrays(bars, "1h")
    rows = lookup_rows(dkey, top["symbol"].to_numpy(), top["date"].to_numpy())
    ok = rows >= 0
    top, rows = top[ok].reset_index(drop=True), rows[ok]
    o, h, l, c = (arr[f][rows] for f in ("open", "high", "low", "close"))
    entry_k = 0 if args.decision == "h1" else -1
    entry_px_raw = c[:, 0] if args.decision == "h1" else o[:, 0]
    slip = np.where(entry_px_raw < 5.0, 0.005, 0.001)
    entry_px = entry_px_raw * (1 + slip)
    n = len(top)
    exits = {"종가 청산": {}, "손절 2%": dict(stop_pct=0.02), "손절 3%": dict(stop_pct=0.03), "손절 5%": dict(stop_pct=0.05),
             "손절 3% + 익절 5%": dict(stop_pct=0.03, target_pct=0.05), "손절 3% + 익절 10%": dict(stop_pct=0.03, target_pct=0.10),
             "트레일 3%": dict(trail_pct=0.03), "트레일 5%": dict(trail_pct=0.05), "손절 3% + 트레일 5%": dict(stop_pct=0.03, trail_pct=0.05)}
    md = [f"# ML 상위 {args.k} 종목 + 장중 청산 규칙 (테스트 {TEST_START}~, 결정 시점 {args.decision}, LightGBM 예측)\n",
          f"거래 {n}건. 비용후 = 수수료 편도 {args.commission_pct * 100:.2f}%x2 + 슬리피지. 손절/익절/트레일은 1시간봉 저가/고가로 판정(같은 봉이면 손절 우선). "
          "시가 진입(open)은 첫 봉부터, 10:30 진입(h1)은 둘째 봉부터 판정.\n",
          "| 청산 규칙 | 승률 | 총평균/거래 | 비용후 평균/거래 | 중앙값 | 손절 비율 | 익절 비율 | 누적(비용후, 일별 동일비중) | 최대낙폭 |", "|---|---|---|---|---|---|---|---|---|"]
    for name, kw in exits.items():
        if args.decision == "open":
            # 시가 진입: 첫 봉 안에서의 손절/익절 판정을 위해 entry_k=-1 (모든 봉 판정)
            ex_px, reason = simulate_exits(o, h, l, c, entry_px, np.full(n, -1), np.full(n, c.shape[1] - 1), **kw)
        else:
            ex_px, reason = simulate_exits(o, h, l, c, entry_px, np.full(n, 0), np.full(n, c.shape[1] - 1), **kw)
        gross = ex_px / entry_px_raw - 1.0
        net = ex_px * (1 - slip) / entry_px - 1.0 - 2 * args.commission_pct
        daily = pd.Series(net).groupby(top["date"].to_numpy()).mean()
        eq = (1 + daily).cumprod()
        mdd = float((eq / eq.cummax() - 1).min())
        md.append(f"| {name} | {(net > 0).mean() * 100:.1f}% | {gross.mean() * 100:+.3f}% | {net.mean() * 100:+.3f}% | {np.median(net) * 100:+.3f}% | "
                  f"{(reason == 1).mean() * 100:.0f}% | {(reason == 2).mean() * 100:.0f}% | {(eq.iloc[-1] - 1) * 100:+.1f}% | {mdd * 100:.1f}% |")
    (out / f"exits_{args.decision}.md").write_text("\n".join(md), encoding="utf-8")
    logging.info("-> %s", out / f"exits_{args.decision}.md")


if __name__ == "__main__":
    main()
