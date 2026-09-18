"""급등주 공식 이벤트 스터디.

모든 스크린(quant/screens.py)을 학습(train)/검증(test) 구간에 적용해 신호 이후 수익률 통계를 만들고,
'급등일' 이벤트 전체를 특징(상승률·거래량배수·가격·시총·마감위치 등) 구간별로 쪼개 어느 값에서 기대값이 살아나는지 본다.

  python scripts/event_study.py --train 2023-11-01:2025-09-17 --test 2025-09-18:2026-09-17 --out reports/surge_study
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import data  # noqa: E402
from quant.events import bucket_stats, fmt_stats_table, forward_returns, summarize_events  # noqa: E402
from quant.screens import SCREENS, _fin, _mcap_ok, compute_features  # noqa: E402

FEATS = ["ret1", "vol_ratio", "close_pos", "price", "mcap", "shares", "gap", "consec_up", "runup30", "days_from_high52", "dv", "atr_pct", "range40_prev", "turnover"]
RET_COLS = {"next_open": ["ret_gap", "ret_oc", "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d"],
            "close": ["ret_on", "ret_1d", "ret_3d", "ret_5d", "ret_10d"],
            "open": ["ret_oc", "mfe1", "mae1", "ret_1d", "ret_3d", "ret_5d", "ret_10d"]}
RET_LABEL = {"ret_gap": "신호종가→진입시가(갭)", "ret_oc": "진입시가→당일종가(단타)", "ret_on": "종가→다음날시가(오버나잇)", "mfe1": "당일 최고가", "mae1": "당일 최저가",
             "ret_1d": "1일 보유", "ret_2d": "2일 보유", "ret_3d": "3일 보유", "ret_5d": "5일 보유", "ret_10d": "10일 보유"}
BUCKETS = {
    "ret1": ([0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 100], "당일 상승률"),
    "vol_ratio": ([2, 3, 5, 10, 20, 50, 1e9], "거래량 배수(20일)"),
    "close_pos": ([0, 0.4, 0.7, 0.9, 1.0], "종가 위치(0=저가,1=고가)"),
    "price": ([0, 1, 2, 5, 10, 20, 1e9], "주가($)"),
    "mcap": ([0, 5e7, 3e8, 2e9, 1e13], "근사 시총($)"),
    "shares": ([0, 1e7, 2e7, 5e7, 1e8, 1e12], "발행주식수(≈float 상한)"),
    "gap": ([-1, 0, 0.05, 0.10, 0.20, 0.50, 100], "갭(%)"),
    "consec_up": ([0, 1, 2, 3, 100], "연속 상승일수"),
    "runup30": ([-1, 0.2, 0.5, 1.0, 2.0, 100], "30일 저점 대비 상승률"),
    "days_from_high52": ([-1, -0.5, -0.2, -0.05, 100], "52주 고가 대비"),
    "dv": ([0, 2e6, 1e7, 5e7, 1e12], "당일 거래대금($)"),
    "atr_pct": ([0, 0.05, 0.10, 0.20, 10], "ATR/가격"),
    "turnover": ([0, 0.05, 0.2, 0.5, 1.0, 100], "회전율(거래량/발행주식수)"),
    "spy_above200": ([-0.5, 0.5, 1.5], "SPY > 200일선 (0/1)"),
    "iwm_above50": ([-0.5, 0.5, 1.5], "IWM(소형주) > 50일선 (0/1)"),
    "earnings": ([-0.5, 0.5, 1.5], "실적발표 재료 (0/1)"),
    "china": ([-0.5, 0.5, 1.5], "중국/홍콩 기업 (0/1)"),
    "weekday": ([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], "요일(0=월)"),
}


def load_context(uni: pd.DataFrame, trading_days: pd.DatetimeIndex):
    """이벤트에 붙일 맥락 정보: 시장 레짐(SPY/IWM), 실적발표일, 국가."""
    ctx = {}
    idx_file = data.CACHE_DIR / "index_close.parquet"
    if idx_file.exists():
        ix = pd.read_parquet(idx_file)
        ix.index = pd.to_datetime(ix.index)
        ctx["spy_above200"] = (ix["SPY"] > ix["SPY"].rolling(200).mean()).astype(float)
        ctx["iwm_above50"] = (ix["IWM"] > ix["IWM"].rolling(50).mean()).astype(float)
    earn_file = data.CACHE_DIR / "earnings_dates.parquet"
    earn = set()
    if earn_file.exists():
        e = pd.read_parquet(earn_file)
        e["date"] = pd.to_datetime(e["date"])
        pos = trading_days.searchsorted(e["date"].to_numpy())
        for sym, d, k in zip(e["symbol"], e["date"], pos):
            earn.add((sym, d))
            # 장 마감 후 발표 -> 다음 거래일 급등: 다음 거래일도 '실적 재료' 로 표시
            if k + 1 < len(trading_days):
                nxt = trading_days[k + 1] if trading_days[k] == d else (trading_days[k] if k < len(trading_days) else None)
                if nxt is not None:
                    earn.add((sym, nxt))
    ctx["earnings"] = earn
    if "country" in uni.columns:
        ctx["china"] = set(uni.loc[uni["country"].isin(["China", "Hong Kong"]), "symbol"])
    return ctx


def attach_context(fr: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    d = pd.to_datetime(fr["date"])
    for k in ("spy_above200", "iwm_above50"):
        if k in ctx:
            fr[k] = ctx[k].reindex(d).to_numpy()
    if ctx.get("earnings"):
        keys = list(zip(fr["symbol"], d))
        fr["earnings"] = np.array([1.0 if k in ctx["earnings"] else 0.0 for k in keys])
    if "china" in ctx:
        fr["china"] = fr["symbol"].isin(ctx["china"]).astype(float)
    fr["weekday"] = d.dt.weekday.astype(float)
    return fr


def window_mask(events: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    m = (events.index >= pd.Timestamp(start)) & (events.index <= pd.Timestamp(end))
    out = events.copy()
    out.loc[~m, :] = False
    return out


def cost_series(df: pd.DataFrame, base_bps: float, penny_bps: float, penny_price: float, commission_pct: float = 0.0025) -> pd.Series:
    """왕복 비용 = 2 x (슬리피지 + 편도 수수료)."""
    slip = np.where(df["entry_price"] < penny_price, penny_bps / 1e4, base_bps / 1e4)
    return pd.Series(2 * (slip + commission_pct), index=df.index)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="all")
    p.add_argument("--train", default="2023-11-01:2025-09-17")
    p.add_argument("--test", default="2025-09-18:2026-09-17")
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--penny-slippage-bps", type=float, default=50.0)
    p.add_argument("--penny-price", type=float, default=5.0)
    p.add_argument("--commission-pct", type=float, default=0.0025, help="편도 정률 수수료 (국내 증권사 해외주식 기본 0.25%%)")
    p.add_argument("--out", default="reports/surge_study")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    (out / "events").mkdir(parents=True, exist_ok=True)
    windows = {"train": args.train.split(":"), "test": args.test.split(":")}

    uni = data.load_universe(args.universe)
    panels = data.load_panels(universe=uni)
    stocks, bench = data.split_benchmark(panels)
    logging.info("%d symbols; computing features ...", stocks["close"].shape[1])
    feats = compute_features(stocks)
    ctx = load_context(uni, stocks["close"].index)
    logging.info("context: regime=%s earnings_dates=%d china=%d", "spy_above200" in ctx, len(ctx.get("earnings", ())), len(ctx.get("china", ())))

    md = [f"# 급등주 공식 이벤트 스터디 ({args.universe}, {stocks['close'].shape[1]}종목)\n",
          f"학습(train) {windows['train'][0]}~{windows['train'][1]} / 검증(test) {windows['test'][0]}~{windows['test'][1]}. "
          f"'비용후' = 왕복 비용 차감: 수수료 편도 {args.commission_pct * 100:.2f}%(국내 증권사 해외주식 기준) x2 + 슬리피지 {2 * args.slippage_bps:.0f}bp(주가 ≥ ${args.penny_price:.0f}) / {2 * args.penny_slippage_bps:.0f}bp(동전주). "
          "승률은 비용후 기준. t = 비용후 평균의 t-통계량(|t|>2 면 우연으로 보기 어려움).\n",
          "각 표는 **같은 이벤트를 어떻게 매매했을 때** 의 결과다: 단타(다음날 시가 매수·종가 매도), N일 보유(다음날 시가 매수·N일째 종가 매도).\n"]
    overview = []
    all_events = {}
    for name, (fn, mode) in SCREENS.items():
        ev_all = fn(stocks, feats)
        md.append(f"## {name}\n\n> {fn.__doc__.strip()}\n")
        for wname, (ws, we) in windows.items():
            ev = window_mask(ev_all, ws, we)
            fr = forward_returns(stocks, ev, mode, features=feats, feature_names=FEATS)
            fr["cost"] = cost_series(fr, args.slippage_bps, args.penny_slippage_bps, args.penny_price, args.commission_pct)
            all_events[(name, wname)] = fr
            fr.to_csv(out / "events" / f"{name}_{wname}.csv.gz", index=False, compression="gzip")
            tbl = summarize_events(fr, RET_COLS[mode], fr["cost"])
            tbl.index = [RET_LABEL.get(i, i) for i in tbl.index]
            n_ev = len(fr)
            per_day = n_ev / max(1, (pd.Timestamp(we) - pd.Timestamp(ws)).days * 252 / 365)
            md.append(f"### {wname} — 이벤트 {n_ev}건 (하루 평균 {per_day:.1f}건)\n\n{fmt_stats_table(tbl, '매매 방식')}\n")
            key = "ret_on" if mode == "close" else "ret_oc"
            s_key, s_3 = (fr[key] - fr["cost"]), (fr["ret_3d"] - fr["cost"]) if "ret_3d" in fr else None
            overview.append({"screen": name, "window": wname, "n": n_ev, "key": RET_LABEL[key], "key_win": (s_key > 0).mean() if n_ev else np.nan,
                             "key_mean": s_key.mean() if n_ev else np.nan, "d3_win": (s_3 > 0).mean() if n_ev else np.nan, "d3_mean": s_3.mean() if n_ev else np.nan})
            logging.info("%s/%s: n=%d %s win=%.1f%% mean=%.2f%% | 3d win=%.1f%% mean=%.2f%%", name, wname, n_ev, key,
                         (s_key > 0).mean() * 100 if n_ev else 0, s_key.mean() * 100 if n_ev else 0,
                         (s_3 > 0).mean() * 100 if n_ev else 0, s_3.mean() * 100 if n_ev else 0)

    # ---- 급등일 전체를 특징 구간별로: '적절값' 탐색
    md.append("## 적절값 탐색 — '급등일' 이벤트를 특징 구간별로 쪼개 보기\n\n"
              "급등일 = 당일 +5% 이상, 거래량 20일 평균의 2배 이상, 당일 거래대금 $1M 이상, 근사 시총 $2B 이하. "
              "다음날 시가에 사서 (a) 당일 종가 매도(단타) (b) 3일째 종가 매도 (c) 5일째 종가 매도. 각 특징 구간별 비용후 기대값.\n")
    base = _fin((feats["ret1"] >= 0.05) & (feats["vol_ratio"] >= 2.0) & (feats["dv"] >= 1e6) & _mcap_ok(feats, max_mcap=2e9))
    for wname, (ws, we) in windows.items():
        fr = forward_returns(stocks, window_mask(base, ws, we), "next_open", features=feats, feature_names=FEATS)
        fr = attach_context(fr, ctx)
        fr["cost"] = cost_series(fr, args.slippage_bps, args.penny_slippage_bps, args.penny_price, args.commission_pct)
        fr.to_csv(out / "events" / f"surge_base_{wname}.csv.gz", index=False, compression="gzip")
        md.append(f"### {wname} — 급등일 {len(fr)}건\n")
        for feat, (bins, label) in BUCKETS.items():
            if feat == "gap":
                continue
            for rc, rl in [("ret_oc", "단타"), ("ret_3d", "3일 보유"), ("ret_5d", "5일 보유")]:
                tbl = bucket_stats(fr, feat, bins, rc, "cost")
                md.append(f"#### {label} × {rl}\n\n{fmt_stats_table(tbl, label)}\n")
    # ---- 갭 이벤트 구간별
    md.append("## 갭 상승 시가 매수(당일 종가 매도) — 갭 크기/가격/시총 구간별\n\n갭 = 당일 시가/전일 종가 - 1 ≥ 3%, 직전 20일 평균 거래대금 $1M 이상, 시가 $0.5 이상.\n")
    gap_base = _fin((feats["gap"] >= 0.03) & (feats["adv20_prev"] >= 1e6) & (stocks["open"] >= 0.5))
    for wname, (ws, we) in windows.items():
        fr = forward_returns(stocks, window_mask(gap_base, ws, we), "open", features=feats, feature_names=FEATS)
        fr["cost"] = cost_series(fr, args.slippage_bps, args.penny_slippage_bps, args.penny_price, args.commission_pct)
        fr.to_csv(out / "events" / f"gap_base_{wname}.csv.gz", index=False, compression="gzip")
        md.append(f"### {wname} — 갭 이벤트 {len(fr)}건\n")
        for feat in ["gap", "price", "mcap", "shares"]:
            bins, label = BUCKETS[feat]
            tbl = bucket_stats(fr, feat, bins, "ret_oc", "cost")
            md.append(f"#### {label} × 시가→종가\n\n{fmt_stats_table(tbl, label)}\n")
            tbl = bucket_stats(fr, feat, bins, "ret_3d", "cost")
            md.append(f"#### {label} × 3일 보유\n\n{fmt_stats_table(tbl, label)}\n")

    ov = pd.DataFrame(overview)
    lines = ["| 스크린 | 구간 | 이벤트 | 대표 매매 | 승률 | 비용후 평균 | 3일보유 승률 | 3일보유 평균 |", "|---|---|---|---|---|---|---|---|"]
    for _, r in ov.iterrows():
        lines.append(f"| {r['screen']} | {r['window']} | {r['n']} | {r['key']} | {r['key_win'] * 100:.1f}% | {r['key_mean'] * 100:+.2f}% | "
                     f"{r['d3_win'] * 100:.1f}% | {r['d3_mean'] * 100:+.2f}% |")
    md.insert(3, "## 한눈에 보기\n\n" + "\n".join(lines) + "\n")
    ov.to_csv(out / "overview.csv", index=False)
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")
    logging.info("-> %s", out / "README.md")


if __name__ == "__main__":
    main()
