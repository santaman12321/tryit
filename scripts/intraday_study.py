"""장중 이벤트 스터디: 급등 다음날(또는 갭 당일)의 1시간봉/15분봉 경로로 진입·청산 규칙을 검증한다.

진입(1시간봉 기준 판단):
  open      : 시가
  h1_green  : 첫 1시간봉이 양봉(종가>시가)이고 봉 상단 마감(close_pos>=0.6) -> 10:30 종가 진입
  orb60     : 2시간째에 첫 1시간봉 고가를 돌파하면 그 고가에 진입 (돌파 확인 매수)
  h1_pull   : 첫 봉 양봉, 둘째 봉 음봉이지만 첫 봉 시가 위에서 마감(눌림 유지) -> 11:30 진입
청산(봉 단위, 15m 데이터면 15분 단위로 손절/익절 판정):
  시간청산(12:30/14:30/종가), 손절 2/3/5%, 익절 3/5/10%, 트레일 3/5%
비용: 편도 수수료 0.25% x2 + 슬리피지 10bp(≥$5)/50bp(<$5) x2

  python scripts/intraday_study.py --interval 1h --out reports/surge_study/intraday_1h
  python scripts/intraday_study.py --interval 15m --out reports/surge_study/intraday_15m
"""
from __future__ import annotations

import argparse
import itertools
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import data  # noqa: E402
from quant.intraday import build_day_arrays, lookup_rows, simulate_exits, slots_for  # noqa: E402
from quant.screens import _fin, _mcap_ok, compute_features  # noqa: E402

REASON = {0: "time", 1: "stop", 2: "target", 3: "trail"}


def stats(net: np.ndarray) -> dict:
    net = net[np.isfinite(net)]
    n = len(net)
    if n == 0:
        return {"n": 0}
    wins, losses = net[net > 0], net[net <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    t = net.mean() / (net.std(ddof=1) / math.sqrt(n)) if n > 1 and net.std(ddof=1) > 0 else float("nan")
    return {"n": n, "win": (net > 0).mean(), "mean": net.mean(), "median": float(np.median(net)), "pf": pf, "t": t,
            "avg_win": wins.mean() if len(wins) else 0.0, "avg_loss": losses.mean() if len(losses) else 0.0}


def fmt_row(name: str, s: dict) -> str:
    if s.get("n", 0) == 0:
        return f"| {name} | 0 | - | - | - | - | - | - | - |"
    return (f"| {name} | {s['n']} | {s['win'] * 100:.1f}% | {s['mean'] * 100:+.2f}% | {s['median'] * 100:+.2f}% | {s['pf']:.2f} | "
            f"{s['avg_win'] * 100:+.2f}% | {s['avg_loss'] * 100:+.2f}% | {s['t']:.1f} |")


HEAD = "| 규칙 | n | 승률 | 평균(비용후) | 중앙값 | PF | 평균 수익 | 평균 손실 | t |\n|---|---|---|---|---|---|---|---|---|"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", default="1h", choices=["1h", "15m"])
    p.add_argument("--universe", default="all")
    p.add_argument("--split", default="2025-09-18", help="이 날짜부터 test")
    p.add_argument("--commission-pct", type=float, default=0.0025)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--penny-slippage-bps", type=float, default=50.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bph = 60 // {"1h": 60, "15m": 15}[args.interval]  # bars per hour
    slots = slots_for(args.interval)
    m = len(slots)
    k_h1 = bph - 1  # 첫 1시간의 마지막 봉 index
    k_h2 = 2 * bph - 1
    k_1230 = 3 * bph - 1
    k_1430 = 5 * bph - 1
    k_close = m - 1

    logging.info("loading daily panels ...")
    uni = data.load_universe(args.universe)
    stocks, bench = data.split_benchmark(data.load_panels(universe=uni))
    feats = compute_features(stocks)
    logging.info("loading %s bars ...", args.interval)
    bars = pd.read_parquet(data.CACHE_DIR / f"intraday_{args.interval}.parquet")
    key, arr = build_day_arrays(bars, args.interval)
    logging.info("%d symbol-days with %s bars (%s ~ %s)", len(key), args.interval, key["date"].min().date(), key["date"].max().date())

    # ---- 이벤트: 급등일(전일) -> 거래일 D = 다음 거래일
    surge = _fin((feats["ret1"] >= 0.05) & (feats["vol_ratio"] >= 2.0) & (feats["dv"] >= 1e6) & _mcap_ok(feats, max_mcap=2e9))
    idx = stocks["close"].index
    ti, si = np.nonzero(surge.to_numpy())
    keep = ti + 1 < len(idx)
    ti, si = ti[keep], si[keep]
    syms = np.array(stocks["close"].columns, dtype=object)[si]
    d_trade = idx[ti + 1]
    rows = lookup_rows(key, syms, d_trade.to_numpy())
    ok = rows >= 0
    logging.info("surge events %d, with intraday bars %d", len(rows), ok.sum())
    ev = pd.DataFrame({"symbol": syms[ok], "signal_date": idx[ti[ok]], "trade_date": d_trade[ok], "row": rows[ok]})
    for f in ["ret1", "vol_ratio", "close_pos", "price", "mcap", "days_from_high52", "consec_up", "vol_avg20_prev"]:
        ev[f] = feats[f].to_numpy()[ti[ok], si[ok]]
    ev["prev_close"] = stocks["close"].to_numpy()[ti[ok], si[ok]]
    ev["prev_high"] = stocks["high"].to_numpy()[ti[ok], si[ok]]
    ev["test"] = ev["trade_date"] >= pd.Timestamp(args.split)

    R = ev["row"].to_numpy()
    o, h, l, c, v = (arr[f][R] for f in ("open", "high", "low", "close", "volume"))
    # 첫 1시간 봉 집계
    h1_open, h1_close = o[:, 0], c[:, k_h1]
    h1_high, h1_low = h[:, : k_h1 + 1].max(axis=1), l[:, : k_h1 + 1].min(axis=1)
    h1_vol = v[:, : k_h1 + 1].sum(axis=1)
    ev["gap"] = h1_open / ev["prev_close"].to_numpy() - 1.0
    ev["h1_ret"] = h1_close / h1_open - 1.0
    rng = h1_high - h1_low
    ev["h1_close_pos"] = np.where(rng > 0, (h1_close - h1_low) / np.where(rng > 0, rng, 1), 1.0)
    ev["h1_rvol"] = h1_vol / (ev["vol_avg20_prev"].to_numpy() / 6.5)  # 첫 1시간 거래량 / (20일 평균 일거래량의 1시간 몫)
    ev["day_ret_oc"] = c[:, k_close] / h1_open - 1.0
    ev["k_high"] = h.argmax(axis=1)
    ev["k_low"] = l.argmin(axis=1)
    # VWAP (당일 누적) — 커뮤니티 'VWAP 위에서만 매수 / VWAP 회복 매수' 검증용
    tp = (h + l + c) / 3.0
    cum_v = np.cumsum(v, axis=1)
    vwap = np.where(cum_v > 0, np.cumsum(tp * v, axis=1) / np.where(cum_v > 0, cum_v, 1), c)
    above_h1 = c[:, k_h1] > vwap[:, k_h1]
    vwap_up = vwap[:, k_h1] >= vwap[:, max(k_h1 - 1, 0)]
    # 첫 2시간 안에 'VWAP 아래 -> 위' 로 회복한 첫 봉
    cross = (c[:, :k_h2] < vwap[:, :k_h2]) & (c[:, 1 : k_h2 + 1] > vwap[:, 1 : k_h2 + 1])
    has_cross = cross.any(axis=1)
    k_cross = np.where(has_cross, cross.argmax(axis=1) + 1, k_h2)
    px_cross = c[np.arange(len(ev)), k_cross]
    slip = np.where(h1_open < 5.0, args.penny_slippage_bps / 1e4, args.slippage_bps / 1e4)
    cost_rt = 2 * (slip + args.commission_pct)
    ev["cost_rt"] = cost_rt

    # ---- 진입 규칙: (마스크, 진입가, 진입 봉 index)
    entries = {
        "open(시가)": (np.ones(len(ev), bool), h1_open * (1 + slip), -1),
        "h1_green(첫1h 양봉·상단마감→10:30)": ((h1_close > h1_open) & (ev["h1_close_pos"].to_numpy() >= 0.6), h1_close * (1 + slip), k_h1),
        "h1_green_strong(+위조건 & 첫1h +3%↑)": ((h1_close > h1_open) & (ev["h1_close_pos"].to_numpy() >= 0.6) & (ev["h1_ret"].to_numpy() >= 0.03), h1_close * (1 + slip), k_h1),
        "orb60(2h째 첫1h 고가 돌파)": ((h[:, k_h1 + 1 : k_h2 + 1].max(axis=1) > h1_high) & (h1_close > h1_open), h1_high * (1 + slip), k_h1),
        "h1_pull(둘째1h 눌림 유지→11:30)": ((h1_close > h1_open) & (c[:, k_h2] < c[:, k_h1]) & (c[:, k_h2] > h1_open), c[:, k_h2] * (1 + slip), k_h2),
        "h1_vwap(10:30 VWAP 위 & VWAP 상승)": (above_h1 & vwap_up & (h1_close > h1_open), h1_close * (1 + slip), k_h1),
        "vwap_reclaim(첫2h 내 VWAP 회복봉 종가)": (has_cross & (k_cross >= 1), px_cross * (1 + slip), k_cross),
    }
    exits = {"→12:30": dict(exit_k=k_1230), "→14:30": dict(exit_k=k_1430), "→종가": dict(exit_k=k_close)}
    risk = {"손절없음": {}, "손절2%": dict(stop_pct=0.02), "손절3%": dict(stop_pct=0.03), "손절5%": dict(stop_pct=0.05),
            "손절3%+익절5%": dict(stop_pct=0.03, target_pct=0.05), "손절3%+익절10%": dict(stop_pct=0.03, target_pct=0.10),
            "손절5%+익절10%": dict(stop_pct=0.05, target_pct=0.10), "트레일3%": dict(trail_pct=0.03), "트레일5%": dict(trail_pct=0.05),
            "손절3%+트레일5%": dict(stop_pct=0.03, trail_pct=0.05)}

    md = [f"# 장중 스터디 ({args.interval} 봉, 급등 다음날, {len(ev)}건: train {int((~ev.test).sum())} / test {int(ev.test.sum())})\n",
          f"급등일 = 전일 +5%↑·거래량 20일평균 2배↑·거래대금 $1M↑·근사시총 $2B↓. 다음 거래일의 {args.interval} 봉 경로로 검증. "
          f"비용후 = 편도 수수료 {args.commission_pct * 100:.2f}%x2 + 슬리피지 왕복 {2 * args.slippage_bps:.0f}bp(≥$5)/{2 * args.penny_slippage_bps:.0f}bp(<$5). "
          f"손절/익절은 {args.interval} 봉의 저가/고가로 판정(같은 봉에 둘 다 걸리면 손절 우선).\n"]

    # ---- 장중 경로 통계
    md.append("## 장중 경로: 고가/저가가 찍히는 시간대\n")
    for wname, mask in [("train", ~ev.test.to_numpy()), ("test", ev.test.to_numpy())]:
        kh = pd.Series(ev["k_high"].to_numpy()[mask]).value_counts(normalize=True).sort_index()
        kl = pd.Series(ev["k_low"].to_numpy()[mask]).value_counts(normalize=True).sort_index()
        md.append(f"### {wname}\n\n| 봉 시작 | 당일 고가 비율 | 당일 저가 비율 |\n|---|---|---|")
        for k in range(m):
            md.append(f"| {slots[k]} | {kh.get(k, 0) * 100:.1f}% | {kl.get(k, 0) * 100:.1f}% |")
        oc = ev["day_ret_oc"].to_numpy()[mask]
        md.append(f"\n시가→종가 평균 {oc.mean() * 100:+.2f}%, 시가보다 낮게 마감 {(oc < 0).mean() * 100:.1f}%, 첫 1시간 양봉 비율 {(ev['h1_ret'].to_numpy()[mask] > 0).mean() * 100:.1f}%\n")

    # ---- 봉별 수익률 (디시 글: "개장~11:30, 13~14시, 15:30~마감 에 오르는 구간이 있다")
    md.append("## 봉별(시간대별) 수익률 — 각 봉의 시가→종가, 비용 전\n")
    for wname, mask in [("train", ~ev.test.to_numpy()), ("test", ev.test.to_numpy())]:
        md.append(f"### {wname}\n\n| 봉 시작 | 평균 | 중앙값 | 양봉 비율 |\n|---|---|---|---|")
        for k in range(m):
            r = c[mask, k] / o[mask, k] - 1.0
            r = r[np.isfinite(r)]
            md.append(f"| {slots[k]} | {r.mean() * 100:+.2f}% | {np.median(r) * 100:+.2f}% | {(r > 0).mean() * 100:.1f}% |")
        md.append("")

    # ---- 전체 조합
    results = []
    md.append("## 진입 × 청산 × 리스크 규칙 (비용후)\n")
    for ename, (emask, epx, ek) in entries.items():
        ek_arr = np.full(len(ev), ek) if np.isscalar(ek) else np.asarray(ek)
        md.append(f"### 진입: {ename}\n")
        for wname, wmask in [("train", ~ev.test.to_numpy()), ("test", ev.test.to_numpy())]:
            mask = emask & wmask & np.isfinite(epx)
            if mask.sum() == 0:
                continue
            md.append(f"#### {wname} (n={int(mask.sum())})\n\n{HEAD}")
            for xname, xkw in exits.items():
                for rname, rkw in risk.items():
                    ex_px, reason = simulate_exits(o[mask], h[mask], l[mask], c[mask], epx[mask], ek_arr[mask], np.full(mask.sum(), xkw["exit_k"]), **rkw)
                    net = ex_px * (1 - slip[mask]) / epx[mask] - 1.0 - 2 * args.commission_pct
                    s = stats(net)
                    s.update(entry=ename, exit=xname, risk=rname, window=wname)
                    results.append(s)
                    md.append(fmt_row(f"{xname} {rname}", s))
            md.append("")
    res = pd.DataFrame(results)
    res.to_csv(out / "combos.csv", index=False)

    # ---- 승률 우선 탐색: 두 기간 모두 승률·평균 양수인 조합
    piv = res.pivot_table(index=["entry", "exit", "risk"], columns="window", values=["n", "win", "mean", "pf"])
    both = piv[(piv[("n", "train")] >= 100) & (piv[("n", "test")] >= 100)].copy()
    both["min_win"] = both[[("win", "train"), ("win", "test")]].min(axis=1)
    both["min_mean"] = both[[("mean", "train"), ("mean", "test")]].min(axis=1)
    top = both.sort_values("min_win", ascending=False).head(15)
    md.insert(3, "## 한눈에: 두 기간 모두 n≥100 인 조합을 '낮은 쪽 승률' 순으로\n\n| 진입 | 청산 | 리스크 | train n/승률/평균/PF | test n/승률/평균/PF |\n|---|---|---|---|---|\n" + "\n".join(
        f"| {e} | {x} | {r} | {int(row[('n', 'train')])} / {row[('win', 'train')] * 100:.1f}% / {row[('mean', 'train')] * 100:+.2f}% / {row[('pf', 'train')]:.2f} | "
        f"{int(row[('n', 'test')])} / {row[('win', 'test')] * 100:.1f}% / {row[('mean', 'test')] * 100:+.2f}% / {row[('pf', 'test')]:.2f} |"
        for (e, x, r), row in top.iterrows()) + "\n\n두 기간 모두 평균(비용후) > 0 인 조합 수: "
        f"{int((both['min_mean'] > 0).sum())} / {len(both)}\n")

    # ---- 조건부(특징 구간) 승률: h1_green 진입 · 종가 청산 · 손절3% 를 기준으로
    md.append("## 조건부 승률 — 진입 h1_green, 청산 종가, 손절 3% 기준으로 특징 구간별\n")
    emask, epx, ek = entries["h1_green(첫1h 양봉·상단마감→10:30)"]
    base_mask = emask & np.isfinite(epx)
    ex_px, reason = simulate_exits(o, h, l, c, np.where(np.isfinite(epx), epx, 1.0), np.full(len(ev), ek), np.full(len(ev), k_close), stop_pct=0.03)
    net_all = ex_px * (1 - slip) / np.where(np.isfinite(epx), epx, 1.0) - 1.0 - 2 * args.commission_pct
    ev["net_h1green_stop3"] = np.where(base_mask, net_all, np.nan)
    ev.to_csv(out / "events.csv.gz", index=False, compression="gzip")
    buckets = {"ret1": [0.05, 0.1, 0.2, 0.3, 0.5, 100], "vol_ratio": [2, 5, 10, 20, 1e9], "price": [0, 2, 5, 10, 20, 1e9], "mcap": [0, 5e7, 3e8, 2e9],
               "gap": [-1, 0, 0.03, 0.1, 0.3, 100], "h1_ret": [-1, 0, 0.03, 0.06, 0.1, 100], "h1_rvol": [0, 2, 5, 10, 20, 1e9], "h1_close_pos": [0, 0.6, 0.8, 0.95, 1.0],
               "days_from_high52": [-1, -0.5, -0.2, 100], "close_pos": [0, 0.6, 0.9, 1.0]}
    for f, bins in buckets.items():
        md.append(f"### {f}\n\n| 구간 | train n | train 승률 | train 평균 | test n | test 승률 | test 평균 |\n|---|---|---|---|---|---|---|")
        cat = pd.cut(ev[f], bins, include_lowest=True)
        for b in cat.cat.categories:
            a = ev[(cat == b) & ~ev.test]["net_h1green_stop3"].dropna()
            z = ev[(cat == b) & ev.test]["net_h1green_stop3"].dropna()
            md.append(f"| {b} | {len(a)} | {(a > 0).mean() * 100 if len(a) else float('nan'):.1f}% | {a.mean() * 100 if len(a) else float('nan'):+.2f}% | "
                      f"{len(z)} | {(z > 0).mean() * 100 if len(z) else float('nan'):.1f}% | {z.mean() * 100 if len(z) else float('nan'):+.2f}% |")
        md.append("")
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")
    logging.info("-> %s", out / "README.md")


if __name__ == "__main__":
    main()
