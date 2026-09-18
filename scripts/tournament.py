"""모델 토너먼트: 모든 스크린/전략을 같은 조건에서 파라미터 탐색 → 학습 구간에서 선택 → 검증 구간에서 평가.

1단계 (스크린 자체 파라미터, 이벤트 스터디로 빠르게): 격자 전체를 학습 구간에 돌려 '대표 보유 수익률(비용후 평균)' 상위 K개 선택 (n>=100)
2단계 (청산 파라미터, 포트폴리오 백테스트): 상위 K개 × (손절 × 익절 × 보유일) → 학습 구간 기대수익/거래 최고 조합 선택 → 검증 구간 평가
전략 클래스(pullback/breakout/surge_chase/surge_pullback/gap_day)는 자체 격자로 바로 2단계.
'재료 있는 급등 제외(exclude_earnings)' 는 1단계 격자 차원으로 포함.

  python scripts/tournament.py --shard 0/3 --out reports/surge_study/tournament
  python scripts/tournament.py --merge --out reports/surge_study/tournament
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import data  # noqa: E402
from quant.backtest import BacktestConfig, run_backtest  # noqa: E402
from quant.events import forward_returns  # noqa: E402
from quant.metrics import summarize  # noqa: E402
from quant.screens import SCREENS, compute_features, earnings_mask  # noqa: E402
from quant.strategies import make_strategy  # noqa: E402
from quant.strategies.base import Signals, Strategy, nan_like  # noqa: E402

TRAIN = ("2023-11-01", "2025-09-17")
TEST = ("2025-09-18", "2026-09-17")
TOP_K = 2
MIN_EVENTS = 100
MIN_TRADES = 60

SCREEN_GRIDS = {
    "ross_5pillars": {"min_ret": [0.05, 0.10, 0.20], "min_rvol": [2, 5, 10], "max_price": [10, 20, 50]},
    "warrior_scanner": {"min_ret": [0.05, 0.10], "min_rvol": [1.5, 3, 5], "min_close_pos": [0.5, 0.7, 0.9]},
    "finviz_top_gainer": {"min_ret": [0.10, 0.20, 0.30], "min_rsi": [50, 60, 70]},
    "kr_screener_basic": {"min_ret": [0.05, 0.10], "min_rvol": [2, 4], "max_rsi": [70, 80]},
    "kr_newhigh_alignment": {"max_mcap": [5e8, 1e9, 5e9], "min_vol_ratio_prev": [1.5, 2, 3]},
    "big_candle_maxvol": {"min_ret": [0.10, 0.20, 0.30], "min_close_pos": [0.5, 0.8], "max_mcap": [3e8, 2e9]},
    "volume_3x_after_base": {"min_rvol": [3, 5, 10], "max_range40": [0.15, 0.25, 0.4]},
    "pullback_lowvol": {"surge_pct": [0.10, 0.15, 0.30], "vol_frac": [0.1, 0.2, 0.3], "hold_pct": [0.85, 0.9]},
    "pullback_ma": {"surge_pct": [0.10, 0.15, 0.30], "ma": ["sma5", "sma10"], "vol_frac": [0.3, 0.5]},
    "threads_us_pullback": {"min_runup30": [0.5, 0.9, 1.5], "max_mcap": [5e8, 1.8e9]},
    "quiet_volume_spike": {"max_abs_ret": [0.02, 0.03, 0.05], "min_rvol": [3, 5, 10]},
    "dryup_first_spike": {"dry_frac": [0.3, 0.5], "min_rvol": [3, 5], "min_ret": [0.03, 0.10]},
    "first_green_day": {"min_ret": [0.10, 0.15, 0.30], "min_close_pos": [0.6, 0.8], "beaten": [0.5, 0.7]},
    "multiday_runner_day2": {"day1_ret": [0.20, 0.30, 0.50], "day1_rvol": [5, 10, 20]},
    "gap_and_go": {"min_gap": [0.04, 0.10, 0.20], "max_price": [10, 20, 50]},
    "toss_second_volume": {"second_mult": [2, 3, 5], "window": [10, 30], "base_adv": [2e6, 1e7]},
    "toss_pullback_after_green": {"day2_max_ret": [-0.03, -0.08], "day2_vol_frac": [0.3, 0.5, 0.7], "base_adv": [2e6, 1e7]},
    "toss_dryup_after_red": {"dry_frac": [0.2, 0.4], "drop_from_first": [0.2, 0.4], "base_adv": [2e6, 1e7]},
    "toss_weak_green_rising_vol": {"max_ret": [0.03, 0.05, 0.10], "window": [5, 10, 20], "base_adv": [2e6, 1e7]},
}
STRATEGY_GRIDS = {
    "pullback": {"rsi_entry": [5, 10, 15], "max_hold_days": [5, 10], "stop_atr_mult": [None, 3.0], "max_mcap": [None, 2e9]},
    "breakout": {"lookback": [20, 55], "vol_mult": [1.5, 2.5], "trail_atr_mult": [3.0, 5.0], "max_mcap": [None, 2e9]},
    "surge_chase": {"surge_pct": [0.10, 0.15, 0.30], "vol_mult": [3.0, 5.0, 10.0], "max_hold_days": [1, 3, 5], "stop_pct": [0.05, 0.10]},
    "surge_pullback": {"pullback_pct": [0.05, 0.08, 0.15], "max_hold_days": [3, 5, 10], "stop_pct": [0.05, 0.10]},
    "gap_day": {"gap_pct": [0.03, 0.05, 0.10], "stop_pct": [0.02, 0.03, 0.05]},
}
EXIT_GRID = {"stop_pct": [0.05, 0.10, 0.20, None], "target_pct": [None, 0.15, 0.40], "max_hold_days": [1, 3, 5, 10]}
KEY_RET = {"next_open": "ret_3d", "close": "ret_on", "open": "ret_oc"}


class EntryStrategy(Strategy):
    """미리 계산한 entry 패널을 그대로 쓰는 경량 전략 (청산 파라미터만 바꿔가며 백테스트)."""

    name = "entry"
    defaults = {"max_hold_days": None}

    def __init__(self, label, signals, hold_mode, **params):
        super().__init__(**params)
        self._label, self._signals, self.hold_mode = label, signals, hold_mode

    @property
    def label(self):
        return self._label

    def generate(self, panels, bench):
        return self._signals


def cost_of(fr: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    slip = np.where(fr["entry_price"] < cfg.penny_price, cfg.penny_slippage_bps / 1e4, cfg.slippage_bps / 1e4)
    return pd.Series(2 * (slip + cfg.commission_pct), index=fr.index)


def window_mask(ev: pd.DataFrame, w) -> pd.DataFrame:
    m = (ev.index >= pd.Timestamp(w[0])) & (ev.index <= pd.Timestamp(w[1]))
    out = ev.copy()
    out.loc[~m, :] = False
    return out


def stage1(screen, fn, mode, stocks, feats, em, cfg) -> list[dict]:
    grid = SCREEN_GRIDS[screen]
    keys = list(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        entry_all = fn(stocks, feats, **params)
        for excl in ([False, True] if em is not None else [False]):
            entry = entry_all & ~em if excl else entry_all
            fr = forward_returns(stocks, window_mask(entry, TRAIN), mode)
            net = fr[KEY_RET[mode]] - cost_of(fr, cfg) if len(fr) else pd.Series(dtype=float)
            rows.append({"screen": screen, "params": json.dumps(params), "exclude_earnings": excl, "n": int(len(fr)),
                         "mean_net": float(net.mean()) if len(net) else float("nan"), "win": float((net > 0).mean()) if len(net) else float("nan")})
    return rows


def signals_for(entry, feats, mode, stocks, stop_pct, target_pct) -> Signals:
    c = stocks["close"]
    ref = stocks["open"] if mode == "open" else c
    adv = feats["adv20_prev"] if mode == "open" else feats["adv20"]
    return Signals(entry=entry, exit=nan_like(c).fillna(False).astype(bool), score=feats["vol_ratio"],
                   stop_dist=(stop_pct * ref) if stop_pct else None, target_pct=target_pct, adv=adv)


def run_both(strategy, signals, stocks, cfg_kw) -> dict:
    out = {}
    for wname, w in (("train", TRAIN), ("test", TEST)):
        cfg = BacktestConfig(start=w[0], end=w[1], **cfg_kw)
        res = run_backtest(stocks, signals, strategy, cfg)
        s = summarize(res.equity, res.trades, res.n_positions)
        for k in ("n_trades", "win_rate", "total_return", "max_drawdown", "profit_factor", "expectancy", "sharpe", "avg_hold_days"):
            out[f"{wname}_{k}"] = s[k]
    return out


def stage2_screen(screen, fn, mode, stocks, feats, em, cfg_kw, selected: list[dict]) -> list[dict]:
    rows = []
    hold_mode = {"next_open": "swing", "close": "overnight", "open": "intraday"}[mode]
    for sel in selected:
        params = json.loads(sel["params"])
        entry = fn(stocks, feats, **params)
        if sel["exclude_earnings"] and em is not None:
            entry = entry & ~em
        if mode == "close":
            exit_grid = [{"stop_pct": None, "target_pct": None, "max_hold_days": 1}]
        elif mode == "open":
            exit_grid = [{"stop_pct": s, "target_pct": t, "max_hold_days": 1} for s in EXIT_GRID["stop_pct"] for t in EXIT_GRID["target_pct"]]
        else:
            exit_grid = [dict(zip(EXIT_GRID, v)) for v in itertools.product(*EXIT_GRID.values())]
        for ex in exit_grid:
            sig = signals_for(entry, feats, mode, stocks, ex["stop_pct"], ex["target_pct"])
            strat = EntryStrategy(screen, sig, hold_mode, max_hold_days=ex["max_hold_days"])
            r = run_both(strat, sig, stocks, cfg_kw)
            r.update(model=screen, kind="screen", params=sel["params"], exclude_earnings=sel["exclude_earnings"], exit=json.dumps(ex))
            rows.append(r)
            logging.info("%s %s excl=%s %s | train n=%d win=%.1f%% exp=%.2f%% ret=%.1f%% | test n=%d win=%.1f%% exp=%.2f%% ret=%.1f%%",
                         screen, sel["params"], sel["exclude_earnings"], ex, r["train_n_trades"], r["train_win_rate"] * 100, r["train_expectancy"] * 100,
                         r["train_total_return"] * 100, r["test_n_trades"], r["test_win_rate"] * 100, r["test_expectancy"] * 100, r["test_total_return"] * 100)
    return rows


def run_strategy_model(name, stocks, bench, cfg_kw) -> list[dict]:
    grid = STRATEGY_GRIDS[name]
    keys = list(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        strat = make_strategy(name, **params)
        sig = strat.generate(stocks, bench)
        r = run_both(strat, sig, stocks, cfg_kw)
        r.update(model=name, kind="strategy", params=json.dumps(params), exclude_earnings=False, exit="{}")
        rows.append(r)
        logging.info("%s %s | train n=%d win=%.1f%% exp=%.2f%% ret=%.1f%% | test n=%d win=%.1f%% exp=%.2f%% ret=%.1f%%", name, params,
                     r["train_n_trades"], r["train_win_rate"] * 100, r["train_expectancy"] * 100, r["train_total_return"] * 100,
                     r["test_n_trades"], r["test_win_rate"] * 100, r["test_expectancy"] * 100, r["test_total_return"] * 100)
    return rows


def leaderboard(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in runs.groupby("model"):
        ok = g[g.train_n_trades >= MIN_TRADES]
        if ok.empty:
            ok = g
        best = ok.sort_values(["train_expectancy", "train_profit_factor"], ascending=False).iloc[0]
        oracle = g[g.test_n_trades >= MIN_TRADES]
        oracle_exp = oracle.test_expectancy.max() if not oracle.empty else float("nan")
        rows.append({"model": model, "kind": best["kind"], "params": best["params"], "exclude_earnings": best["exclude_earnings"], "exit": best["exit"],
                     **{c: best[c] for c in runs.columns if c.startswith(("train_", "test_"))},
                     "both_positive": bool(best.train_expectancy > 0 and best.test_expectancy > 0),
                     "n_configs": len(g), "oracle_test_expectancy": oracle_exp})
    return pd.DataFrame(rows).sort_values("test_expectancy", ascending=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", default="0/1", help="i/n")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--out", default="reports/surge_study/tournament")
    p.add_argument("--commission-pct", type=float, default=0.0025)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.merge:
        runs = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(out / "runs_*.csv")))], ignore_index=True)
        runs.to_csv(out / "runs_all.csv", index=False)
        lb = leaderboard(runs)
        lb.to_csv(out / "leaderboard.csv", index=False)
        s1 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(out / "stage1_*.csv")))], ignore_index=True) if glob.glob(str(out / "stage1_*.csv")) else pd.DataFrame()
        if not s1.empty:
            s1.to_csv(out / "stage1_all.csv", index=False)
        lines = [f"# 모델 토너먼트 리더보드 (학습 {TRAIN[0]}~{TRAIN[1]} 에서 선택 → 검증 {TEST[0]}~{TEST[1]})\n",
                 f"선택 기준: 학습 구간 거래 ≥ {MIN_TRADES}건 중 기대수익/거래(모든 비용 차감) 최고. 비용: 수수료 편도 {args.commission_pct * 100:.2f}% + 슬리피지. "
                 "'oracle' = 검증 구간에서 사후적으로 가장 좋았던 조합(과최적화 참고용, 실제로는 고를 수 없음).\n",
                 "| 순위 | 모델 | 선택된 파라미터 | 청산 | 재료제외 | 학습 거래/승률/기대/수익률/MDD/PF | 검증 거래/승률/기대/수익률/MDD/PF | 둘 다 + | oracle 검증 기대 |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(lb.itertuples(), 1):
            lines.append(f"| {i} | {r.model} | `{r.params}` | `{r.exit}` | {'예' if r.exclude_earnings else '아니오'} | "
                         f"{int(r.train_n_trades)} / {r.train_win_rate * 100:.1f}% / {r.train_expectancy * 100:+.2f}% / {r.train_total_return * 100:+.1f}% / {r.train_max_drawdown * 100:.1f}% / {r.train_profit_factor:.2f} | "
                         f"{int(r.test_n_trades)} / {r.test_win_rate * 100:.1f}% / {r.test_expectancy * 100:+.2f}% / {r.test_total_return * 100:+.1f}% / {r.test_max_drawdown * 100:.1f}% / {r.test_profit_factor:.2f} | "
                         f"{'**예**' if r.both_positive else '아니오'} | {r.oracle_test_expectancy * 100:+.2f}% |")
        n_both = int(lb.both_positive.sum())
        lines.append(f"\n총 {len(lb)}개 모델, {len(runs)}개 설정. 학습에서 고른 설정이 검증에서도 기대수익 > 0 인 모델: **{n_both}개**. "
                     f"검증 최고 승률(선택된 설정 기준): {lb.test_win_rate.max() * 100:.1f}%.\n")
        (out / "leaderboard.md").write_text("\n".join(lines), encoding="utf-8")
        logging.info("-> %s", out / "leaderboard.md")
        return

    i, n = (int(x) for x in args.shard.split("/"))
    models = sorted(SCREEN_GRIDS) + sorted(STRATEGY_GRIDS)
    mine = [m for k, m in enumerate(models) if k % n == i]
    logging.info("shard %d/%d: %s", i, n, mine)
    uni = data.load_universe("all")
    stocks, bench = data.split_benchmark(data.load_panels(universe=uni))
    feats = compute_features(stocks)
    em = earnings_mask(stocks)
    cfg_kw = dict(commission_pct=args.commission_pct)
    cfg = BacktestConfig(**cfg_kw)
    runs, s1_rows = [], []
    for m in mine:
        t0 = time.time()
        if m in SCREEN_GRIDS:
            fn, mode = SCREENS[m]
            s1 = stage1(m, fn, mode, stocks, feats, em, cfg)
            s1_rows.extend(s1)
            s1df = pd.DataFrame(s1)
            ok = s1df[s1df.n >= MIN_EVENTS].sort_values("mean_net", ascending=False)
            selected = (ok if not ok.empty else s1df.sort_values("n", ascending=False)).head(TOP_K).to_dict("records")
            logging.info("%s stage1: %d configs, selected %s", m, len(s1df), [(s["params"], s["exclude_earnings"], round(s["mean_net"] * 100, 2)) for s in selected])
            runs.extend(stage2_screen(m, fn, mode, stocks, feats, em, cfg_kw, selected))
        else:
            runs.extend(run_strategy_model(m, stocks, bench, cfg_kw))
        pd.DataFrame(runs).to_csv(out / f"runs_{i}.csv", index=False)
        if s1_rows:
            pd.DataFrame(s1_rows).to_csv(out / f"stage1_{i}.csv", index=False)
        logging.info("%s done in %.0fs", m, time.time() - t0)
    logging.info("shard %d done: %d runs", i, len(runs))


if __name__ == "__main__":
    main()
