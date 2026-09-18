"""20거래일 보유용 ML 랭커: 일봉 특징 → 20일 선행 수익률의 횡단면 순위를 LightGBM 으로 예측, 반기마다 확장 재학습(walk-forward).

  python scripts/ml_rank_train.py            -> data/cache/ml_rank_pred.parquet (date x symbol 예측 패널)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import data  # noqa: E402
from quant.indicators import rsi, sma  # noqa: E402
from quant.screens import compute_features  # noqa: E402

HORIZON = 20
FEATS = ["ret5", "ret20", "runup30", "days_from_high52", "atr_pct", "rsi14", "vol_ratio", "consec_up", "range40_prev", "turnover", "close_pos"]
EXTRA = ["ret60", "ret120", "ret252_21", "vol20", "vol60", "log_adv", "log_mcap", "log_price", "above_sma50", "above_sma200", "spy_ret20", "spy_above200", "iwm_ret20", "dv_ratio"]
REFITS = [("2024-07-01", "2024-12-31"), ("2025-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31"), ("2026-01-01", "2026-12-31")]


def build(stocks, bench):
    c, v = stocks["close"], stocks["volume"]
    f = compute_features(stocks)
    X = {k: f[k] for k in FEATS}
    X["ret60"] = c / c.shift(60) - 1
    X["ret120"] = c / c.shift(120) - 1
    X["ret252_21"] = c.shift(21) / c.shift(252) - 1
    lr = np.log(c / c.shift(1))
    X["vol20"], X["vol60"] = lr.rolling(20).std(), lr.rolling(60).std()
    X["log_adv"] = np.log(f["adv20"].clip(lower=1))
    X["log_mcap"] = np.log(stocks["mcap"].clip(lower=1e5)) if "mcap" in stocks else f["adv20"] * np.nan
    X["log_price"] = np.log(c.clip(lower=0.01))
    X["above_sma50"] = (c > sma(c, 50)).astype(float)
    X["above_sma200"] = (c > sma(c, 200)).astype(float)
    X["dv_ratio"] = f["dv"] / f["adv20_prev"]
    spy = bench["close"]
    idx = pd.read_parquet(data.CACHE_DIR / "index_close.parquet")
    idx.index = pd.to_datetime(idx.index)
    iwm = idx["IWM"].reindex(c.index).ffill()
    spy_ret20 = (spy / spy.shift(20) - 1).reindex(c.index)
    spy_ab = (spy > spy.rolling(200).mean()).astype(float).reindex(c.index)
    iwm_ret20 = iwm / iwm.shift(20) - 1
    for name, s in (("spy_ret20", spy_ret20), ("spy_above200", spy_ab), ("iwm_ret20", iwm_ret20)):
        X[name] = pd.DataFrame(np.repeat(s.to_numpy()[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
    y = c.shift(-HORIZON) / c - 1.0
    liq = (f["adv20"] >= 5e6) & (c >= 5.0)
    names = FEATS + EXTRA
    long = {k: X[k].where(liq).stack(future_stack=True) for k in names}
    df = pd.DataFrame(long)
    df["y"] = y.where(liq).stack(future_stack=True)
    df = df.dropna(subset=names, thresh=len(names) - 3)
    df.index.names = ["date", "symbol"]
    df = df.reset_index()
    df["y_rank"] = df.groupby("date")["y"].rank(pct=True) - 0.5
    return df, names


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import lightgbm as lgb

    uni = data.load_universe("all")
    stocks, bench = data.split_benchmark(data.load_panels(universe=uni))
    df, names = build(stocks, bench)
    logging.info("rows %d, dates %d, symbols %d", len(df), df["date"].nunique(), df["symbol"].nunique())
    preds = []
    params = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, num_threads=4)
    for start, end in REFITS:
        cutoff = pd.Timestamp(start) - pd.Timedelta(days=HORIZON * 2)  # 라벨 겹침 방지 엠바고
        tr = df[(df["date"] <= cutoff) & df["y_rank"].notna()]
        te = df[(df["date"] >= start) & (df["date"] <= end)]
        if te.empty:
            continue
        model = lgb.train(params, lgb.Dataset(tr[names], tr["y_rank"]), num_boost_round=400)
        p = model.predict(te[names])
        preds.append(pd.DataFrame({"date": te["date"].to_numpy(), "symbol": te["symbol"].to_numpy(), "pred": p}))
        ok = te["y_rank"].notna()
        ic = pd.DataFrame({"date": te["date"][ok], "p": p[ok.to_numpy()], "y": te["y_rank"][ok]}).groupby("date").apply(lambda g: g["p"].corr(g["y"], method="spearman"))
        logging.info("refit %s~%s: train %d rows (<= %s), IC mean %.4f, IC>0 %.0f%%", start, end, len(tr), cutoff.date(), ic.mean(), (ic > 0).mean() * 100)
        imp = pd.Series(model.feature_importance("gain"), index=names).sort_values(ascending=False)
        logging.info("top features: %s", ", ".join(f"{k}={v:.0f}" for k, v in imp.head(6).items()))
    out = pd.concat(preds).pivot(index="date", columns="symbol", values="pred")
    out.to_parquet(data.CACHE_DIR / "ml_rank_pred.parquet")
    logging.info("saved prediction panel %s -> %s", out.shape, data.CACHE_DIR / "ml_rank_pred.parquet")


if __name__ == "__main__":
    main()
