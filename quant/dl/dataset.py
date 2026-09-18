"""1시간봉 + 일봉 맥락으로 '결정 시점' 샘플을 만든다.

결정 시점 두 가지 (문헌 차용):
  "h1"  : 첫 1시간봉 마감(10:30) 직후 — Gao et al.(2018) 장중 모멘텀(첫 구간 수익률이 마지막 구간을 예측), Zarattini et al.(2024) '거래량 터진 종목(stocks in play)'
  "open": 시가 — Fischer & Krauss(2018)/Ghosh et al.(2021) 의 '시가 매수·종가 매도' 횡단면 랭킹

샘플 = (symbol, 거래일). 입력:
  seq  : 최근 L 개 1시간봉 [로그수익률, 봉 범위/종가, 거래량 z(20일 평균 시간봉 거래량 대비), 시간대 인덱스] — 결정 시점 이전 봉만
  ctx  : 일봉 맥락(갭, 첫1시간 수익률·상대거래량, 전일 수익률, 5/20/60일 수익률, ATR%, 52주고가 대비, log 거래대금, log 가격, 시총 구간, SPY 첫1시간·전일 수익률, SPY 200일선 위 여부)
라벨:
  y_rod : 결정 시점 → 당일 종가 수익률 (데이트레이딩)      y_on : 당일 종가 → 다음날 시가 (오버나잇, 참고)
  y_nd  : 결정 시점 → 다음날 같은 시점 (1일 보유, 참고)
모든 특징은 결정 시점까지의 정보만 사용한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import CACHE_DIR, load_panels, load_universe, split_benchmark
from ..intraday import build_day_arrays

log = logging.getLogger(__name__)
SEQ_FEATS = ["logret", "range", "volz", "tod"]
CTX_FEATS = ["gap", "h1_ret", "h1_rvol", "prev_ret", "ret5", "ret20", "ret60", "atr_pct", "from_high52", "log_adv", "log_price", "log_mcap",
             "spy_h1", "spy_prev", "spy_above200", "vol_ratio_prev", "consec_up"]


@dataclass
class Dataset:
    key: pd.DataFrame  # symbol, date, + labels + ctx columns
    seq: np.ndarray  # (n, L, len(SEQ_FEATS)) float32
    ctx: np.ndarray  # (n, len(CTX_FEATS)) float32
    decision: str


def _daily_context(stocks: dict, bench: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from ..indicators import atr, rolling_max, sma

    c, h, l, v, o = stocks["close"], stocks["high"], stocks["low"], stocks["volume"], stocks["open"]
    f = {}
    f["prev_ret"] = c / c.shift(1) - 1.0
    f["ret5"], f["ret20"], f["ret60"] = c / c.shift(5) - 1.0, c / c.shift(20) - 1.0, c / c.shift(60) - 1.0
    f["atr_pct"] = atr(h, l, c, 14) / c
    f["from_high52"] = c / rolling_max(h, 252) - 1.0
    f["log_adv"] = np.log((c * v).rolling(20, min_periods=20).mean().clip(lower=1.0))
    f["log_price"] = np.log(c.clip(lower=0.01))
    f["vol_ratio_prev"] = v / sma(v, 20).shift(1)
    up = (c > c.shift(1)).to_numpy()
    consec = np.zeros(up.shape)
    for t in range(1, up.shape[0]):
        consec[t] = np.where(up[t], consec[t - 1] + 1.0, 0.0)
    f["consec_up"] = pd.DataFrame(consec, index=c.index, columns=c.columns)
    f["hourly_vol_avg"] = sma(v, 20) / 6.5  # 20일 평균 '시간당' 거래량
    f["h1_vol_avg14"] = None  # 아래 intraday 에서 계산
    if "mcap" in stocks:
        f["log_mcap"] = np.log(stocks["mcap"].clip(lower=1e5))
    else:
        f["log_mcap"] = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    spy = bench["close"]
    f["spy_prev"] = (spy / spy.shift(1) - 1.0)
    f["spy_above200"] = (spy > spy.rolling(200).mean()).astype(float)
    return f


def build_dataset(decision: str = "h1", seq_len: int = 35, min_adv: float = 2e6, min_price: float = 2.0, universe: str = "all",
                  start: str | None = None, max_symbols: int | None = None) -> Dataset:
    assert decision in ("h1", "open")
    uni = load_universe(universe)
    panels = load_panels(universe=uni)
    stocks, bench = split_benchmark(panels)
    ctxp = _daily_context(stocks, bench)
    bars = pd.read_parquet(CACHE_DIR / "intraday_1h.parquet")
    if max_symbols:
        keep = sorted(set(bars["symbol"].unique()))[:max_symbols]
        bars = bars[bars["symbol"].isin(keep)]
    key, arr = build_day_arrays(bars, "1h")
    key["date"] = pd.to_datetime(key["date"])
    if start:
        pass  # 시퀀스 warm-up 이 필요하므로 전체를 만들고 마지막에 자른다
    o, h, l, c, v = (arr[k] for k in ("open", "high", "low", "close", "volume"))
    m = c.shape[1]  # 7 slots
    # SPY 봉
    spy_rows = key[key["symbol"] == "SPY"].set_index("date")["row"]
    # ---- 봉 단위 시퀀스 특징 (심볼별 연속 시간축)
    key = key.sort_values(["symbol", "date"]).reset_index(drop=True)
    rows = key["row"].to_numpy()
    o, h, l, c, v = o[rows], h[rows], l[rows], c[rows], v[rows]
    key["row"] = np.arange(len(key))
    n = len(key)
    flat_c = c.reshape(-1)
    flat_prev = np.roll(flat_c, 1)
    sym_start = np.r_[True, key["symbol"].to_numpy()[1:] != key["symbol"].to_numpy()[:-1]]
    flat_prev = flat_prev.copy()
    # 심볼 첫날 첫 봉의 '전 봉' 은 자기 자신
    first_idx = np.flatnonzero(sym_start) * m
    flat_prev[first_idx] = flat_c[first_idx]
    logret = np.clip(np.log(np.clip(flat_c, 1e-6, None) / np.clip(flat_prev, 1e-6, None)).reshape(n, m), -0.5, 0.5)
    rng = np.clip((h - l) / np.clip(c, 1e-6, None), 0.0, 1.0)
    # 거래량 z: 봉 거래량 / 20일 평균 시간당 거래량 (일봉에서, 전일 값)
    sym_idx = stocks["close"].columns.get_indexer(key["symbol"])
    date_idx = stocks["close"].index.get_indexer(key["date"])
    ok_daily = (sym_idx >= 0) & (date_idx >= 1)
    hv = np.full(n, np.nan)
    hv[ok_daily] = ctxp["hourly_vol_avg"].to_numpy()[date_idx[ok_daily] - 1, sym_idx[ok_daily]]
    volz = np.clip(np.log1p(v / np.clip(hv[:, None], 1.0, None)), 0.0, 8.0)
    tod = np.tile(np.arange(m, dtype=float) / (m - 1), (n, 1))
    seq_all = np.stack([logret, rng, volz, tod], axis=-1).astype(np.float32)  # (n, m, 4)
    flat = seq_all.reshape(n * m, len(SEQ_FEATS))
    # ---- 결정 시점 인덱스: h1 -> 첫 봉(0) 까지 포함, open -> 전일 마지막 봉까지
    k_dec = 0 if decision == "h1" else -1
    end_flat = key["row"].to_numpy() * m + k_dec  # 포함되는 마지막 봉의 flat index
    # 시퀀스: [end_flat - L + 1, end_flat]
    idx_mat = end_flat[:, None] + np.arange(-seq_len + 1, 1)[None, :]
    valid = (idx_mat[:, 0] >= 0)
    # 같은 심볼 안에서만 (심볼 시작 flat index 이상)
    sym_first_flat = np.repeat(first_idx, np.diff(np.r_[np.flatnonzero(sym_start), n]))
    valid &= idx_mat[:, 0] >= sym_first_flat
    idx_mat = np.clip(idx_mat, 0, n * m - 1)
    seq = flat[idx_mat]  # (n, L, 4)
    # ---- 라벨 & 맥락
    dec_px = c[:, 0] if decision == "h1" else o[:, 0]
    y_rod = c[:, -1] / dec_px - 1.0
    next_same = np.roll(dec_px, -1)
    next_open = np.roll(o[:, 0], -1)
    last_of_sym = np.r_[key["symbol"].to_numpy()[1:] != key["symbol"].to_numpy()[:-1], True]
    y_nd = np.where(last_of_sym, np.nan, next_same / dec_px - 1.0)
    y_on = np.where(last_of_sym, np.nan, next_open / c[:, -1] - 1.0)
    prev_close = np.full(n, np.nan)
    prev_close[ok_daily] = stocks["close"].to_numpy()[date_idx[ok_daily] - 1, sym_idx[ok_daily]]
    # 다일 보유 라벨: 결정가 -> t+2 / t+4 거래일 마지막 봉 종가 (같은 시간봉 기준, 같은 심볼 안에서만)
    sym_arr = key["symbol"].to_numpy()
    y_3d, y_5d = np.full(n, np.nan), np.full(n, np.nan)
    for arr_out, off in ((y_3d, 2), (y_5d, 4)):
        fut = np.roll(c[:, -1], -off)
        same = np.roll(sym_arr, -off) == sym_arr
        same[-off:] = False
        arr_out[same] = fut[same] / dec_px[same] - 1.0
    ctx = pd.DataFrame(index=key.index)
    # 갭은 같은 기준(시간봉)의 전일 마지막 봉 종가로 계산 — 시간봉은 배당 미조정이라 일봉 조정종가와 섞으면 배당주가 왜곡된다
    prev_close_h = np.roll(c[:, -1], 1)
    prev_close_h[sym_start] = np.nan
    ctx["gap"] = np.clip(o[:, 0] / prev_close_h - 1.0, -0.5, 0.5)
    if decision == "h1":
        ctx["h1_ret"] = np.clip(c[:, 0] / o[:, 0] - 1.0, -0.5, 0.5)
        # 첫 1시간 거래량 / 직전 14일 첫 1시간 평균 거래량 (stocks in play)
        h1v = pd.Series(v[:, 0])
        g = h1v.groupby(key["symbol"].to_numpy())
        h1_avg = g.transform(lambda s: s.shift(1).rolling(14, min_periods=5).mean())
        ctx["h1_rvol"] = np.log1p(h1v / h1_avg.clip(lower=1.0))
    else:
        ctx["h1_ret"] = 0.0
        ctx["h1_rvol"] = 0.0
    clipping = {"prev_ret": 0.5, "ret5": 1.0, "ret20": 2.0, "ret60": 3.0, "atr_pct": 0.5, "from_high52": 1.0, "vol_ratio_prev": 50.0, "consec_up": 20.0}
    for name in ("prev_ret", "ret5", "ret20", "ret60", "atr_pct", "from_high52", "log_adv", "log_price", "log_mcap", "vol_ratio_prev", "consec_up"):
        col = np.full(n, np.nan)
        col[ok_daily] = ctxp[name].to_numpy()[date_idx[ok_daily] - 1, sym_idx[ok_daily]]
        if name in clipping:
            col = np.clip(col, -clipping[name], clipping[name])
        ctx[name] = col
    spy_date_idx = bench.index.get_indexer(key["date"])
    okb = spy_date_idx >= 1
    sp = np.full(n, np.nan)
    sp[okb] = ctxp["spy_prev"].to_numpy()[spy_date_idx[okb] - 1]
    ctx["spy_prev"] = sp
    sa = np.full(n, np.nan)
    sa[okb] = ctxp["spy_above200"].to_numpy()[spy_date_idx[okb] - 1]
    ctx["spy_above200"] = sa
    spy_h1 = np.full(n, np.nan)
    if decision == "h1" and len(spy_rows):
        srow = spy_rows.reindex(key["date"]).to_numpy()
        oks = ~np.isnan(srow)
        # SPY 행은 정렬 전 row 기준이므로 원본 배열에서 다시 읽는다
        spy_c0 = arr["close"][srow[oks].astype(int), 0]
        spy_o0 = arr["open"][srow[oks].astype(int), 0]
        spy_h1[oks] = spy_c0 / spy_o0 - 1.0
    ctx["spy_h1"] = np.nan_to_num(spy_h1, nan=0.0) if decision == "h1" else 0.0
    # 라벨 클리핑: ±50% 밖은 데이터 오류(역분할/오류 틱)로 보고 잘라낸다
    key["y_rod"], key["y_nd"], key["y_on"] = np.clip(y_rod, -0.5, 0.5), np.clip(y_nd, -0.5, 0.5), np.clip(y_on, -0.5, 0.5)
    key["y_3d"], key["y_5d"] = np.clip(y_3d, -0.8, 1.0), np.clip(y_5d, -0.8, 1.0)
    key["dec_px"] = dec_px
    key["adv"] = np.exp(ctx["log_adv"])
    for cname in CTX_FEATS:
        key[cname] = ctx[cname].to_numpy()
    # ---- 필터: 유동성/가격/유효성/SPY 제외
    good = valid & np.isfinite(y_rod) & (key["adv"].to_numpy() >= min_adv) & (prev_close >= min_price) & ok_daily
    good &= ~key["symbol"].isin(["SPY", "QQQ", "IWM"]).to_numpy()
    good &= np.isfinite(seq).all(axis=(1, 2))
    if start:
        good &= key["date"].to_numpy() >= np.datetime64(start)
    key = key[good].reset_index(drop=True)
    seq = seq[good]
    ctxm = np.nan_to_num(key[CTX_FEATS].to_numpy(dtype=np.float32), nan=0.0)
    log.info("dataset %s: %d samples, %d symbols, %s ~ %s", decision, len(key), key["symbol"].nunique(), key["date"].min().date(), key["date"].max().date())
    return Dataset(key=key.drop(columns=["row"]), seq=seq.astype(np.float32), ctx=ctxm, decision=decision)
