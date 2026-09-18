"""모델: GRU(시퀀스+맥락) / LightGBM(요약 특징) / 문헌 규칙 베이스라인. 목표값 = 일자별 횡단면 순위(0~1)-0.5.

문헌 차용:
  - Fischer & Krauss(2018), Ghosh et al.(2021): 횡단면 상위/하위 분위 예측 → 상위 K 매수. LSTM 25 units, 시퀀스 240 (여기선 GRU 64, 시퀀스 35 시간봉)
  - Kaggle(Jane Street/Optiver) 상위권: LightGBM + 롤링 통계 특징 + 시간 기반 검증, NN 과 앙상블
  - López de Prado: 시간 분할 + 엠바고(라벨 겹침 방지)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def cs_rank_target(key: pd.DataFrame, col: str = "y_rod") -> np.ndarray:
    """일자별 횡단면 백분위 순위(0~1) - 0.5."""
    r = key.groupby("date")[col].rank(pct=True)
    return (r.to_numpy() - 0.5).astype(np.float32)


def seq_summary(seq: np.ndarray) -> np.ndarray:
    """LightGBM 용 시퀀스 요약: 마지막 7봉 로그수익률, 7/35봉 누적수익률, 범위·거래량 평균, 마지막 봉 거래량 z, 변동성."""
    lr, rng, volz = seq[:, :, 0], seq[:, :, 1], seq[:, :, 2]
    feats = [lr[:, -7:], lr[:, -7:].sum(1, keepdims=True), lr.sum(1, keepdims=True), rng[:, -7:].mean(1, keepdims=True), rng.mean(1, keepdims=True),
             volz[:, -7:].mean(1, keepdims=True), volz[:, -1:], lr.std(1, keepdims=True), lr[:, -7:].std(1, keepdims=True),
             (lr[:, -14:-7]).sum(1, keepdims=True), (lr[:, -21:-14]).sum(1, keepdims=True)]
    return np.concatenate(feats, axis=1).astype(np.float32)


SEQ_SUMMARY_NAMES = [f"lr_m{i}" for i in range(7, 0, -1)] + ["lr_sum7", "lr_sum35", "rng_mean7", "rng_mean35", "volz_mean7", "volz_last", "lr_std35", "lr_std7", "lr_sum_14_7", "lr_sum_21_14"]


# ----------------------------------------------------------------------------- LightGBM
def train_lgbm(X_tr, y_tr, X_va, y_va, feature_names, seed=0):
    import lightgbm as lgb

    params = dict(objective="regression", learning_rate=0.02, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed, num_threads=4)
    dtr = lgb.Dataset(X_tr, y_tr, feature_name=feature_names)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
    log.info("lgbm best_iteration=%d", model.best_iteration)
    return model


# ----------------------------------------------------------------------------- GRU (PyTorch)
def train_gru(seq_tr, ctx_tr, y_tr, seq_va, ctx_va, y_va, key_va, hidden=64, epochs=8, batch=4096, lr=1e-3, seed=0, patience=2):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    seq_mu, seq_sd = seq_tr.reshape(-1, seq_tr.shape[-1]).mean(0), seq_tr.reshape(-1, seq_tr.shape[-1]).std(0) + 1e-6
    ctx_mu, ctx_sd = ctx_tr.mean(0), ctx_tr.std(0) + 1e-6

    class Net(nn.Module):
        def __init__(self, nf, nc):
            super().__init__()
            self.gru = nn.GRU(nf, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + nc, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

        def forward(self, s, c):
            _, hN = self.gru(s)
            return self.head(torch.cat([hN[-1], c], dim=1)).squeeze(-1)

    def prep(seq, ctx):
        return (torch.tensor((seq - seq_mu) / seq_sd, dtype=torch.float32), torch.tensor((ctx - ctx_mu) / ctx_sd, dtype=torch.float32))

    S_tr, C_tr = prep(seq_tr, ctx_tr)
    S_va, C_va = prep(seq_va, ctx_va)
    Y_tr = torch.tensor(y_tr, dtype=torch.float32)
    net = Net(seq_tr.shape[-1], ctx_tr.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.MSELoss()
    n = len(Y_tr)
    best_ic, best_state, bad = -1e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            out = net(S_tr[idx], C_tr[idx])
            loss = lossf(out, Y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        pred_va = predict_gru((net, seq_mu, seq_sd, ctx_mu, ctx_sd), seq_va, ctx_va)
        ic = daily_ic(key_va, pred_va, y_va)["ic_mean"]
        log.info("gru epoch %d train_mse=%.5f val_ic=%.4f", ep + 1, tot / n, ic)
        if ic > best_ic:
            best_ic, bad = ic, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return (net, seq_mu, seq_sd, ctx_mu, ctx_sd)


def predict_gru(model, seq, ctx, batch=16384) -> np.ndarray:
    import torch

    net, seq_mu, seq_sd, ctx_mu, ctx_sd = model
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seq), batch):
            s = torch.tensor((seq[i : i + batch] - seq_mu) / seq_sd, dtype=torch.float32)
            c = torch.tensor((ctx[i : i + batch] - ctx_mu) / ctx_sd, dtype=torch.float32)
            out.append(net(s, c).numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- 평가
def daily_ic(key: pd.DataFrame, pred: np.ndarray, target: np.ndarray) -> dict:
    df = pd.DataFrame({"date": key["date"].to_numpy(), "p": pred, "y": target})
    ics = df.groupby("date").apply(lambda g: g["p"].corr(g["y"], method="spearman") if len(g) >= 20 else np.nan).dropna()
    return {"ic_mean": float(ics.mean()), "ic_std": float(ics.std()), "ic_t": float(ics.mean() / ics.std() * np.sqrt(len(ics))) if ics.std() > 0 else float("nan"),
            "ic_pos": float((ics > 0).mean()), "n_days": int(len(ics))}


def topk_portfolio(key: pd.DataFrame, pred: np.ndarray, ret_col: str, k: int, cost_rt: np.ndarray) -> dict:
    """매일 예측 상위 k 종목 동일비중 매수 (결정가 진입 → ret_col 청산). 비용후 일별 수익률 시계열과 통계."""
    df = pd.DataFrame({"date": key["date"].to_numpy(), "symbol": key["symbol"].to_numpy(), "p": pred, "y": key[ret_col].to_numpy(), "cost": cost_rt})
    df = df.dropna(subset=["y"])
    top = df.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(k)
    top["net"] = top["y"] - top["cost"]
    daily = top.groupby("date").agg(gross=("y", "mean"), net=("net", "mean"), n=("y", "size"))
    trades = top
    eq = (1 + daily["net"]).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {"daily": daily, "trades": trades, "gross_mean": float(trades["y"].mean()), "net_mean": float(trades["net"].mean()), "win": float((trades["net"] > 0).mean()),
            "n_trades": int(len(trades)), "total_return": float(eq.iloc[-1] - 1), "sharpe": float(daily["net"].mean() / daily["net"].std() * np.sqrt(252)) if daily["net"].std() > 0 else float("nan"),
            "max_dd": float(dd), "days": int(len(daily))}


def decile_spread(key: pd.DataFrame, pred: np.ndarray, ret_col: str) -> dict:
    df = pd.DataFrame({"date": key["date"].to_numpy(), "p": pred, "y": key[ret_col].to_numpy()}).dropna()
    df["dec"] = df.groupby("date")["p"].rank(pct=True).mul(10).clip(upper=9.999).astype(int)
    m = df.groupby("dec")["y"].mean()
    return {"top": float(m.get(9, np.nan)), "bottom": float(m.get(0, np.nan)), "spread": float(m.get(9, np.nan) - m.get(0, np.nan)), "by_decile": m}
