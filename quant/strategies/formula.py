"""카탈로그의 스크린(quant/screens.py)을 그대로 포트폴리오 전략으로 감싼다.

  make_strategy("formula", screen="ross_5pillars", screen_params={"min_ret": 0.15}, max_hold_days=3, stop_pct=0.1)

hold_mode 는 스크린의 진입 방식에 따라 자동 결정: next_open -> swing, close -> overnight, open -> intraday.
청산: max_hold_days(시간), stop_pct(손절), target_pct(익절), exit_rule("rsi2_70": RSI(2)>70 / "sma5": 종가>SMA5 / "below_sma5": 종가<SMA5 / None).
"""
from __future__ import annotations

from ..screens import SCREENS, compute_features, earnings_mask
from .base import Signals, Strategy, nan_like


class FormulaStrategy(Strategy):
    name = "formula"
    defaults = dict(
        screen="ross_5pillars",
        screen_params={},
        max_hold_days=3,
        stop_pct=0.10,
        target_pct=None,
        exit_rule=None,
        rank_by="vol_ratio",  # feats 의 컬럼명. 클수록 우선
        direction=1,  # -1 이면 공매도 (급등 다음날 시가 공매도 -> N일 뒤 환매)
        entry_mode=None,  # None=스크린 기본 / "next_open"(다음날 시가) / "close"(당일 종가 진입·다음날 시가 청산) / "open"
        exclude_earnings=False,  # True 면 실적발표일(±1거래일) 신호 제외 = '이유 없는 급등' 만
    )

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["screen"] not in SCREENS:
            raise KeyError(f"unknown screen {self.params['screen']!r}; choose from {sorted(SCREENS)}")
        mode = self.params["entry_mode"] or SCREENS[self.params["screen"]][1]
        self.hold_mode = {"next_open": "swing", "close": "overnight", "open": "intraday"}[mode]
        self.direction = -1 if self.params["direction"] == -1 else 1

    @property
    def label(self) -> str:
        return f"formula_{self.params['screen']}" + ("_short" if self.direction == -1 else "") + (f"_{self.params['entry_mode']}" if self.params["entry_mode"] else "")

    def generate(self, panels, bench) -> Signals:
        p = self.params
        fn, mode = SCREENS[p["screen"]]
        mode = p["entry_mode"] or mode
        feats = compute_features(panels)
        entry = fn(panels, feats, **p["screen_params"])
        if p["exclude_earnings"]:
            em = earnings_mask(panels)
            if em is not None:
                entry = entry & ~em
        c = panels["close"]
        rule = p["exit_rule"]
        if rule == "rsi2_70":
            exit_ = feats["rsi2"] > 70
        elif rule == "sma5":
            exit_ = c > feats["sma5"]
        elif rule == "below_sma5":
            exit_ = c < feats["sma5"]
        elif rule == "toss":  # 음봉 전환 또는 거래량이 전일의 2배 (토스 글 4-(3) 보유 규칙)
            exit_ = (feats["ret1"] < 0) | (panels["volume"] >= 2.0 * panels["volume"].shift(1))
        else:
            exit_ = nan_like(c).fillna(False)
        score = feats[p["rank_by"]] if p["rank_by"] in feats else nan_like(c)
        ref = panels["open"] if mode == "open" else c
        stop_dist = p["stop_pct"] * ref if p["stop_pct"] else None
        adv = feats["adv20_prev"] if mode == "open" else feats["adv20"]
        return Signals(entry=entry.fillna(False).astype(bool), exit=exit_.fillna(False).astype(bool), score=score,
                       stop_dist=stop_dist, target_pct=p["target_pct"], adv=adv)
