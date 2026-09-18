"""브로커 어댑터: DryRun(주문을 내지 않고 계획만 기록) / KIS(한국투자증권 Open API, 해외주식).

국내 증권사 해외주식 API 기준이라 **공매도 없음(매수/매도만)**, 수수료는 정률(기본 0.25%).

KIS 사용 시 환경변수
  KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT(종합계좌 8자리, CANO), KIS_ACCOUNT_PRDT(상품코드, 기본 "01"),
  KIS_PAPER=1 이면 모의투자 도메인(openapivts) 사용.
주의: tr_id 와 필드명은 KIS Developers 문서(해외주식 주문/잔고) 기준으로 작성했으나 API 는 바뀔 수 있으므로
      실제 사용 전 모의투자에서 반드시 확인할 것. 미국 주식은 지정가(ORD_DVSN "00")만 지원한다.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

EXCHANGE_CODES = {"NASDAQ": "NASD", "NYSE": "NYSE", "AMEX": "AMEX"}


@dataclass
class Order:
    symbol: str
    qty: int
    side: str  # "buy" | "sell"
    limit_price: float  # KIS 해외주식은 지정가만 -> 시장가 흉내는 현재가에 버퍼를 준 지정가
    exchange: str = "NASD"
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Broker:
    name = "base"

    def account(self) -> dict:  # {"equity": float, "cash": float}
        raise NotImplementedError

    def positions(self) -> dict[str, dict]:  # {symbol: {"qty": int, "avg_price": float}}
        raise NotImplementedError

    def submit(self, order: Order) -> dict:
        raise NotImplementedError


class DryRunBroker(Broker):
    """주문을 실제로 내지 않고 로그로만 남긴다. 포지션/현금은 state 파일(quant.live)에서 관리."""

    name = "dryrun"

    def __init__(self, state_file: Path, equity: float = 100_000.0):
        self.state_file = Path(state_file)
        self.equity = equity
        self.submitted: list[Order] = []

    def _state(self) -> dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {"positions": {}, "cash": self.equity, "equity": self.equity}

    def account(self) -> dict:
        st = self._state()
        return {"equity": st.get("equity", self.equity), "cash": st.get("cash", self.equity)}

    def positions(self) -> dict[str, dict]:
        return {s: {"qty": v["qty"], "avg_price": v.get("entry_price") or 0.0} for s, v in self._state()["positions"].items()}

    def submit(self, order: Order) -> dict:
        self.submitted.append(order)
        log.info("[DRY-RUN] %s %d %s @ %.2f (%s)", order.side.upper(), order.qty, order.symbol, order.limit_price, order.reason)
        return {"status": "dry-run", **order.to_dict()}


class KISBroker(Broker):
    """한국투자증권 Open API 해외주식(미국) 어댑터."""

    name = "kis"

    def __init__(self):
        self.app_key = os.environ.get("KIS_APP_KEY")
        self.app_secret = os.environ.get("KIS_APP_SECRET")
        self.cano = os.environ.get("KIS_ACCOUNT")
        self.prdt = os.environ.get("KIS_ACCOUNT_PRDT", "01")
        self.paper = os.environ.get("KIS_PAPER", "1") == "1"
        if not all([self.app_key, self.app_secret, self.cano]):
            raise RuntimeError("KIS_APP_KEY / KIS_APP_SECRET / KIS_ACCOUNT 환경변수가 필요합니다")
        self.base = "https://openapivts.koreainvestment.com:29443" if self.paper else "https://openapi.koreainvestment.com:9443"
        self.session = requests.Session()
        self._token: str | None = None
        self._token_exp = 0.0

    # -- auth
    def _auth(self) -> str:
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        r = self.session.post(f"{self.base}/oauth2/tokenP", json={"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret}, timeout=30)
        r.raise_for_status()
        j = r.json()
        self._token = j["access_token"]
        self._token_exp = time.time() + float(j.get("expires_in", 86400))
        return self._token

    def _headers(self, tr_id: str) -> dict:
        return {"content-type": "application/json; charset=utf-8", "authorization": f"Bearer {self._auth()}", "appkey": self.app_key,
                "appsecret": self.app_secret, "tr_id": tr_id, "custtype": "P"}

    def _tr(self, real: str) -> str:
        return ("V" + real[1:]) if self.paper else real

    # -- account
    def account(self) -> dict:
        pos = self._balance()
        cash = float(pos.get("output2", {}).get("frcr_pchs_amt1", 0) or 0)  # 참고용; 정확한 주문가능금액은 별도 API(해외주식 매수가능금액조회)
        equity = sum(float(p["ovrs_stck_evlu_amt"]) for p in pos.get("output1", [])) + cash
        return {"equity": equity, "cash": cash}

    def _balance(self) -> dict:
        params = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "OVRS_EXCG_CD": "NASD", "TR_CRCY_CD": "USD", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""}
        r = self.session.get(f"{self.base}/uapi/overseas-stock/v1/trading/inquire-balance", headers=self._headers(self._tr("TTTS3012R")), params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def positions(self) -> dict[str, dict]:
        out = {}
        for p in self._balance().get("output1", []):
            qty = int(float(p.get("ovrs_cblc_qty", 0)))
            if qty > 0:
                out[p["ovrs_pdno"]] = {"qty": qty, "avg_price": float(p.get("pchs_avg_pric", 0))}
        return out

    # -- orders
    def submit(self, order: Order) -> dict:
        tr = self._tr("TTTT1002U" if order.side == "buy" else "TTTT1006U")
        body = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "OVRS_EXCG_CD": order.exchange, "PDNO": order.symbol,
                "ORD_QTY": str(order.qty), "OVRS_ORD_UNPR": f"{order.limit_price:.2f}", "ORD_SVR_DVSN_CD": "0", "ORD_DVSN": "00"}
        log.info("[KIS%s] %s %d %s @ %.2f (%s)", " 모의" if self.paper else "", order.side.upper(), order.qty, order.symbol, order.limit_price, order.reason)
        r = self.session.post(f"{self.base}/uapi/overseas-stock/v1/trading/order", headers=self._headers(tr), data=json.dumps(body), timeout=30)
        j = r.json()
        if r.status_code >= 400 or j.get("rt_cd") not in (None, "0"):
            raise RuntimeError(f"KIS order failed: {r.status_code} {j.get('msg_cd')} {j.get('msg1')}")
        return j


def make_broker(name: str, state_file: Path, equity: float = 100_000.0) -> Broker:
    if name == "dryrun":
        return DryRunBroker(state_file, equity)
    if name == "kis":
        return KISBroker()
    raise ValueError(f"unknown broker {name!r}")
