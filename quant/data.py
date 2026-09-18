"""데이터 계층: 유니버스(S&P 1500) 수집, Yahoo Finance 일봉 다운로드, 로컬 parquet 캐시.

가격 패널(panel) 포맷
---------------------
dict[str, pd.DataFrame] — 키: open/high/low/close/volume,
각 DataFrame 은 index=거래일(DatetimeIndex), columns=심볼. 결측은 NaN.
전략과 백테스터는 모두 이 wide 패널을 입력으로 받는다(벡터 연산이 빠르다).
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
UNIVERSE_FILE = DATA_DIR / "universe.csv"          # S&P 1500
UNIVERSE_ALL_FILE = DATA_DIR / "universe_all.csv"  # 미국 전체 상장 보통주 (NASDAQ 스크리너)
PRICES_FILE = CACHE_DIR / "prices.parquet"
NASDAQ_SCREENER = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true"
# 보통주가 아닌 것(워런트/유닛/우선주/채권/ETF/SPAC 등) 제외용
EXCLUDE_NAME_RE = (
    r"warrant|\bunits?\b|\brights?\b|preferred|depositary|\bnotes?\b|debenture|\betf\b|\betn\b|\bfund\b|"
    r"acquisition|\bspac\b|% |subordinated|\bbonds?\b|trust preferred|closed[- ]end"
)

WIKI = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
BENCHMARK = "SPY"
FIELDS = ["open", "high", "low", "close", "volume"]


def to_yahoo_symbol(symbol: str) -> str:
    """Wikipedia/NASDAQ 표기(BRK.B, AKO/A) -> Yahoo 표기(BRK-B, AKO-A)."""
    return symbol.strip().upper().replace(".", "-").replace("/", "-")


# --------------------------------------------------------------------------- universe
def fetch_universe(indices=("sp500", "sp400", "sp600")) -> pd.DataFrame:
    """Wikipedia 에서 S&P 500/400/600 구성 종목을 읽어 하나의 표로 합친다.

    주의: '현재' 구성 종목이므로 과거 구간에 대해 생존 편향(survivorship bias)이 있다.
    README 의 한계 항목 참고.
    """
    frames = []
    headers = {"User-Agent": "Mozilla/5.0 (quant research)"}
    for name in indices:
        url = WIKI[name]
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        table = next(t for t in tables if "Symbol" in t.columns)
        df = pd.DataFrame(
            {
                "symbol": table["Symbol"].map(to_yahoo_symbol),
                "name": table["Security"],
                "sector": table["GICS Sector"],
                "index": name,
            }
        )
        frames.append(df)
        log.info("%s: %d symbols", name, len(df))
    uni = pd.concat(frames, ignore_index=True).drop_duplicates("symbol").sort_values("symbol")
    return uni.reset_index(drop=True)


def fetch_universe_all() -> pd.DataFrame:
    """NASDAQ 스크리너에서 미국 전체 상장 종목(NASDAQ/NYSE/AMEX)을 받아 보통주만 남긴다.

    columns: symbol, name, sector, industry, market_cap(현재), shares_out(=market_cap/lastsale), ipo_year
    시총은 '현재' 값이므로 과거 시점의 시총은 shares_out * 당시 종가로 근사한다(자사주/증자 무시).
    """
    import re

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(NASDAQ_SCREENER, headers=headers, timeout=60)
    r.raise_for_status()
    rows = r.json()["data"]["rows"]
    raw = pd.DataFrame(rows)
    df = pd.DataFrame(
        {
            "symbol": raw["symbol"].str.strip(),
            "name": raw["name"].str.strip(),
            "sector": raw["sector"].replace("", np.nan),
            "industry": raw["industry"].replace("", np.nan),
            "market_cap": pd.to_numeric(raw["marketCap"], errors="coerce"),
            "last_sale": pd.to_numeric(raw["lastsale"].str.replace("$", "", regex=False), errors="coerce"),
            "ipo_year": pd.to_numeric(raw["ipoyear"], errors="coerce"),
            "country": raw["country"].replace("", np.nan),
        }
    )
    bad_name = df["name"].str.contains(EXCLUDE_NAME_RE, flags=re.IGNORECASE, regex=True)
    bad_symbol = df["symbol"].str.contains(r"[\^\s]", regex=True) | (df["symbol"].str.len() > 5)
    df = df[~bad_name & ~bad_symbol & df["last_sale"].notna() & (df["last_sale"] > 0)].copy()
    df["symbol"] = df["symbol"].map(to_yahoo_symbol)
    df["shares_out"] = df["market_cap"] / df["last_sale"]
    df = df.drop(columns=["last_sale"]).drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    log.info("universe_all: %d common stocks (from %d rows)", len(df), len(raw))
    return df


def save_universe(uni: pd.DataFrame, path: Path = UNIVERSE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uni.to_csv(path, index=False)


def load_universe(name: str = "sp1500") -> pd.DataFrame:
    """name: "sp1500" | "all" 또는 CSV 경로."""
    path = {"sp1500": UNIVERSE_FILE, "all": UNIVERSE_ALL_FILE}.get(name, Path(name))
    return pd.read_csv(path)


# --------------------------------------------------------------------------- prices
def download_prices(symbols: list[str], start: str, end: str, batch_size: int = 100, pause: float = 1.0) -> pd.DataFrame:
    """Yahoo Finance 에서 일봉(OHLCV, 분할/배당 조정)을 받아 long 포맷으로 반환.

    columns: date, symbol, open, high, low, close, volume
    """
    import yfinance as yf

    symbols = list(dict.fromkeys(symbols))
    frames = []
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i : i + batch_size]
        for attempt in range(3):
            try:
                raw = yf.download(
                    chunk,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                break
            except Exception as e:  # pragma: no cover - network
                log.warning("batch %d attempt %d failed: %s", i, attempt, e)
                time.sleep(5 * (attempt + 1))
        else:
            continue
        if raw is None or raw.empty:
            continue
        if not isinstance(raw.columns, pd.MultiIndex):  # single ticker
            raw.columns = pd.MultiIndex.from_product([chunk, raw.columns])
        for sym in chunk:
            if sym not in raw.columns.get_level_values(0):
                continue
            df = raw[sym].dropna(how="all")
            if df.empty:
                continue
            df = df.rename(columns=str.lower)[FIELDS].copy()
            df["symbol"] = sym
            df.index.name = "date"
            frames.append(df.reset_index())
        log.info("downloaded %d/%d symbols", min(i + batch_size, len(symbols)), len(symbols))
        time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", *FIELDS])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def save_prices(long_df: pd.DataFrame, path: Path = PRICES_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(path, index=False)


def load_prices_long(path: Path = PRICES_FILE) -> pd.DataFrame:
    return pd.read_parquet(path)


def to_panels(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """long -> wide 패널. 모든 필드가 같은 index/columns 를 공유하도록 정렬한다."""
    panels = {}
    for f in FIELDS:
        wide = long_df.pivot(index="date", columns="symbol", values=f).sort_index()
        panels[f] = wide.astype(float)
    cols = panels["close"].columns
    for f in FIELDS:
        panels[f] = panels[f].reindex(columns=cols)
    return panels


def load_panels(path: Path = PRICES_FILE, symbols: list[str] | None = None, universe: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """parquet -> 패널. universe 에 shares_out 이 있으면 근사 시총 패널 "mcap" 을 추가한다."""
    long_df = load_prices_long(path)
    if universe is not None:
        symbols = list(set(universe["symbol"]) | ({BENCHMARK} if symbols is None else set(symbols)))
    if symbols is not None:
        long_df = long_df[long_df["symbol"].isin(set(symbols))]
    panels = to_panels(long_df)
    if universe is not None and "shares_out" in universe.columns:
        shares = universe.set_index("symbol")["shares_out"].reindex(panels["close"].columns)
        panels["mcap"] = panels["close"] * shares.to_numpy()[None, :]
    return panels


def merge_prices(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """기존 long 데이터에 새 심볼/일자를 합친다(중복은 새 데이터 우선)."""
    if existing is None or existing.empty:
        return new
    out = pd.concat([existing, new], ignore_index=True)
    out = out.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def slice_panels(panels: dict[str, pd.DataFrame], start=None, end=None) -> dict[str, pd.DataFrame]:
    return {k: v.loc[start:end] for k, v in panels.items()}


def split_benchmark(panels: dict[str, pd.DataFrame], symbol: str = BENCHMARK):
    """벤치마크(SPY)를 패널에서 분리해 (stocks_panel, benchmark_df) 로 반환."""
    bench = pd.DataFrame({f: panels[f][symbol] for f in FIELDS})
    stocks = {f: v.drop(columns=[symbol], errors="ignore") for f, v in panels.items()}
    return stocks, bench
