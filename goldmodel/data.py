"""Market data for XAUUSD spot and gold futures.

Sources
-------
spot     history : Dukascopy free datafeed (1-minute + hourly bid candles, many years back)
spot     live    : Twelve Data (set TWELVE_DATA_API_KEY) or Dukascopy (lags a little)
futures  history : yfinance GC=F (daily = full history, 1h = 730 days, 5m = 60 days only)
                   -> for a 2-year futures backtest import an M5 CSV from your platform
futures  live    : yfinance GC=F 5m (delayed ~10-15 min by the exchange)
any      CSV     : import_csv() for MT5 / TradingView / broker exports

Every frame is OHLCV with a tz-aware UTC DatetimeIndex marking the bar START.
Forming (incomplete) bars are always dropped, so the model only ever sees closed bars.
"""
from __future__ import annotations

import datetime as dt
import lzma
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
COLS = ["open", "high", "low", "close", "volume"]
UA = {"User-Agent": "Mozilla/5.0 (gold-model research script)"}
DUKA = "https://datafeed.dukascopy.com/datafeed"
_BI5 = np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4")])

INSTRUMENTS = {
    "spot": {"label": "XAUUSD spot", "duka": "XAUUSD", "point": 1000.0, "spread": 0.30},
    "futures": {"label": "Gold futures GC (front month)", "yf": "GC=F", "spread": 0.10},
}


# ----------------------------------------------------------------------------- helpers
def empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in COLS}, index=pd.DatetimeIndex([], tz="UTC"))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty()
    df = df[COLS].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0)]
    return df


def ohlc_agg(df: pd.DataFrame, keys) -> pd.DataFrame:
    g = df.groupby(keys, sort=True)
    return g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                 close=("close", "last"), volume=("volume", "sum"))


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """UTC-aligned resample (for M5, M15, H1)."""
    if df.empty:
        return empty()
    return ohlc_agg(df, df.index.floor(rule))


def drop_forming(df: pd.DataFrame, bar: pd.Timedelta, now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = now or pd.Timestamp.now(tz="UTC")
    return df[df.index + bar <= now]


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return clean(new)
    if new is None or new.empty:
        return old
    return clean(pd.concat([old, new]))


def save(df: pd.DataFrame, name: str) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / f"{name}.csv.gz"
    out = df.copy()
    out.index = out.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.index.name = "time"
    out.to_csv(p)
    return p


def load(name: str) -> pd.DataFrame:
    p = DATA / f"{name}.csv.gz"
    if not p.exists():
        return empty()
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return clean(df)


# ----------------------------------------------------------------------------- Dukascopy
def _http_get(url: str, cache: Path | None) -> bytes:
    if cache is not None and cache.exists():
        return cache.read_bytes()
    last = None
    for attempt in range(6):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 404:
                data = b""
            elif r.status_code == 200:
                data = r.content
            elif r.status_code in (429, 503):          # throttled: back off properly
                last = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            else:
                raise RuntimeError(f"HTTP {r.status_code}")
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
            return data
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Dukascopy download failed for {url}: {last}")


def decode_bi5(raw: bytes, base: pd.Timestamp, point: float) -> pd.DataFrame:
    """Decode a Dukascopy *_candles_*.bi5 file (LZMA, 24-byte big-endian records)."""
    if not raw:
        return empty()
    buf = lzma.decompress(raw)
    a = np.frombuffer(buf[: len(buf) // 24 * 24], dtype=_BI5)
    if a.size == 0:
        return empty()
    idx = base + pd.to_timedelta(a["t"].astype(np.int64), unit="s")
    df = pd.DataFrame({"open": a["o"] / point, "high": a["h"] / point, "low": a["l"] / point,
                       "close": a["c"] / point, "volume": a["v"].astype(float)},
                      index=pd.DatetimeIndex(idx))
    # Dukascopy pads closed-market minutes with flat zero-volume candles
    flat = (df["volume"] <= 0) & (df["high"] == df["low"])
    return df[~flat]


def _fix_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Guard against a wrong point value: gold must be priced between $200 and $20,000."""
    if df.empty:
        return df
    med = float(df["close"].median())
    for f in (1, 10, 0.1, 100, 0.01, 1000, 0.001):
        if 200 <= med * f <= 20000:
            if f != 1:
                df = df.copy()
                df[["open", "high", "low", "close"]] *= f
            return df
    raise ValueError(f"Unexpected gold price scale (median {med})")


def dukascopy(symbol: str, kind: str, start: str, end: str | None = None,
              point: float = 1000.0, workers: int | None = None, progress=print) -> pd.DataFrame:
    """kind='min' -> 1-minute candles (one file per day); kind='hour' -> 1h (one file per month)."""
    today = pd.Timestamp.now(tz="UTC").normalize()
    s = pd.Timestamp(start, tz="UTC").normalize()
    e = pd.Timestamp(end, tz="UTC").normalize() if end else today
    jobs = []
    if kind == "min":
        for d in pd.date_range(s, e, freq="D"):
            if d.dayofweek == 5:  # Saturday: market closed
                continue
            url = f"{DUKA}/{symbol}/{d.year:04d}/{d.month - 1:02d}/{d.day:02d}/BID_candles_min_1.bi5"
            cache = RAW / symbol / "min" / f"{d:%Y-%m-%d}.bi5" if d < today else None
            jobs.append((url, d, cache))
    elif kind == "hour":
        for m in pd.date_range(s.replace(day=1), e, freq="MS"):
            url = f"{DUKA}/{symbol}/{m.year:04d}/{m.month - 1:02d}/BID_candles_hour_1.bi5"
            month_done = (m + pd.offsets.MonthBegin(1)) <= today
            cache = RAW / symbol / "hour" / f"{m:%Y-%m}.bi5" if month_done else None
            jobs.append((url, m, cache))
    else:
        raise ValueError(kind)

    frames, done = [], 0
    workers = workers or int(os.environ.get("DUKA_WORKERS", "4"))

    def work(job):
        url, base, cache = job
        return decode_bi5(_http_get(url, cache), base, point)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for df in ex.map(work, jobs):
            frames.append(df)
            done += 1
            if progress and (done % 20 == 0 or done == len(jobs)):
                progress(f"  {symbol} {kind}: {done}/{len(jobs)} files")
    frames = [f for f in frames if not f.empty]
    if not frames:
        return empty()
    return _fix_scale(clean(pd.concat(frames)))


# ----------------------------------------------------------------------------- yfinance (futures)
def yf_download(ticker: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf  # optional dependency

    df = yf.download(ticker, interval=interval, period=period, auto_adjust=False, progress=False, timeout=30, threads=False)
    if df is None or df.empty:
        return empty()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    if "volume" not in df:
        df["volume"] = 0.0
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("UTC")
    return clean(df)


# ----------------------------------------------------------------------------- OANDA (spot, recommended)
def _oanda_base() -> str:
    env = os.environ.get("OANDA_ENV", "practice").lower()
    return "https://api-fxtrade.oanda.com" if env in ("live", "trade", "fxtrade") else "https://api-fxpractice.oanda.com"


def oanda_parse(candles: list) -> pd.DataFrame:
    rows = [c for c in candles if c.get("complete")]
    if not rows:
        return empty()
    px = [c.get("bid") or c.get("mid") for c in rows]
    df = pd.DataFrame({"open": [float(p["o"]) for p in px], "high": [float(p["h"]) for p in px],
                       "low": [float(p["l"]) for p in px], "close": [float(p["c"]) for p in px],
                       "volume": [float(c.get("volume", 0)) for c in rows]},
                      index=pd.DatetimeIndex(pd.to_datetime([c["time"] for c in rows], utc=True)))
    return clean(df)


def oanda(granularity: str, start: str | pd.Timestamp | None = None, count: int | None = None,
          instrument: str = "XAU_USD", progress=print) -> pd.DataFrame:
    """Bid candles (M5, H1, D...) from OANDA v20. Needs OANDA_API_TOKEN (a free demo account works)."""
    token = os.environ.get("OANDA_API_TOKEN")
    if not token:
        raise RuntimeError("OANDA_API_TOKEN not set")
    url = f"{_oanda_base()}/v3/instruments/{instrument}/candles"
    hdr = {"Authorization": f"Bearer {token}", "Accept-Datetime-Format": "RFC3339"}

    def get(params):
        for attempt in range(4):
            r = requests.get(url, headers=hdr, params=params, timeout=30)
            if r.status_code == 200:
                return r.json().get("candles", [])
            if r.status_code in (401, 403):
                raise RuntimeError(f"OANDA rejected the token (HTTP {r.status_code}). Check OANDA_API_TOKEN / OANDA_ENV.")
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"OANDA HTTP {r.status_code}: {r.text[:200]}")

    if start is None:
        return oanda_parse(get({"granularity": granularity, "price": "B", "count": count or 500}))
    frames, cur, now, n = [], pd.Timestamp(start, tz="UTC") if not isinstance(start, pd.Timestamp) else start, pd.Timestamp.now(tz="UTC"), 0
    while cur < now:
        c = get({"granularity": granularity, "price": "B", "count": 5000, "from": cur.strftime("%Y-%m-%dT%H:%M:%SZ")})
        if not c:
            break
        frames.append(oanda_parse(c))
        last = pd.Timestamp(c[-1]["time"])
        if last <= cur:
            break
        cur = last + pd.Timedelta(seconds=1)
        n += 1
        if progress and n % 5 == 0:
            progress(f"  OANDA {instrument} {granularity}: up to {last:%Y-%m-%d}")
        if len(c) < 5000:
            break
    frames = [f for f in frames if not f.empty]
    return clean(pd.concat(frames)) if frames else empty()


# ----------------------------------------------------------------------------- Twelve Data (spot live)
def twelvedata(symbol: str = "XAU/USD", interval: str = "5min", n: int = 300, key: str | None = None) -> pd.DataFrame:
    key = key or os.environ.get("TWELVE_DATA_API_KEY")
    if not key:
        raise RuntimeError("TWELVE_DATA_API_KEY not set")
    r = requests.get("https://api.twelvedata.com/time_series", timeout=30, params=dict(
        symbol=symbol, interval=interval, outputsize=n, apikey=key, timezone="UTC", order="ASC"))
    j = r.json()
    if j.get("status") != "ok":
        raise RuntimeError(f"Twelve Data: {j.get('message', j)}")
    df = pd.DataFrame(j["values"])
    df.index = pd.to_datetime(df.pop("datetime"), utc=True)
    if "volume" not in df:
        df["volume"] = 0.0
    return clean(df.astype(float))


# ----------------------------------------------------------------------------- CSV import
def import_csv(path: str, tz: str = "UTC", time_col: str | None = None) -> pd.DataFrame:
    """Import a broker/platform export. Accepts MT5 (<DATE> <TIME> ...), TradingView (time,open,...) etc."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip("<>").strip().lower() for c in df.columns]
    if time_col:
        t = df[time_col.lower()]
    elif "date" in df and "time" in df:
        t = df["date"].astype(str) + " " + df["time"].astype(str)
    else:
        t = df[[c for c in df.columns if c in ("time", "datetime", "date", "timestamp")][0]]
    if np.issubdtype(np.asarray(t).dtype, np.number):
        idx = pd.to_datetime(t, unit="s", utc=True)
    else:
        idx = pd.to_datetime(t.astype(str).str.replace(".", "-", regex=False), utc=False)
        idx = idx.dt.tz_localize(tz) if idx.dt.tz is None else idx
        idx = idx.dt.tz_convert("UTC")
    df.index = pd.DatetimeIndex(idx)
    vol = next((c for c in ("volume", "tickvol", "vol") if c in df), None)
    df["volume"] = df[vol] if vol else 0.0
    return clean(df)


# ----------------------------------------------------------------------------- high level
def download_history(inst: str, h1_start: str = "2015-01-01", m5_start: str | None = None, progress=print):
    """Fetch and store everything the model needs. Spot: Dukascopy. Futures: yfinance."""
    m5_start = m5_start or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365 * 2 + 120)).strftime("%Y-%m-%d")
    if inst == "spot" and os.environ.get("OANDA_API_TOKEN"):
        progress(f"Downloading OANDA XAU_USD hourly from {h1_start} and 5-minute from {m5_start}...")
        h1 = oanda("H1", h1_start, progress=progress)
        m5 = oanda("M5", m5_start, progress=progress)
        if m5.empty:
            raise RuntimeError("OANDA returned no XAU_USD data")
        save(h1, "spot_h1"), save(m5, "spot_m5")
        return {"source": "OANDA", "h1": len(h1), "m5": len(m5)}
    if inst == "spot":
        sym, pt = INSTRUMENTS["spot"]["duka"], INSTRUMENTS["spot"]["point"]
        progress(f"Downloading {sym} hourly from {h1_start} (for monthly/weekly bias)...")
        h1 = dukascopy(sym, "hour", h1_start, point=pt, progress=progress)
        progress(f"Downloading {sym} 1-minute from {m5_start} (for M5/M15 entries and the backtest)...")
        m1 = dukascopy(sym, "min", m5_start, point=pt, progress=progress)
        m5 = drop_forming(resample(m1, "5min"), pd.Timedelta(minutes=5))
        save(h1, "spot_h1"), save(m5, "spot_m5")
        return {"h1": len(h1), "m5": len(m5)}
    if inst == "futures":
        t = INSTRUMENTS["futures"]["yf"]
        d1 = yf_download(t, "1d", "max")
        if d1.empty:
            raise RuntimeError("Yahoo returned no futures data")
        h1 = yf_download(t, "1h", "730d")
        m5 = drop_forming(yf_download(t, "5m", "60d"), pd.Timedelta(minutes=5))
        save(d1, "futures_d1"), save(h1, "futures_h1"), save(merge(load("futures_m5"), m5), "futures_m5")
        return {"d1": len(d1), "h1": len(h1), "m5": len(load("futures_m5"))}
    raise ValueError(inst)


def update_live(inst: str, m5: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Append the latest closed M5 bars. Returns (frame, source used)."""
    bar = pd.Timedelta(minutes=5)
    if inst == "spot":
        if os.environ.get("OANDA_API_TOKEN"):
            new, src = oanda("M5", count=300), "OANDA (live)"
        elif os.environ.get("TWELVE_DATA_API_KEY"):
            new, src = twelvedata(n=300), "Twelve Data"
        else:
            start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            m1 = dukascopy(INSTRUMENTS["spot"]["duka"], "min", start, progress=None)
            new, src = resample(m1, "5min"), "Dukascopy"
    else:
        new, src = yf_download(INSTRUMENTS["futures"]["yf"], "5m", "5d"), "Yahoo GC=F"
    return merge(m5, drop_forming(new, bar)), src
