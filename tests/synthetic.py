"""Synthetic gold-like M5 data with trending regimes, for tests only (real data comes from data.py)."""
import numpy as np
import pandas as pd


def make_m5(start="2022-06-01", end="2026-11-02", seed=7, price=1800.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="5min", tz="UTC")
    ny = idx.tz_convert("America/New_York")
    # gold trades Sun 18:00 - Fri 17:00 NY with a 17:00-18:00 daily break
    open_ = ~((ny.dayofweek == 5) | ((ny.dayofweek == 4) & (ny.hour >= 17)) |
              ((ny.dayofweek == 6) & (ny.hour < 18)) | (ny.hour == 17))
    idx = idx[open_]
    n = len(idx)
    hours = idx.hour.to_numpy()
    vol = np.where((hours >= 7) & (hours < 17), 1.0, 0.45) * price * 0.0009
    regime = np.repeat(rng.choice([-1, 1], size=n // 20000 + 1) * rng.uniform(0.02, 0.06, n // 20000 + 1), 20000)[:n]
    ret = rng.standard_t(4, n) * vol / price * 0.6 + regime * vol / price * 0.1
    close = price * np.exp(np.cumsum(ret))
    op = np.concatenate([[price], close[:-1]])
    spread = np.abs(rng.normal(0, vol / 2, n))
    high = np.maximum(op, close) + spread * rng.uniform(0.2, 1.2, n)
    low = np.minimum(op, close) - spread * rng.uniform(0.2, 1.2, n)
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close,
                         "volume": rng.integers(50, 500, n).astype(float)}, index=idx)
