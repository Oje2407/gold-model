"""Backtest reporting: last 2 years vs this year, and the breakdowns that show what is working now."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Config, Engine


def stats(tr: list[dict]) -> dict:
    r = np.array([t["r"] for t in tr if t.get("r") is not None], dtype=float)
    if r.size == 0:
        return {"trades": 0}
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq).max())
    wins, losses = r[r > 0.05], r[r < -0.05]
    return {
        "trades": int(r.size),
        "win_rate": round(float((r > 0.05).mean() * 100), 1),
        "breakeven_pct": round(float((np.abs(r) <= 0.05).mean() * 100), 1),
        "avg_r": round(float(r.mean()), 3),
        "total_r": round(float(r.sum()), 2),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.size else None,
        "avg_win_r": round(float(wins.mean()), 2) if wins.size else 0,
        "avg_loss_r": round(float(losses.mean()), 2) if losses.size else 0,
        "max_dd_r": round(dd, 2),
        "longest_losing_streak": int(max((len(s) for s in "".join("L" if x < -0.05 else "W" for x in r).split("W")), default=0)),
    }


def breakdown(tr: list[dict], key) -> list[dict]:
    groups: dict = {}
    for t in tr:
        if t.get("r") is None:
            continue
        groups.setdefault(key(t), []).append(t)
    return [{"key": k, **stats(v)} for k, v in sorted(groups.items())]


def periods(end: pd.Timestamp) -> dict:
    return {
        "last_2_years": (end - pd.DateOffset(years=2), end),
        f"{end.year}_ytd": (pd.Timestamp(f"{end.year}-01-01", tz="UTC"), end),
        f"{end.year - 1}": (pd.Timestamp(f"{end.year - 1}-01-01", tz="UTC"), pd.Timestamp(f"{end.year}-01-01", tz="UTC")),
    }


def run_backtest(frames: dict, cfg: Config, years: float = 2.0) -> dict:
    eng = Engine(frames, cfg)
    end = frames["M5"].df.index[-1] + pd.Timedelta(minutes=5)
    start = max(end - pd.DateOffset(years=years), frames["M5"].df.index[0] + pd.Timedelta(days=3))
    trades = [t for t in eng.run(int(start.timestamp()), int(end.timestamp())) if t["status"] == "CLOSED"]

    def within(a, b):
        return [t for t in trades if a <= pd.Timestamp(t["entry_time"]) < b]

    per = {name: stats(within(a, b)) for name, (a, b) in periods(end).items()}
    month = lambda t: t["entry_time"][:7]
    eq, c = [], 0.0
    for t in trades:
        c += t["r"]
        eq.append({"time": t["exit_time"], "r": round(c, 3)})
    return {
        "range": [start.isoformat(), end.isoformat()],
        "data_start": frames["M5"].df.index[0].isoformat(),
        "config": cfg.__dict__,
        "periods": per,
        "by_setup": {p: breakdown(within(a, b), lambda t: t["setup"]) for p, (a, b) in periods(end).items()},
        "by_grade": {p: breakdown(within(a, b), lambda t: t["grade"]) for p, (a, b) in periods(end).items()},
        "by_session": {p: breakdown(within(a, b), lambda t: t["session"]) for p, (a, b) in periods(end).items()},
        "by_direction": {p: breakdown(within(a, b), lambda t: t["direction"]) for p, (a, b) in periods(end).items()},
        "by_month": breakdown(trades, month),
        "equity": eq,
        "trades": trades[-400:],
        "regime_note": regime_note(per, end),
    }


def regime_note(per: dict, end) -> str:
    a, b = per.get("last_2_years", {}), per.get(f"{end.year}_ytd", {})
    if not a.get("trades") or not b.get("trades"):
        return "Not enough trades in one of the periods to compare."
    diff = b["avg_r"] - a["avg_r"]
    if b["trades"] < 20:
        return f"Only {b['trades']} trades this year - treat the {end.year} numbers as indicative only."
    if diff < -0.15:
        return (f"The edge has weakened this year: {b['avg_r']:+.2f}R/trade in {end.year} vs "
                f"{a['avg_r']:+.2f}R over 2 years. Check the setup/session tables for what stopped working.")
    if diff > 0.15:
        return f"This year is running better than the 2-year average ({b['avg_r']:+.2f}R vs {a['avg_r']:+.2f}R per trade)."
    return f"This year is in line with the 2-year average ({b['avg_r']:+.2f}R vs {a['avg_r']:+.2f}R per trade)."


SWEEP = [
    {"label": "Default (monthly required, 1 of W/D/H4)", "cfg": {}},
    {"label": "Monthly + weekly required", "cfg": {"require_weekly": True}},
    {"label": "Full alignment M/W/D/H4", "cfg": {"require_weekly": True, "min_htf_agree": 3}},
    {"label": "Daily-candle setups only (no S/R)", "cfg": {"use_sr_setups": False}},
    {"label": "No 'struggle' filter", "cfg": {"struggle_required": False}},
    {"label": "1:3 minimum R:R", "cfg": {"min_rr": 3.0, "max_rr": 5.0}},
]


def sweep(frames: dict, base: dict, years: float = 2.0) -> list[dict]:
    """Same model, several rule variants, scored per period - shows which variant suits the current regime."""
    end = frames["M5"].df.index[-1] + pd.Timedelta(minutes=5)
    start = max(end - pd.DateOffset(years=years), frames["M5"].df.index[0] + pd.Timedelta(days=3))
    out = []
    for v in SWEEP:
        cfg = Config.from_dict({**base, **v["cfg"]})
        tr = [t for t in Engine(frames, cfg).run(int(start.timestamp()), int(end.timestamp())) if t["status"] == "CLOSED"]
        row = {"label": v["label"]}
        for name, (a, b) in periods(end).items():
            s = stats([t for t in tr if a <= pd.Timestamp(t["entry_time"]) < b])
            row[name] = {k: s.get(k) for k in ("trades", "win_rate", "avg_r", "total_r", "max_dd_r")}
        out.append(row)
    return out
