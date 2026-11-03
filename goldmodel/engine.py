"""Top-down gold model.

Monthly -> Weekly -> Daily -> H4 decide direction; M15 finds the rejection; M5 confirms the entry.

Setups
  PDH_REJECT  sell: yesterday struggled to break the day-before's high, price returns to yesterday's
              high (PDH) and an M15 candle wicks into it and closes back below.
  PDL_REJECT  buy : mirror image at yesterday's low (wick included).
  SR_RESIST / SR_SUPPORT : same rejection logic at clustered daily/H4 support & resistance zones.

Look-ahead safety: a higher-timeframe bar is only used once its END time is <= the decision time,
swing pivots are only used once the n bars to their right have closed, and entries are filled at the
OPEN of the bar after the trigger bar.  tests/test_engine.py re-runs the model on data truncated at
each entry and checks every signal is reproduced exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data import ohlc_agg

NY = "America/New_York"
SHIFT = pd.Timedelta(hours=7)  # 17:00 New York (gold's daily close) -> 00:00 "trading clock"
HTF = ["M", "W", "D", "H4"]
SWING_N = {"M": 2, "W": 2, "D": 3, "H4": 3}


@dataclass
class Config:
    # --- direction / confluence
    require_monthly: bool = True        # trade only in the monthly direction
    require_weekly: bool = False        # True = weekly must agree too ("ideally")
    min_htf_agree: int = 1              # of W, D, H4 at least this many must agree
    # --- daily-candle setup
    struggle_required: bool = True      # prior day must have struggled at the day-before's high/low
    struggle_atr: float = 0.25          # "struggled" = got within this many daily ATRs and closed back
    use_sr_setups: bool = True
    min_zone_strength: float = 3.0
    # --- rejection & trigger
    level_tol_atr: float = 0.10         # wick may fall this far short of the level and still count
    wick_ratio: float = 0.40            # rejection wick >= 40% of the M15 candle range
    confirm_bars: int = 6               # M5 bars allowed for the confirmation (30 min)
    # --- risk
    sl_buffer_atr: float = 0.05
    min_risk_atr: float = 0.05
    max_risk_atr: float = 0.60
    min_rr: float = 2.0                 # minimum 1:2
    max_rr: float = 4.0
    obstacle_rr: float = 1.0            # skip if an opposing level sits closer than 1R
    be_at_r: float = 1.0                # move stop to breakeven at +1R (0 = off)
    # --- timing
    session_start_utc: int = 7          # London open
    session_end_utc: int = 20           # NY afternoon
    max_trades_per_day: int = 2
    # --- costs
    spread: float = 0.30
    # --- support / resistance
    zone_lookback_d: int = 250
    zone_lookback_h4: int = 300
    zone_merge_atr: float = 0.25
    zone_pad_atr: float = 0.05

    @classmethod
    def from_dict(cls, d: dict | None) -> "Config":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                setattr(c, k, type(getattr(c, k))(v))
        return c


# ============================================================================ frames
def _secs(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(idx.as_unit("s").asi8, dtype=np.int64)


def _shifted(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert(NY).tz_localize(None) + SHIFT


def _unshift(naive: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (naive - SHIFT).tz_localize(NY, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")


@dataclass
class Frame:
    df: pd.DataFrame          # OHLCV, index = bar start (UTC)
    end: np.ndarray           # bar end, epoch seconds

    def __len__(self):
        return len(self.df)


def build_frames(m5: pd.DataFrame, h1: pd.DataFrame | None = None, d1: pd.DataFrame | None = None) -> dict:
    """Build M5, M15, H4, D, W, M frames. Days run 17:00-17:00 New York (gold's settlement)."""
    m5 = m5.sort_index()
    m5 = m5[~m5.index.duplicated(keep="last")]
    f = {"M5": Frame(m5, _secs(m5.index) + 300)}
    m15 = ohlc_agg(m5, m5.index.floor("15min"))
    f["M15"] = Frame(m15, _secs(m15.index) + 900)

    h1m = ohlc_agg(m5, m5.index.floor("h"))
    hb = h1m if h1 is None or h1.empty else pd.concat([h1[h1.index < h1m.index[0]], h1m])
    h4 = ohlc_agg(hb, (hb.index - pd.Timedelta(hours=2)).floor("4h") + pd.Timedelta(hours=2))
    f["H4"] = Frame(h4, _secs(h4.index) + 4 * 3600)

    day = ohlc_agg(hb, _shifted(hb.index).floor("D"))           # index: naive trading date
    if d1 is not None and not d1.empty:                          # older daily history (futures)
        old = d1.copy()
        old.index = pd.DatetimeIndex(old.index.tz_localize(None).normalize())
        day = pd.concat([old[old.index < day.index[0]], day])
    day = day[day.index.dayofweek < 5]                           # stray weekend prints
    wk = ohlc_agg(day, day.index - pd.to_timedelta(day.index.dayofweek, unit="D"))
    mo = ohlc_agg(day, day.index.to_period("M").to_timestamp())

    def finish(df, end_naive):
        out = df.copy()
        out.index = _unshift(pd.DatetimeIndex(df.index))
        return Frame(out, _secs(_unshift(pd.DatetimeIndex(end_naive))))

    f["D"] = finish(day, day.index + pd.Timedelta(days=1))
    f["W"] = finish(wk, wk.index + pd.Timedelta(days=5))         # week closes Fri 17:00 NY
    f["M"] = finish(mo, mo.index + pd.offsets.MonthBegin(1))
    return f


# ============================================================================ indicators
def ema(a: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def pivots(h: np.ndarray, l: np.ndarray, n: int):
    """Swing highs/lows. A pivot at i is only *known* at bar i+n (returned as `confirm`)."""
    N = len(h)
    hs = pd.Series(h)
    ls = pd.Series(l)
    cmax = hs.rolling(2 * n + 1, center=True).max().to_numpy()
    cmin = ls.rolling(2 * n + 1, center=True).min().to_numpy()
    lmax = hs.shift(1).rolling(n).max().to_numpy()   # strictly-left window, avoids duplicate pivots
    lmin = ls.shift(1).rolling(n).min().to_numpy()
    with np.errstate(invalid="ignore"):
        ph = np.where((h == cmax) & (h > lmax))[0]
        pl = np.where((l == cmin) & (l < lmin))[0]
    ph, pl = ph[ph + n < N], pl[pl + n < N]
    return ph, pl


def bias(df: pd.DataFrame, n: int):
    """+1 bull / -1 bear / 0 neutral for each CLOSED bar, using only bars up to and including it.

    Two votes: EMA stack (close vs EMA21 vs EMA50) and market structure (HH+HL / LH+LL, with a
    close through the last swing = break of structure).  Opposing votes cancel to neutral.
    """
    c, h, l = (df[k].to_numpy() for k in ("close", "high", "low"))
    N = len(c)
    e21, e50 = ema(c, 21), ema(c, 50)
    ema_v = np.where((c > e21) & (e21 > e50), 1, np.where((c < e21) & (e21 < e50), -1, 0))
    ema_v[: min(N, 21)] = 0
    ph, pl = pivots(h, l, n)
    ph_at = dict(zip(ph + n, ph))
    pl_at = dict(zip(pl + n, pl))
    struct = np.zeros(N, dtype=int)
    H, L = [], []
    for j in range(N):
        if j in ph_at:
            H.append(h[ph_at[j]])
        if j in pl_at:
            L.append(l[pl_at[j]])
        s = 0
        if len(H) >= 2 and len(L) >= 2:
            if H[-1] > H[-2] and L[-1] > L[-2]:
                s = 1
            elif H[-1] < H[-2] and L[-1] < L[-2]:
                s = -1
        if H and c[j] > H[-1]:
            s = 1
        if L and c[j] < L[-1]:
            s = -1
        struct[j] = s
    tot = ema_v + struct
    return np.sign(tot).astype(int), ema_v, struct


# ============================================================================ engine
@dataclass
class DayCtx:
    pdi: int
    pdh: float
    pdl: float
    ppdh: float
    ppdl: float
    pdc: float
    atr: float
    struggle_high: bool
    struggle_low: bool
    zones: list = field(default_factory=list)


class Engine:
    def __init__(self, frames: dict, cfg: Config | None = None):
        self.f = frames
        self.cfg = cfg or Config()
        self.bias = {}
        for tf in HTF:
            b, ev, sv = bias(frames[tf].df, SWING_N[tf])
            self.bias[tf] = (b, ev, sv)
        d = frames["D"].df
        self.d_atr = atr(d)
        self.d_h, self.d_l, self.d_c = (d[k].to_numpy() for k in ("high", "low", "close"))
        self.d_end = frames["D"].end
        h4 = frames["H4"].df
        self.piv = {}
        for tf, df_ in (("D", d), ("H4", h4)):
            ph, pl = pivots(df_["high"].to_numpy(), df_["low"].to_numpy(), SWING_N[tf])
            self.piv[tf] = (ph, df_["high"].to_numpy()[ph], pl, df_["low"].to_numpy()[pl])
        m5 = frames["M5"].df
        self.o5, self.h5, self.l5, self.c5 = (m5[k].to_numpy() for k in ("open", "high", "low", "close"))
        self.e5 = frames["M5"].end
        m15 = frames["M15"].df
        self.o15, self.h15, self.l15, self.c15 = (m15[k].to_numpy() for k in ("open", "high", "low", "close"))
        self.e15 = frames["M15"].end
        self.hour15 = ((self.e15 - 900) // 3600) % 24
        self.b15 = {tf: self.bias_at(tf, self.e15) for tf in HTF}
        self.pd15 = np.searchsorted(self.d_end, self.e15, side="right") - 1
        self._ctx = {}

    # ---------------------------------------------------------------- context
    def bias_at(self, tf: str, t):
        end = self.f[tf].end
        i = np.searchsorted(end, t, side="right") - 1
        b = self.bias[tf][0]
        return np.where(i >= 0, b[np.clip(i, 0, None)], 0) if len(b) else np.zeros_like(np.asarray(t))

    def zones(self, pdi: int) -> list[dict]:
        cfg, A = self.cfg, self.d_atr[pdi]
        h4i = np.searchsorted(self.f["H4"].end, self.d_end[pdi], side="right") - 1
        pts = []
        for tf, last, look, w in (("D", pdi, cfg.zone_lookback_d, 2.0), ("H4", h4i, cfg.zone_lookback_h4, 1.0)):
            n = SWING_N[tf]
            ph, phv, pl, plv = self.piv[tf]
            for pos, val in ((ph, phv), (pl, plv)):
                m = (pos + n <= last) & (pos >= last - look)
                pts += [(v, w) for v in val[m]]
        if not pts:
            return []
        pts.sort()
        tol = cfg.zone_merge_atr * A
        clusters, cur = [], [pts[0]]
        for p in pts[1:]:
            if p[0] - cur[0][0] <= tol:            # anchored: a zone is never wider than tol
                cur.append(p)
            else:
                clusters.append(cur)
                cur = [p]
        clusters.append(cur)
        pad = cfg.zone_pad_atr * A
        out = []
        for cl in clusters:
            s = sum(w for _, w in cl)
            if s >= 2:
                v = [x[0] for x in cl]
                out.append({"lo": round(min(v) - pad, 2), "hi": round(max(v) + pad, 2),
                            "mid": round(float(np.mean(v)), 2), "strength": s, "touches": len(cl)})
        return out

    def day_ctx(self, pdi: int) -> DayCtx | None:
        if pdi < 20:
            return None
        if pdi in self._ctx:
            return self._ctx[pdi]
        A = self.d_atr[pdi]
        h, l, c = self.d_h, self.d_l, self.d_c
        ppdh, ppdl = h[pdi - 1], l[pdi - 1]
        sa = self.cfg.struggle_atr * A
        ctx = DayCtx(pdi, h[pdi], l[pdi], ppdh, ppdl, c[pdi], A,
                     struggle_high=bool(h[pdi] >= ppdh - sa and c[pdi] < ppdh),
                     struggle_low=bool(l[pdi] <= ppdl + sa and c[pdi] > ppdl),
                     zones=self.zones(pdi))
        self._ctx[pdi] = ctx
        return ctx

    def levels(self, ctx: DayCtx, d: int) -> list[tuple]:
        cfg, A = self.cfg, ctx.atr
        out = []
        strong = [z for z in ctx.zones if z["strength"] >= cfg.min_zone_strength]
        if d == -1:
            if ctx.struggle_high or not cfg.struggle_required:
                conf = any(z["lo"] - 0.1 * A <= ctx.pdh <= z["hi"] + 0.1 * A for z in strong)
                out.append(("PDH_REJECT", ctx.pdh, conf, ctx.struggle_high))
            if cfg.use_sr_setups:
                out += [("SR_RESIST", z["lo"], True, False) for z in strong if abs(z["lo"] - ctx.pdh) > 0.2 * A]
        else:
            if ctx.struggle_low or not cfg.struggle_required:
                conf = any(z["lo"] - 0.1 * A <= ctx.pdl <= z["hi"] + 0.1 * A for z in strong)
                out.append(("PDL_REJECT", ctx.pdl, conf, ctx.struggle_low))
            if cfg.use_sr_setups:
                out += [("SR_SUPPORT", z["hi"], True, False) for z in strong if abs(z["hi"] - ctx.pdl) > 0.2 * A]
        return out

    # ---------------------------------------------------------------- scan
    def run(self, t_from: int | None = None, t_to: int | None = None) -> list[dict]:
        cfg = self.cfg
        k0 = 0 if t_from is None else int(np.searchsorted(self.e15, t_from, side="left"))
        k1 = len(self.e15) if t_to is None else int(np.searchsorted(self.e15, t_to, side="right"))
        trades, busy, per_day = [], -1, {}
        for k in range(k0, k1):
            t = int(self.e15[k])
            if t <= busy or not (cfg.session_start_utc <= self.hour15[k] < cfg.session_end_utc):
                continue
            pdi = int(self.pd15[k])
            ctx = self.day_ctx(pdi)
            if ctx is None or per_day.get(pdi, 0) >= cfg.max_trades_per_day:
                continue
            o, h, l, c = self.o15[k], self.h15[k], self.l15[k], self.c15[k]
            rng = h - l
            if rng <= 0:
                continue
            for d in (-1, 1):
                b = {tf: int(self.b15[tf][k]) for tf in HTF}
                if cfg.require_monthly and b["M"] != d:
                    continue
                if cfg.require_weekly and b["W"] != d:
                    continue
                agree = sum(b[tf] == d for tf in ("W", "D", "H4"))
                if agree < cfg.min_htf_agree:
                    continue
                tol = cfg.level_tol_atr * ctx.atr
                sig = None
                for setup, lvl, sr_conf, struggle in self.levels(ctx, d):
                    if d == -1:
                        ok = h >= lvl - tol and c < lvl and (h - max(o, c)) >= cfg.wick_ratio * rng and c < (h + l) / 2
                    else:
                        ok = l <= lvl + tol and c > lvl and (min(o, c) - l) >= cfg.wick_ratio * rng and c > (h + l) / 2
                    if ok:
                        sig = self._confirm(k, d, setup, lvl, ctx, b, agree, sr_conf, struggle)
                        if sig:
                            break
                if sig:
                    trades.append(sig)
                    per_day[pdi] = per_day.get(pdi, 0) + 1
                    busy = sig["_busy"]
                    break
        for tr in trades:
            tr.pop("_busy", None)
        return trades

    def _confirm(self, k, d, setup, lvl, ctx, b, agree, sr_conf, struggle):
        cfg, A, n5 = self.cfg, ctx.atr, len(self.e5)
        t = self.e15[k]
        j0 = int(np.searchsorted(self.e5, t, side="right"))
        ext = self.h15[k] if d == -1 else self.l15[k]
        tol = cfg.level_tol_atr * A
        for j in range(j0, min(j0 + cfg.confirm_bars, n5)):
            if j < 3:
                continue
            if d == -1:
                if self.h5[j] > ext + tol:
                    return None                      # rejection failed: price pushed through
                ext = max(ext, self.h5[j])
                trig = self.c5[j] < self.l5[j - 3:j].min()
            else:
                if self.l5[j] < ext - tol:
                    return None
                ext = min(ext, self.l5[j])
                trig = self.c5[j] > self.h5[j - 3:j].max()
            if trig:
                return self._build(k, j, d, setup, lvl, ctx, b, agree, sr_conf, struggle, ext)
        return None

    def _build(self, k, j, d, setup, lvl, ctx, b, agree, sr_conf, struggle, ext):
        cfg, A, sp, n5 = self.cfg, ctx.atr, self.cfg.spread, len(self.e5)
        e = j + 1
        live = e >= n5
        if not live and ctx.pdi + 1 < len(self.d_end) and self.e5[e] > self.d_end[ctx.pdi + 1]:
            return None                              # never carry a fresh entry across the daily close
        raw = self.c5[j] if live else self.o5[e]
        if d == -1:
            entry, sl = raw, ext + cfg.sl_buffer_atr * A + sp   # sell at bid, stop on ask
            risk = sl - entry
        else:
            entry, sl = raw + sp, ext - cfg.sl_buffer_atr * A   # buy at ask, stop on bid
            risk = entry - sl
        if not (cfg.min_risk_atr * A <= risk <= cfg.max_risk_atr * A):
            return None
        # targets: opposite side of yesterday / day-before, intraday extreme, opposing S/R zones
        ds = int(np.searchsorted(self.e5, self.d_end[ctx.pdi], side="right"))
        if d == -1:
            cands = [("PDL", ctx.pdl), ("PPDL", ctx.ppdl), ("Day low", self.l5[ds:j + 1].min())]
            cands += [("Support", z["hi"]) for z in ctx.zones]
            cands = [(nm, x, (entry - x) / risk) for nm, x in cands if x < entry]
        else:
            cands = [("PDH", ctx.pdh), ("PPDH", ctx.ppdh), ("Day high", self.h5[ds:j + 1].max())]
            cands += [("Resistance", z["lo"]) for z in ctx.zones]
            cands = [(nm, x, (x - entry) / risk) for nm, x in cands if x > entry]
        cands.sort(key=lambda x: x[2])
        if cands and cands[0][2] < cfg.obstacle_rr:
            return None                              # opposing level too close: poor trade location
        pick = next((x for x in cands if x[2] >= cfg.min_rr), None)
        rr = cfg.min_rr if pick is None else min(pick[2], cfg.max_rr)
        tp_kind = "fixed %.1fR" % rr if pick is None or pick[2] > cfg.max_rr else pick[0]
        tp = entry + d * rr * risk
        score = agree + int(sr_conf) + int(struggle)
        grade = "A" if score >= 4 else "B" if score == 3 else "C"
        sig = {
            "direction": "SELL" if d == -1 else "BUY", "setup": setup, "level": round(float(lvl), 2),
            "rejection_time": _iso(self.e15[k] - 900), "trigger_time": _iso(self.e5[j]),
            "entry_time": _iso(self.e5[j]) if live else _iso(self.e5[e] - 300),
            "entry": round(float(entry), 2), "sl": round(float(sl), 2), "tp": round(float(tp), 2),
            "risk": round(float(risk), 2), "rr": round(float(rr), 2), "tp_kind": tp_kind,
            "grade": grade, "score": score, "bias": b, "htf_agree": agree,
            "sr_confluence": bool(sr_conf), "struggle": bool(struggle), "atr_d": round(float(A), 2),
            "session": _session(int(self.e5[j])),
        }
        sig.update(self._simulate(d, e, entry, sl, tp, risk, ctx))
        return sig

    def _simulate(self, d, e, entry, sl, tp, risk, ctx):
        cfg, sp, n5 = self.cfg, self.cfg.spread, len(self.e5)
        day_end = self.d_end[ctx.pdi + 1] if ctx.pdi + 1 < len(self.d_end) else np.iinfo(np.int64).max
        be = False
        if e >= n5:
            return {"status": "ENTER NOW", "exit": None, "exit_time": None, "exit_reason": None,
                    "r": None, "be_moved": False, "_busy": np.iinfo(np.int64).max}
        for i in range(e, n5):
            if self.e5[i] > day_end:
                px = self.c5[i - 1] + (sp if d == -1 else 0)
                return self._closed(d, entry, px, risk, i - 1, "time (day close)", be)
            o, h, l = self.o5[i], self.h5[i], self.l5[i]
            if d == -1:
                ao, ah, al = o + sp, h + sp, l + sp
                if i > e and ao >= sl:
                    return self._closed(d, entry, ao, risk, i, "SL gap" if not be else "BE", be)
                if ah >= sl:
                    return self._closed(d, entry, sl, risk, i, "SL" if not be else "BE", be)
                if al <= tp:
                    return self._closed(d, entry, tp, risk, i, "TP", be)
                if cfg.be_at_r and not be and al <= entry - cfg.be_at_r * risk:
                    sl, be = entry, True
            else:
                if i > e and o <= sl:
                    return self._closed(d, entry, o, risk, i, "SL gap" if not be else "BE", be)
                if l <= sl:
                    return self._closed(d, entry, sl, risk, i, "SL" if not be else "BE", be)
                if h >= tp:
                    return self._closed(d, entry, tp, risk, i, "TP", be)
                if cfg.be_at_r and not be and h >= entry + cfg.be_at_r * risk:
                    sl, be = entry, True
        last = self.c5[-1] + (sp if d == -1 else 0)
        return {"status": "OPEN", "exit": None, "exit_time": None, "exit_reason": None,
                "r": None, "open_r": round(float((last - entry) * d / risk), 2), "current_sl": round(float(sl), 2),
                "be_moved": be, "_busy": np.iinfo(np.int64).max}

    def _closed(self, d, entry, px, risk, i, why, be):
        return {"status": "CLOSED", "exit": round(float(px), 2), "exit_time": _iso(self.e5[i]),
                "exit_reason": why, "r": round(float((px - entry) * d / risk), 3), "be_moved": be,
                "_busy": int(self.e5[i])}

    # ---------------------------------------------------------------- live snapshot
    def snapshot(self) -> dict:
        t = int(self.e5[-1])
        price = float(self.c5[-1])
        pdi = int(np.searchsorted(self.d_end, t, side="right") - 1)
        ctx = self.day_ctx(pdi)
        b = {tf: int(self.bias_at(tf, t)) for tf in HTF}
        detail = {}
        for tf in HTF:
            i = int(np.searchsorted(self.f[tf].end, t, side="right") - 1)
            _, ev, sv = self.bias[tf]
            detail[tf] = {"bias": b[tf], "ema_vote": int(ev[i]) if i >= 0 else 0,
                          "structure_vote": int(sv[i]) if i >= 0 else 0,
                          "bar": _iso(self.f[tf].end[i]) if i >= 0 else None}
        watch = []
        if ctx:
            for d in (-1, 1):
                if self.cfg.require_monthly and b["M"] != d:
                    continue
                agree = sum(b[tf] == d for tf in ("W", "D", "H4"))
                for setup, lvl, conf, strg in self.levels(ctx, d):
                    dist = (lvl - price) * (-d)          # >0: price still on the approach side
                    if -0.1 * ctx.atr <= dist <= 0.6 * ctx.atr:
                        watch.append({"direction": "SELL" if d == -1 else "BUY", "setup": setup,
                                      "level": round(float(lvl), 2), "distance": round(abs(lvl - price), 2),
                                      "distance_atr": round(abs(lvl - price) / ctx.atr, 2),
                                      "htf_agree": agree, "sr_confluence": bool(conf),
                                      "meets_filters": agree >= self.cfg.min_htf_agree and (not self.cfg.require_weekly or b["W"] == d)})
        watch.sort(key=lambda w: w["distance"])
        day = None
        if ctx:
            day = {"pdh": round(ctx.pdh, 2), "pdl": round(ctx.pdl, 2), "ppdh": round(ctx.ppdh, 2),
                   "ppdl": round(ctx.ppdl, 2), "atr": round(ctx.atr, 2), "struggle_high": ctx.struggle_high,
                   "struggle_low": ctx.struggle_low,
                   "zones": sorted(ctx.zones, key=lambda z: abs(z["mid"] - price))[:10]}
        return {"time": _iso(t), "price": round(price, 2), "bias": b, "bias_detail": detail, "day": day,
                "watch": watch[:8], "aligned": "BUY" if all(b[x] == 1 for x in ("M", "W", "D", "H4"))
                else "SELL" if all(b[x] == -1 for x in ("M", "W", "D", "H4")) else "MIXED"}

    def review(self, tr: dict) -> list[str]:
        """Intraday review notes for an open position."""
        notes, d = [], (1 if tr["direction"] == "BUY" else -1)
        t = int(self.e5[-1])
        now_b = {tf: int(self.bias_at(tf, t)) for tf in HTF}
        if tr.get("open_r") is not None:
            notes.append(f"Running {tr['open_r']:+.2f}R. Stop {tr.get('current_sl')} | target {tr['tp']}.")
        if tr.get("be_moved"):
            notes.append("+1R reached earlier: stop is at breakeven, trade is risk-free.")
        for tf in ("H4", "D"):
            if now_b[tf] == -d and tr["bias"].get(tf) != -d:
                notes.append(f"{tf} bias has flipped against the trade since entry - consider tightening or closing.")
        if now_b["M"] != d:
            notes.append("Monthly bias no longer supports this direction - exit rather than hold overnight.")
        pdi = int(np.searchsorted(self.d_end, t, side="right") - 1)
        if pdi + 1 < len(self.d_end):
            mins = (self.d_end[pdi + 1] - t) // 60
            if mins < 120:
                notes.append(f"{mins} min to the 17:00 New York daily close - the model exits at the close.")
        if not notes:
            notes.append("No change: higher timeframes still aligned. Let it work.")
        return notes


def _iso(s) -> str:
    return pd.Timestamp(int(s), unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _session(s: int) -> str:
    h = (s // 3600) % 24
    return "Asia" if h < 7 else "London" if h < 12 else "London/NY overlap" if h < 16 else "New York"
