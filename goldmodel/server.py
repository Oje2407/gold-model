"""Web app: live scanner + backtest dashboard.   python -m goldmodel serve   ->  http://localhost:8000"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import data
from .engine import HTF, Config, Engine, build_frames, ema
from .report import run_backtest, sweep

STATIC = Path(__file__).resolve().parent / "static"
CFG_FILE = data.DATA / "config.json"
TFS = {"M5": "M5", "M15": "M15", "H4": "H4", "D": "D", "W": "W", "M": "M"}


class Market:
    def __init__(self, inst: str, demo: bool = False):
        self.inst, self.demo = inst, demo
        self.lock = threading.Lock()
        self.m5 = self.h1 = self.d1 = None
        self.engine: Engine | None = None
        self.source, self.error, self.updated = "", None, None
        self.live: list[dict] = []
        self.bt = {"status": "idle"}

    def load(self):
        if self.demo:
            import sys
            sys.path.insert(0, str(data.ROOT))
            from tests.synthetic import make_m5
            end = pd.Timestamp.now(tz="UTC").floor("5min")
            self.m5 = make_m5(end=end.strftime("%Y-%m-%d %H:%M"), price=2400.0 if self.inst == "spot" else 2410.0, seed=3 if self.inst == "spot" else 4)
            self.h1 = self.d1 = None
            self.source = "DEMO (synthetic data - not real prices)"
        else:
            self.m5 = data.load(f"{self.inst}_m5")
            self.h1 = data.load(f"{self.inst}_h1")
            self.d1 = data.load(f"{self.inst}_d1")
            self.source = "local history"
        self.rebuild()

    def cfg(self) -> Config:
        allc = json.loads(CFG_FILE.read_text()) if CFG_FILE.exists() else {}
        c = Config.from_dict(allc.get(self.inst, {}))
        if self.inst not in allc or "spread" not in allc[self.inst]:
            c.spread = data.INSTRUMENTS[self.inst]["spread"]
        return c

    def frames(self):
        return build_frames(self.m5, self.h1, self.d1)

    def rebuild(self):
        if self.m5 is None or self.m5.empty:
            self.error = f"No {self.inst} data yet. Run:  python -m goldmodel download {self.inst}"
            return
        eng = Engine(self.frames(), self.cfg())
        t_from = int(eng.e5[-1]) - 3 * 86400
        live = eng.run(t_from)
        for tr in live:
            if tr["status"] in ("OPEN", "ENTER NOW"):
                tr["review"] = eng.review(tr)
        with self.lock:
            self.engine, self.live, self.error = eng, live, None
            self.updated = pd.Timestamp.now(tz="UTC").isoformat()

    def poll(self):
        if self.demo:
            return self.rebuild()
        try:
            self.m5, src = data.update_live(self.inst, self.m5)
            self.source = src
            data.save(self.m5, f"{self.inst}_m5")
            self.rebuild()
        except Exception as e:  # noqa: BLE001
            self.error = f"Live update failed ({e}). Showing last stored data."
            if self.engine is None:
                self.rebuild()

    def start_backtest(self, overrides: dict, years: float):
        if self.bt.get("status") == "running":
            return
        self.bt = {"status": "running", "started": time.time()}

        def job():
            try:
                f = self.frames()
                base = {**self.cfg().__dict__, **(overrides or {})}
                res = run_backtest(f, Config.from_dict(base), years)
                res["sweep"] = sweep(f, base, years)
                res["instrument"] = self.inst
                res["data_source"] = self.source
                self.bt = {"status": "done", "result": res, "seconds": round(time.time() - self.bt["started"], 1)}
                data.DATA.mkdir(parents=True, exist_ok=True)
                (data.DATA / f"{self.inst}_backtest.json").write_text(json.dumps(res, default=str))
            except Exception as e:  # noqa: BLE001
                self.bt = {"status": "error", "error": f"{e}", "trace": traceback.format_exc()[-2000:]}

        threading.Thread(target=job, daemon=True).start()


def _cloud_bootstrap(markets: dict, m5_days: int):
    import os

    """First start on a host (e.g. Render): download history in the background, then go live."""
    def boot(inst):
        m = markets[inst]
        if m.engine is not None:
            return
        m.error = "Loading market data - the first start takes a few minutes. This page refreshes itself."
        try:
            print(f"[bootstrap] {inst} starting", flush=True)
            if inst == "futures":
                res = data.download_history("futures", progress=lambda x: print(x, flush=True))
            else:
                fast = bool(os.environ.get("OANDA_API_TOKEN"))   # OANDA serves 2 years in about a minute
                start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=m5_days if fast else 45)).strftime("%Y-%m-%d")
                res = data.download_history("spot", "2015-01-01", start, progress=lambda x: print(x, flush=True))
            m.load()
            m.poll()
            print(f"[bootstrap] {inst} ready: {res}; last bar {m.m5.index[-1] if m.m5 is not None and len(m.m5) else None}; error={m.error}", flush=True)
        except Exception as e:  # noqa: BLE001
            m.error = f"Could not load {inst} data on this server ({e})."
            print(f"[bootstrap] {inst} FAILED: {e}", flush=True)

    # spot and futures load side by side, so a slow source never holds up the other
    threads = [threading.Thread(target=boot, args=(i,), daemon=True) for i in ("spot", "futures")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # extend spot 1-minute history for the 2-year backtest, in the background (Dukascopy only)
    m = markets["spot"]
    if not os.environ.get("OANDA_API_TOKEN") and m.m5 is not None and len(m.m5) and (m.m5.index[0] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=m5_days - 10)):
        try:
            old_start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=m5_days)).strftime("%Y-%m-%d")
            m1 = data.dukascopy("XAUUSD", "min", old_start, m.m5.index[0].strftime("%Y-%m-%d"), progress=lambda x: print(x, flush=True))
            m.m5 = data.merge(data.resample(m1, "5min"), m.m5)
            data.save(m.m5, "spot_m5")
            m.rebuild()
            print(f"[bootstrap] spot history extended to {m.m5.index[0]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[bootstrap] spot history extension failed: {e}", flush=True)


def create_app(demo: bool = False, poll_seconds: int = 60, cloud: bool = False, m5_days: int = 760) -> FastAPI:
    import os

    from fastapi import Request
    from fastapi.responses import HTMLResponse, RedirectResponse

    app = FastAPI(title="Gold top-down model")
    password = os.environ.get("APP_PASSWORD", "")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if not password or request.cookies.get("gm_key") == password:
            return await call_next(request)
        if request.query_params.get("key") == password:
            r = await call_next(request)
            r.set_cookie("gm_key", password, max_age=60 * 60 * 24 * 180, httponly=True, samesite="lax", secure=True)
            return r
        return HTMLResponse('<meta name=viewport content="width=device-width,initial-scale=1"><body style="font:16px system-ui;background:#0e0f11;color:#e7e4dc;padding:40px 20px"><h3>Gold model</h3><form><input name=key type=password placeholder="Password" style="font-size:16px;padding:10px;width:100%;max-width:320px"><br><br><button style="font-size:16px;padding:10px 18px">Open</button></form>', 401)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    markets = {k: Market(k, demo) for k in data.INSTRUMENTS}
    for m in markets.values():
        m.load()
    if cloud and not demo:
        threading.Thread(target=_cloud_bootstrap, args=(markets, m5_days), daemon=True).start()

    def loop():
        while True:
            time.sleep(poll_seconds)
            for m in markets.values():
                if m.m5 is not None and not m.m5.empty:
                    m.poll()

    threading.Thread(target=loop, daemon=True).start()

    def mk(inst) -> Market:
        if inst not in markets:
            raise HTTPException(404, "unknown instrument")
        return markets[inst]

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/instruments")
    def instruments():
        return [{"id": k, "label": v["label"], "has_data": markets[k].engine is not None} for k, v in data.INSTRUMENTS.items()]

    @app.get("/api/state")
    def state(inst: str = "spot"):
        m = mk(inst)
        if m.engine is None:
            return {"error": m.error, "instrument": inst}
        snap = m.engine.snapshot()
        return {"instrument": inst, "label": data.INSTRUMENTS[inst]["label"], "source": m.source,
                "updated": m.updated, "error": m.error, "demo": m.demo, **snap,
                "signals": list(reversed(m.live)), "config": m.engine.cfg.__dict__}

    @app.post("/api/refresh")
    def refresh(inst: str = "spot"):
        m = mk(inst)
        m.poll()
        return {"ok": True, "error": m.error}

    @app.get("/api/candles")
    def candles(inst: str = "spot", tf: str = "M15", n: int = 400):
        m = mk(inst)
        if m.engine is None or tf not in TFS:
            raise HTTPException(400, m.error or "bad timeframe")
        df = m.engine.f[tf].df
        c = df["close"].to_numpy()
        emas = {str(p): ema(c, p) for p in (8, 21, 50, 200)}
        sl = slice(max(0, len(df) - n), len(df))
        t = (df.index[sl].as_unit("s").asi8).tolist()
        cols = {k: np.round(df[k].to_numpy()[sl], 2).tolist() for k in ("open", "high", "low", "close")}
        bars = [{"time": t[i], "open": cols["open"][i], "high": cols["high"][i], "low": cols["low"][i],
                 "close": cols["close"][i]} for i in range(len(t))]
        lines = {p: [{"time": t[i], "value": round(float(v[sl][i]), 2)} for i in range(len(t))] for p, v in emas.items()}
        b = m.engine.bias[tf][0] if tf in HTF else None
        return {"bars": bars, "ema": lines, "bias": b[sl].tolist() if b is not None else None}

    @app.get("/api/config")
    def get_config(inst: str = "spot"):
        return mk(inst).cfg().__dict__

    @app.post("/api/config")
    def set_config(inst: str = "spot", body: dict = Body(...)):
        mk(inst)
        allc = json.loads(CFG_FILE.read_text()) if CFG_FILE.exists() else {}
        allc[inst] = Config.from_dict(body).__dict__
        data.DATA.mkdir(parents=True, exist_ok=True)
        CFG_FILE.write_text(json.dumps(allc, indent=2))
        markets[inst].rebuild()
        return allc[inst]

    @app.post("/api/backtest")
    def backtest(inst: str = "spot", years: float = 2.0, body: dict = Body(default={})):
        m = mk(inst)
        if m.engine is None:
            raise HTTPException(400, m.error)
        m.start_backtest(body, years)
        return {"status": m.bt["status"]}

    @app.get("/api/backtest")
    def backtest_result(inst: str = "spot"):
        m = mk(inst)
        if m.bt.get("status") == "idle":
            p = data.DATA / f"{inst}_backtest.json"
            if p.exists() and not m.demo:
                return {"status": "done", "result": json.loads(p.read_text()), "cached": True}
        return m.bt

    return app
