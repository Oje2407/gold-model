"""Run: python -m pytest -q tests   (or: python tests/test_engine.py)

The key test is the look-ahead check: every signal found on the full history must be found
identically when the model only has data up to that signal's entry bar.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldmodel.engine import Config, Engine, build_frames  # noqa: E402
from goldmodel.report import run_backtest  # noqa: E402
from tests.synthetic import make_m5  # noqa: E402

M5 = make_m5()
START = int(pd.Timestamp("2024-11-01", tz="UTC").timestamp())


def test_no_lookahead():
    full = [t for t in Engine(build_frames(M5), Config()).run(START) if t["status"] == "CLOSED"]
    assert len(full) > 50
    rng = np.random.default_rng(1)
    sample = [full[i] for i in sorted(rng.choice(len(full), 25, replace=False))]
    for t in sample:
        cut = pd.Timestamp(t["entry_time"]) + pd.Timedelta(minutes=5)   # data ends with the entry bar
        eng = Engine(build_frames(M5[M5.index < cut]), Config())
        again = [x for x in eng.run(START) if x["rejection_time"] == t["rejection_time"]]
        assert again, f"signal at {t['rejection_time']} disappeared with truncated data"
        a = again[0]
        for k in ("direction", "setup", "level", "entry", "sl", "tp", "grade", "bias"):
            assert a[k] == t[k], (k, a[k], t[k], t["rejection_time"])


def test_htf_bars_only_used_when_closed():
    f = build_frames(M5)
    eng = Engine(f, Config())
    for tf in ("M", "W", "D", "H4"):
        end = f[tf].end
        t = end[len(end) // 2]
        # one second before the bar closes, the model must still be on the previous bar
        i_before = np.searchsorted(end, t - 1, side="right") - 1
        i_at = np.searchsorted(end, t, side="right") - 1
        assert i_at == i_before + 1, tf
        assert eng.bias_at(tf, t - 1) == eng.bias[tf][0][i_before]


def test_trade_geometry_and_rr():
    cfg = Config()
    for t in Engine(build_frames(M5), cfg).run(START):
        if t["direction"] == "SELL":
            assert t["sl"] > t["entry"] > t["tp"]
        else:
            assert t["sl"] < t["entry"] < t["tp"]
        assert cfg.min_rr - 1e-6 <= t["rr"] <= cfg.max_rr + 1e-6
        assert t["bias"]["M"] == (1 if t["direction"] == "BUY" else -1)   # monthly direction rule
        if t["r"] is not None:
            assert t["r"] >= -1.0 - 1e-6 or t["exit_reason"] == "SL gap"
            assert t["r"] <= t["rr"] + 0.01  # rr is rounded to 2dp


def test_report_periods():
    r = run_backtest(build_frames(M5), Config())
    assert set(r["periods"]) == {"last_2_years", "2026_ytd", "2025"}
    assert r["periods"]["last_2_years"]["trades"] >= r["periods"]["2026_ytd"]["trades"]


def test_dukascopy_decoder_roundtrip():
    import lzma
    import struct
    from goldmodel.data import decode_bi5

    recs = b"".join(struct.pack(">IIIIIf", 60 * i, 2650120 + i, 2650500, 2649900, 2650800, 12.5) for i in range(3))
    raw = lzma.compress(recs, format=lzma.FORMAT_ALONE)
    df = decode_bi5(raw, pd.Timestamp("2026-01-05", tz="UTC"), 1000.0)
    assert len(df) == 3 and df.index[1] == pd.Timestamp("2026-01-05 00:01", tz="UTC")
    assert abs(df["open"].iloc[0] - 2650.12) < 1e-9 and df["high"].iloc[0] == 2650.8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
