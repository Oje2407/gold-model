"""Command line.

  python -m goldmodel download spot            # Dukascopy XAUUSD: hourly since 2015 + 1-minute for ~2.3 years
  python -m goldmodel download futures         # Yahoo GC=F (needs: pip install yfinance)
  python -m goldmodel import-csv futures m5 path/to/GC_M5.csv [--tz Europe/London]
  python -m goldmodel backtest spot [--years 2]
  python -m goldmodel serve [--port 8000] [--demo] [--poll 60]
"""
import argparse
import json

from . import data


def main():
    p = argparse.ArgumentParser(prog="goldmodel")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("inst", choices=list(data.INSTRUMENTS))
    d.add_argument("--h1-start", default="2015-01-01")
    d.add_argument("--m5-start", default=None)
    c = sub.add_parser("import-csv")
    c.add_argument("inst", choices=list(data.INSTRUMENTS))
    c.add_argument("tf", choices=["m5", "h1", "d1"])
    c.add_argument("path")
    c.add_argument("--tz", default="UTC", help="timezone of the CSV timestamps, e.g. Europe/London or EET for MT5")
    b = sub.add_parser("backtest")
    b.add_argument("inst", choices=list(data.INSTRUMENTS))
    b.add_argument("--years", type=float, default=2.0)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--demo", action="store_true", help="synthetic data, to try the app before downloading")
    s.add_argument("--poll", type=int, default=60, help="seconds between live updates")
    s.add_argument("--cloud", action="store_true", help="hosted mode: download data on start, listen on $PORT")
    a = p.parse_args()

    if a.cmd == "download":
        print(data.download_history(a.inst, a.h1_start, a.m5_start))
    elif a.cmd == "import-csv":
        df = data.import_csv(a.path, tz=a.tz)
        if a.tf == "m5":
            df = data.resample(df, "5min")
        name = f"{a.inst}_{a.tf}"
        print(f"{len(df)} bars {df.index[0]} -> {df.index[-1]} saved to", data.save(data.merge(data.load(name), df), name))
    elif a.cmd == "backtest":
        from .engine import Config, build_frames
        from .report import run_backtest, sweep
        from .server import Market

        m = Market(a.inst)
        m.load()
        if m.m5 is None or m.m5.empty:
            raise SystemExit(m.error)
        f = build_frames(m.m5, m.h1, m.d1)
        cfg = m.cfg()
        r = run_backtest(f, cfg, a.years)
        print(json.dumps(r["periods"], indent=2))
        print(r["regime_note"])
        print("\nBy setup (last 2 years):")
        for row in r["by_setup"]["last_2_years"]:
            print(f"  {row['key']:<12} {row['trades']:>4} trades  win {row.get('win_rate', 0):>5}%  avg {row.get('avg_r', 0):+.2f}R")
        print("\nRule variants:")
        for row in sweep(f, cfg.__dict__, a.years):
            print(" ", row["label"], {k: v for k, v in row.items() if k != "label"})
    elif a.cmd == "serve":
        import uvicorn

        from .server import create_app

        import os

        if a.cloud:
            a.host, a.port = "0.0.0.0", int(os.environ.get("PORT", a.port))
        print(f"Open http://{a.host}:{a.port}")
        app = create_app(a.demo, a.poll, cloud=a.cloud, m5_days=int(os.environ.get("M5_DAYS", "760")))
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
