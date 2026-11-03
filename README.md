# Gold Top-Down Model (XAUUSD spot + GC futures)

A rules-based model of a top-down gold strategy, with a live scanner and a backtester, served as a web page on your own computer.

## What it does

**Direction (closed bars only).** Monthly, Weekly, Daily and 4-hour are each scored bull / bear / neutral from two votes: the EMA stack (close vs EMA21 vs EMA50) and market structure (HH+HL vs LH+LL, with a close through the last swing counting as a break of structure). By default a trade must follow the **monthly** bias, and at least one of W / D / H4 must agree. Making the weekly compulsory is one switch on the Rules tab.

**Daily-candle setup.**
* **Sell at PDH:** yesterday struggled at the day-before's high (it got within 0.25 ATR and closed back below). Today price returns to yesterday's high, an **M15** candle wicks into it and closes back below, and an **M5** candle then breaks the last three M5 lows.
* **Buy at PDL:** the mirror image at yesterday's low, wick included.

**Support / resistance.** Swing pivots from daily and H4 bars are clustered into zones and given a strength score. Zones are used in three ways: as confluence (PDH inside a zone raises the grade), as a setup in their own right (SR_RESIST / SR_SUPPORT), and as targets.

**Risk.**
* Stop goes beyond the rejection wick plus a buffer.
* Target is the first structural level (PDL/PDH, the day's extreme, or an opposing zone) that gives at least 1:2. If none exists, a fixed 2R target is used.
* The trade is skipped if an opposing level sits within 1R.
* Stop moves to breakeven at +1R.
* Any open trade is exited at the 17:00 New York daily close.
* Spread is charged on every trade.

**Grades.** A, B or C, based on how many timeframes agree plus S/R confluence plus the struggle filter.

**Intraday review.** For an open trade the page shows:
* the running R and current stop;
* whether H4, Daily or Monthly have flipped against the position;
* the time left to the daily close.

**Backtest.**
* Results are reported separately for the **last 2 years**, **this year-to-date** and **last year**, with a note on whether the edge is weakening.
* Breakdowns by setup, session, grade, direction and month.
* A **rule-variant table** runs the same data with weekly required, full alignment, no S/R and 1:3 R:R, so you can see which version suits the current market.

**No look-ahead.** `tests/test_engine.py` re-runs the model on data cut off at every sampled entry and checks that each signal comes out identical.

## Setup (Python 3.10+)

```bash
cd gold-model
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m goldmodel serve --demo            # try the page on synthetic data: http://127.0.0.1:8000
```

## Real data

**Recommended for spot: OANDA.** Create a free OANDA demo (practice) account, generate an API token under *Manage API Access*, and set `OANDA_API_TOKEN` (add `OANDA_ENV=live` only for a live-account token). The model then loads XAU_USD hourly history since 2015 and 2 years of 5-minute bid candles in about a minute, and polls live 5-minute candles. Dukascopy remains the fallback when no token is set, but it throttles cloud servers heavily.


```bash
python -m goldmodel download spot           # Dukascopy XAUUSD: hourly since 2015 + 1-minute for ~2.3 years (first run 10-30 min, cached after)
python -m goldmodel download futures        # Yahoo GC=F: daily history, 730 days hourly, but only 60 days of 5-minute
python -m goldmodel backtest spot           # prints the period comparison and rule variants
python -m goldmodel serve                   # live scanner, polls every 60 s
```

**Live spot prices.** Without a key, the app uses Dukascopy, which can lag by a few minutes. For timelier prices, set a Twelve Data key (free tier) before `serve`:
`export TWELVE_DATA_API_KEY=...` (Windows: `set TWELVE_DATA_API_KEY=...`). At a 60-second poll that is about 1,440 calls a day, more than the free tier's daily allowance. Use `--poll 120` or slower on the free tier.

**Futures backtest over 2 years** needs your own 5-minute history, because Yahoo only keeps 60 days. Export M5 GC data from TradingView, NinjaTrader or a broker, then:
`python -m goldmodel import-csv futures m5 GC_M5.csv --tz America/New_York`
For MT5 exports the timezone is usually the broker's server time, e.g. `--tz EET`.

## Files

| File | Purpose |
|---|---|
| `goldmodel/engine.py` | timeframes, bias, S/R zones, setups, trade simulation, live snapshot and review |
| `goldmodel/data.py` | Dukascopy / Yahoo / Twelve Data / CSV loaders |
| `goldmodel/report.py` | statistics, period comparison, rule variants |
| `goldmodel/server.py` + `static/index.html` | web page and API |
| `data/config.json` | rules saved from the Rules tab, kept per instrument |

## Caveats

* This is a research and alerting tool. It does not place trades.
* The backtest fills at the next M5 open plus spread. It checks the stop before the target within a bar, which is conservative, but real slippage around news will be worse.
* Treat small samples (under about 30 trades in a period) as anecdotes.
* Tuning the rules until the 2-year table looks good is curve-fitting. Prefer rules that hold up in **both** periods.
* Dukascopy's file layout is reverse-engineered. If a download returns odd prices, the loader stops with an error rather than using them.
