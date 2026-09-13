"""
NSE Swing-Trade Scanner
========================
Downloads NSE's daily Bhavcopy (official free EOD data for every traded
stock), maintains a rolling local history, computes technical indicators,
and ranks stocks with a composite "swing score".

Run this once a day, after market close (e.g. via the GitHub Actions
workflow in .github/workflows/daily-scan.yml). It writes results.json,
which the website (index.html) reads.

Usage:
    python scanner.py
"""

import io
import json
import math
import zipfile
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config — tune these to change what counts as a "good" swing candidate
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "history.parquet"
SECTORS_FILE = DATA_DIR / "sectors.json"
RESULTS_FILE = Path("results.json")

MIN_PRICE = 20                  # exclude penny stocks
MIN_AVG_TURNOVER_20D = 5_00_00_000   # ₹5 crore/day min liquidity
MIN_HISTORY_DAYS = 210          # need ~200 trading days for SMA200
MAX_PCT_ABOVE_SMA50 = 25        # hard exclude: more than this far above the 50-day
                                 # average is "chasing," regardless of how good the
                                 # rest of the score looks — the soft extension_score
                                 # penalty alone wasn't enough to stop a fresh spike
                                 # from still ranking #1 on trend+momentum strength
TOP_N = 150                     # how many stocks to publish — kept generous since
                                 # the site lets you filter by price/sector afterward

WEIGHTS = {
    "trend": 0.25,
    "momentum": 0.20,
    "volume": 0.15,
    "delivery": 0.15,       # NEW: % of volume actually taken delivery, not just intraday churn
    "extension": 0.10,      # NEW: penalizes stocks already too far above their 50-day average
    "volatility": 0.10,
    "relative_strength": 0.05,
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# 1. Download today's bhavcopy from NSE
# ---------------------------------------------------------------------------

def new_session() -> requests.Session:
    """Create and warm up a session (NSE requires a homepage visit first
    to set cookies before any file download will succeed)."""
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://www.nseindia.com", timeout=15)
    return session


def fetch_bhavcopy(target_date: dt.date, session: requests.Session | None = None) -> pd.DataFrame | None:
    """Fetch NSE's daily CM (equities) bhavcopy for a given date.
    Returns None if that date has no file (weekend/holiday).

    Pass an existing `session` (from new_session()) when calling this many
    times in a row — e.g. during a backfill — to avoid re-authenticating
    on every single request."""
    owns_session = session is None
    if owns_session:
        session = new_session()

    date_str = target_date.strftime("%Y%m%d")
    url = (
        "https://nsearchives.nseindia.com/content/cm/"
        f"BhavCopy_NSE_CM_0_0_0_{date_str}_F_0000.csv.zip"
    )
    resp = session.get(url, timeout=20)
    if resp.status_code != 200 or len(resp.content) < 1000:
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    df.columns = [c.strip() for c in df.columns]
    # Keep only normal-series equities (excludes illiquid trade-to-trade series)
    if "SctySrs" in df.columns:
        df = df[df["SctySrs"] == "EQ"]

    keep = {
        "TckrSymb": "symbol",
        "OpnPric": "open",
        "HghPric": "high",
        "LwPric": "low",
        "ClsPric": "close",
        "TtlTradgVol": "volume",
        "TtlTrfVal": "turnover",
    }
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected bhavcopy format, missing columns: {missing}")

    df = df[list(keep)].rename(columns=keep)
    df["date"] = pd.Timestamp(target_date)
    return df.reset_index(drop=True)


def fetch_delivery(target_date: dt.date, session: requests.Session | None = None) -> pd.DataFrame | None:
    """Fetch delivery % data for a given date from NSE's 'full bhavcopy' file
    (a separate, older-format file from the OHLCV one — it's the only place
    NSE publishes DELIV_PER, the % of traded volume actually taken delivery
    rather than squared off intraday). Same archive domain, same reliability
    as the OHLCV download."""
    owns_session = session is None
    if owns_session:
        session = new_session()

    date_str = target_date.strftime("%d%m%Y")
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
    resp = session.get(url, timeout=20)
    if resp.status_code != 200 or len(resp.content) < 500:
        return None

    df = pd.read_csv(io.BytesIO(resp.content))
    # This file is notorious for stray whitespace in both headers and values
    df.columns = [c.strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}
    symbol_col, series_col, deliv_col = cols.get("symbol"), cols.get("series"), cols.get("deliv_per")
    if not symbol_col or not deliv_col:
        return None

    if series_col:
        df = df[df[series_col].astype(str).str.strip() == "EQ"]

    out = pd.DataFrame({
        "symbol": df[symbol_col].astype(str).str.strip(),
        "delivery_pct": pd.to_numeric(df[deliv_col].astype(str).str.strip(), errors="coerce"),
    })
    return out.reset_index(drop=True)


def fetch_day(target_date: dt.date, session: requests.Session | None = None) -> pd.DataFrame | None:
    """Fetch one day's OHLCV + delivery % together. Returns None if the
    OHLCV file is missing (holiday/weekend). If delivery data specifically
    is missing but OHLCV isn't, delivery_pct is just left as NaN for that
    day rather than failing the whole day."""
    owns_session = session is None
    if owns_session:
        session = new_session()

    ohlcv = fetch_bhavcopy(target_date, session=session)
    if ohlcv is None or ohlcv.empty:
        return None

    try:
        deliv = fetch_delivery(target_date, session=session)
    except Exception:
        deliv = None

    if deliv is not None and not deliv.empty:
        ohlcv = ohlcv.merge(deliv, on="symbol", how="left")
    else:
        ohlcv["delivery_pct"] = np.nan
    return ohlcv


def fetch_latest_bhavcopy(max_lookback_days: int = 6) -> pd.DataFrame:
    """Try today, then walk backwards until a trading day is found."""
    session = new_session()
    d = dt.date.today()
    for _ in range(max_lookback_days):
        df = fetch_day(d, session=session)
        if df is not None and not df.empty:
            return df
        d -= dt.timedelta(days=1)
    raise RuntimeError("Could not fetch a bhavcopy in the last week — NSE may be blocking this environment.")


# ---------------------------------------------------------------------------
# 2. Maintain rolling history
# ---------------------------------------------------------------------------

def update_history(new_day: pd.DataFrame) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    if HISTORY_FILE.exists():
        hist = pd.read_parquet(HISTORY_FILE)
        hist = hist[hist["date"] != new_day["date"].iloc[0]]  # avoid dupes on re-run
        hist = pd.concat([hist, new_day], ignore_index=True)
    else:
        hist = new_day

    # Keep a bounded window (~300 trading days ≈ 14 months) so the file doesn't grow forever
    cutoff = hist["date"].max() - pd.Timedelta(days=460)
    hist = hist[hist["date"] >= cutoff]
    hist.to_parquet(HISTORY_FILE, index=False)
    return hist


# ---------------------------------------------------------------------------
# 3. Indicators (hand-rolled, no external TA dependency)
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


# ---------------------------------------------------------------------------
# 4. Scoring
# ---------------------------------------------------------------------------

def score_symbol(g: pd.DataFrame) -> dict | None:
    g = g.sort_values("date").reset_index(drop=True)
    if len(g) < MIN_HISTORY_DAYS:
        return None

    close = g["close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    r = rsi(close)
    macd_line, signal_line = macd(close)
    a = atr(g)
    vol_avg20 = g["volume"].rolling(20).mean()
    turnover_avg20 = (g["close"] * g["volume"]).rolling(20).mean()

    last = -1
    px = close.iloc[last]
    if px < MIN_PRICE or turnover_avg20.iloc[last] < MIN_AVG_TURNOVER_20D:
        return None
    if pd.isna(sma200.iloc[last]):
        return None

    # --- Trend score (30%) ---
    above_sma50 = px > sma50.iloc[last]
    sma50_above_sma200 = sma50.iloc[last] > sma200.iloc[last]
    sma50_rising = sma50.iloc[last] > sma50.iloc[last - 10]
    trend_score = above_sma50 * 40 + sma50_above_sma200 * 35 + sma50_rising * 25

    # --- Momentum score (25%): RSI sweet spot ~55-65, plus MACD confirmation ---
    rsi_val = r.iloc[last]
    rsi_score = max(0, 100 - abs(rsi_val - 60) * 4) if not pd.isna(rsi_val) else 0
    macd_bullish = macd_line.iloc[last] > signal_line.iloc[last]
    macd_above_zero = macd_line.iloc[last] > 0
    macd_bonus = 15 if (macd_bullish and macd_above_zero) else (5 if macd_bullish else 0)
    momentum_score = min(100, rsi_score * 0.8 + macd_bonus)

    # --- Volume confirmation score (20%) ---
    vol_ratio = g["volume"].iloc[last] / vol_avg20.iloc[last] if vol_avg20.iloc[last] else 0
    volume_score = np.clip((vol_ratio - 1) * 50 + 50, 0, 100)

    # --- Volatility score (15%): sweet spot ATR 2-5% of price ---
    atr_pct = (a.iloc[last] / px) * 100 if px else 0
    if 2 <= atr_pct <= 5:
        volatility_score = 100
    else:
        dist = min(abs(atr_pct - 2), abs(atr_pct - 5)) if atr_pct < 2 or atr_pct > 5 else 0
        volatility_score = max(0, 100 - dist * 20)

    # --- Delivery score (15%): % of volume actually taken delivery, not
    # just squared off intraday. Higher = more conviction behind the move,
    # not just leveraged/speculative churn. Neutral (50) when unavailable
    # (older backfilled days, or NSE's delivery file missing for that date).
    deliv_pct = g["delivery_pct"].iloc[last] if "delivery_pct" in g.columns else np.nan
    if pd.isna(deliv_pct):
        delivery_score = 50.0
    else:
        delivery_score = float(np.clip(deliv_pct * 1.4, 0, 100))

    # --- Extension score (10%): penalizes stocks already too far above
    # their 50-day average — chasing an overextended move is a common way
    # momentum strategies lose money on the pullback.
    pct_above_sma50 = (px / sma50.iloc[last] - 1) * 100 if sma50.iloc[last] else 0
    if pct_above_sma50 > MAX_PCT_ABOVE_SMA50:
        # Hard cutoff, not just a scoring penalty — a stock that just spiked
        # this far shouldn't be able to buy its way into the rankings via a
        # strong trend/momentum score from that same spike.
        return None
    if pct_above_sma50 <= 8:
        extension_score = 100.0
    else:
        extension_score = max(0.0, 100 - (pct_above_sma50 - 8) * 6)

    # --- Relative strength (5%): 20-day return, ranked later against peers ---
    ret_20d = (px / close.iloc[last - 20] - 1) * 100 if len(g) > 20 else 0

    return {
        "symbol": g["symbol"].iloc[0],
        "price": round(float(px), 2),
        "trend_score": round(float(trend_score), 1),
        "momentum_score": round(float(momentum_score), 1),
        "volume_score": round(float(volume_score), 1),
        "volatility_score": round(float(volatility_score), 1),
        "delivery_score": round(delivery_score, 1),
        "delivery_pct": round(float(deliv_pct), 1) if not pd.isna(deliv_pct) else None,
        "extension_score": round(extension_score, 1),
        "pct_above_sma50": round(float(pct_above_sma50), 1),
        "return_20d_pct": round(float(ret_20d), 2),
        "rsi": round(float(rsi_val), 1) if not pd.isna(rsi_val) else None,
        "atr_pct": round(float(atr_pct), 2),
        "vol_vs_avg20": round(float(vol_ratio), 2),
    }


def json_safe(obj):
    """Recursively replace NaN/Infinity with None. Python's json module
    happily writes bare `NaN` by default, but that's not valid JSON per
    spec — browsers' JSON.parse() rejects it outright. Fields like
    delivery_pct or rsi can be NaN on days where that data wasn't
    available, so this has to run before every json.dumps() of results."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def load_sectors() -> dict:
    if SECTORS_FILE.exists():
        return json.loads(SECTORS_FILE.read_text())
    return {}


def build_rankings(hist: pd.DataFrame) -> list[dict]:
    rows = []
    for _, g in hist.groupby("symbol"):
        result = score_symbol(g)
        if result:
            rows.append(result)

    if not rows:
        return []

    df = pd.DataFrame(rows)
    # Relative strength = percentile rank of 20-day return across today's universe
    df["rs_score"] = df["return_20d_pct"].rank(pct=True) * 100

    df["composite_score"] = (
        df["trend_score"] * WEIGHTS["trend"]
        + df["momentum_score"] * WEIGHTS["momentum"]
        + df["volume_score"] * WEIGHTS["volume"]
        + df["delivery_score"] * WEIGHTS["delivery"]
        + df["extension_score"] * WEIGHTS["extension"]
        + df["volatility_score"] * WEIGHTS["volatility"]
        + df["rs_score"] * WEIGHTS["relative_strength"]
    ).round(1)

    df = df.sort_values("composite_score", ascending=False).head(TOP_N)

    # Attach sector/industry, if build_sectors.py has been run. Symbols not
    # yet classified (new listings, or sectors.json not built) get "Unknown"
    # rather than being dropped, so the site still shows them.
    sectors = load_sectors()
    records = df.to_dict(orient="records")
    for r in records:
        info = sectors.get(r["symbol"], {})
        r["sector"] = info.get("sector", "Unknown")
        r["industry"] = info.get("industry", "Unknown")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Fetching latest bhavcopy...")
    today_df = fetch_latest_bhavcopy()
    as_of = str(today_df["date"].iloc[0].date())
    print(f"Got {len(today_df)} equity rows for {as_of}")

    print("Updating rolling history...")
    hist = update_history(today_df)
    print(f"History now spans {hist['date'].nunique()} trading days")

    print("Scoring universe...")
    rankings = build_rankings(hist)
    print(f"{len(rankings)} stocks passed filters and were scored")

    output = {
        "as_of": as_of,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "methodology": {
            "weights": WEIGHTS,
            "filters": {
                "min_price": MIN_PRICE,
                "min_avg_turnover_20d": MIN_AVG_TURNOVER_20D,
                "min_history_days": MIN_HISTORY_DAYS,
            },
        },
        "results": rankings,
    }
    RESULTS_FILE.write_text(json.dumps(json_safe(output), indent=2))
    print(f"Wrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()