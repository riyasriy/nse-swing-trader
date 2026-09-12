"""
Sector/Industry mapping builder for the NSE Swing Scanner (v2)
==================================================================
The previous approach (hitting nseindia.com/api/quote-equity per symbol)
gets blocked by NSE's bot protection on that live API — the 403 "Access
Denied" errors you saw are from that, not from a network/ISP issue.

This version instead downloads NSE's official sectoral & thematic index
membership lists — plain CSV files, published on the SAME archive domain
(nsearchives.nseindia.com) your bhavcopy download already uses successfully.
No blocked API involved.

Coverage note: this only classifies stocks that belong to one of NSE's
~25 sectoral/thematic indices (roughly 500-700 of the more liquid, widely
traded names) — not all ~2959 listed stocks. That's a reasonable trade-off
here: your scanner already filters out illiquid stocks (under ₹5 crore/day
turnover), and there's heavy overlap between "liquid enough to trade" and
"a member of at least one sectoral index." Anything not covered just shows
as "Unknown" on the site, same as before.

Usage:
    python build_sectors.py
"""

import io
import json
import time

import pandas as pd
import requests

import scanner

SECTORS_FILE = scanner.DATA_DIR / "sectors.json"
REQUEST_DELAY_SECONDS = 0.5

# filename -> friendly sector label. Not every filename is guaranteed to
# exist (NSE adds/renames indices occasionally) — 404s are skipped silently,
# they just mean that particular list isn't available right now.
SECTOR_INDEX_FILES = {
    "ind_niftyautolist.csv": "Automobile",
    "ind_niftybanklist.csv": "Banking",
    "ind_niftyfinancelist.csv": "Financial Services",
    "ind_niftyfmcglist.csv": "FMCG",
    "ind_niftyitlist.csv": "IT",
    "ind_niftymetallist.csv": "Metals & Mining",
    "ind_niftypharmalist.csv": "Pharma",
    "ind_niftyrealtylist.csv": "Realty",
    "ind_niftyenergylist.csv": "Energy",
    "ind_niftymedialist.csv": "Media",
    "ind_niftypsubanklist.csv": "PSU Bank",
    "ind_niftyprivatebanklist.csv": "Private Bank",
    "ind_niftyhealthcarelist.csv": "Healthcare",
    "ind_niftyconsumerdurableslist.csv": "Consumer Durables",
    "ind_niftyoilgaslist.csv": "Oil & Gas",
    "ind_niftyinfralist.csv": "Infrastructure",
    "ind_niftypselist.csv": "Public Sector Enterprises",
    "ind_niftycpselist.csv": "Central PSE",
    "ind_niftycommoditieslist.csv": "Commodities",
    "ind_niftymnclist.csv": "MNC",
    "ind_niftyconsumptionlist.csv": "Consumption",
    "ind_niftyserv_sectorlist.csv": "Services",
    "ind_niftyindustrlist.csv": "Industrials",
    "ind_niftychemicalslist.csv": "Chemicals",
    "ind_niftycapitalmarketslist.csv": "Capital Markets",
    "ind_niftynon_cyclicalconsumerlist.csv": "Non-Cyclical Consumer",
    "ind_niftymobilitylist.csv": "Mobility",
    "ind_niftytourismlist.csv": "Tourism",
    "ind_niftyipolist.csv": "Recent IPOs",
}

BASE_URL = "https://nsearchives.nseindia.com/content/indices/"


def fetch_index_list(session, filename: str) -> pd.DataFrame | None:
    resp = session.get(BASE_URL + filename, timeout=20)
    if resp.status_code != 200 or len(resp.content) < 50:
        return None
    try:
        df = pd.read_csv(io.BytesIO(resp.content))
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    return df


def build():
    session = scanner.new_session()
    mapping = {}
    found_files = 0
    missing_files = []

    for filename, sector_label in SECTOR_INDEX_FILES.items():
        try:
            df = fetch_index_list(session, filename)
        except Exception as e:
            print(f"  {filename:45s} ERROR: {e}")
            df = None

        if df is None:
            missing_files.append(filename)
            print(f"  {filename:45s} not available, skipping")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        # These CSVs typically have a 'Symbol' column and often an
        # 'Industry' column with a finer sub-classification than the
        # index name itself — prefer that when present.
        symbol_col = next((c for c in df.columns if c.lower() == "symbol"), None)
        industry_col = next((c for c in df.columns if c.lower() == "industry"), None)

        if symbol_col is None:
            print(f"  {filename:45s} unexpected format (no Symbol column), skipping")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        count = 0
        for _, row in df.iterrows():
            symbol = str(row[symbol_col]).strip()
            if not symbol or symbol.lower() == "nan":
                continue
            industry = str(row[industry_col]).strip() if industry_col and pd.notna(row[industry_col]) else sector_label
            # Don't overwrite a symbol already classified by an earlier,
            # more specific list (e.g. Private Bank before generic Banking).
            if symbol not in mapping:
                mapping[symbol] = {"sector": sector_label, "industry": industry}
                count += 1

        found_files += 1
        print(f"  {filename:45s} +{count} symbols  ({sector_label})")
        time.sleep(REQUEST_DELAY_SECONDS)

    scanner.DATA_DIR.mkdir(exist_ok=True)
    SECTORS_FILE.write_text(json.dumps(mapping, indent=2, sort_keys=True))

    print(f"\n{found_files}/{len(SECTOR_INDEX_FILES)} index lists downloaded successfully, "
          f"{len(missing_files)} not available (renamed/retired — safe to ignore).")
    print(f"{len(mapping)} unique symbols classified. Saved to {SECTORS_FILE}")
    print("Re-run scanner.py (or backfill.py) to refresh results.json with sector data included.")


if __name__ == "__main__":
    build()