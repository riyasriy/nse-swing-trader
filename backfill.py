"""
Backfill history for the NSE Swing Scanner
============================================
Downloads roughly a year of NSE bhavcopy files in one run and builds
data/history.parquet, so scanner.py has enough history (200+ trading days)
to score stocks right away — instead of waiting ~9-10 months for the daily
workflow to accumulate it one day at a time.

Resumable: dates already saved in data/history.parquet are skipped, and
progress is checkpointed every CHECKPOINT_EVERY successful days — so if it
gets interrupted (network blip, rate limit, Ctrl+C), just re-run it and it
picks up where it left off instead of starting over.

Run this once, locally or in a one-off GitHub Actions run, then commit the
resulting data/history.parquet (and results.json) to your repo. After that,
the normal daily-scan.yml workflow just adds one more day at a time.

Usage:
    python backfill.py                 # last 400 calendar days (~year, with buffer)
    python backfill.py --days 600      # go back further
"""

import argparse
import datetime as dt
import json
import sys
import time

import pandas as pd

import scanner  # reuses fetch_bhavcopy, build_rankings, WEIGHTS, etc.

REQUEST_DELAY_SECONDS = 1.0    # be polite to NSE's servers between requests
MAX_CONSECUTIVE_FAILURES = 15  # abort early if something's systematically broken (e.g. blocked)
CHECKPOINT_EVERY = 15          # save progress to disk every N successful days


def daterange_weekdays(start: dt.date, end: dt.date):
    """Yield weekdays only (Mon-Fri) from start to end inclusive.
    Saves needless requests for guaranteed-closed days; actual market
    holidays within weekdays are simply skipped when the download 404s."""
    d = start
    while d <= end:
        if d.weekday() < 5:  # 0=Mon ... 4=Fri
            yield d
        d += dt.timedelta(days=1)


def load_existing_history() -> pd.DataFrame | None:
    if scanner.HISTORY_FILE.exists():
        return pd.read_parquet(scanner.HISTORY_FILE)
    return None


def save_history(existing: pd.DataFrame | None, new_frames: list) -> pd.DataFrame:
    """Merge new day-frames into existing history and write to disk."""
    parts = ([existing] if existing is not None else []) + new_frames
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
    cutoff = combined["date"].max() - pd.Timedelta(days=460)
    combined = combined[combined["date"] >= cutoff]

    scanner.DATA_DIR.mkdir(exist_ok=True)
    combined.to_parquet(scanner.HISTORY_FILE, index=False)
    return combined


def write_results(hist: pd.DataFrame):
    rankings = scanner.build_rankings(hist)
    output = {
        "as_of": str(hist["date"].max().date()),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "methodology": {
            "weights": scanner.WEIGHTS,
            "filters": {
                "min_price": scanner.MIN_PRICE,
                "min_avg_turnover_20d": scanner.MIN_AVG_TURNOVER_20D,
                "min_history_days": scanner.MIN_HISTORY_DAYS,
            },
        },
        "results": rankings,
    }
    scanner.RESULTS_FILE.write_text(json.dumps(scanner.json_safe(output), indent=2))
    return rankings


def load_known_holidays() -> set:
    """Dates we've already confirmed have no trading data (holidays), so we
    stop re-requesting them on every future run."""
    path = scanner.DATA_DIR / "known_holidays.json"
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def save_known_holidays(holidays: set):
    path = scanner.DATA_DIR / "known_holidays.json"
    scanner.DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(sorted(str(d) for d in holidays)))


def backfill(days_back: int):
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    candidate_days = list(daterange_weekdays(start, end))

    existing = load_existing_history()
    already_have = set(existing["date"].dt.date) if existing is not None else set()
    known_holidays = {dt.date.fromisoformat(d) for d in load_known_holidays()}
    todo_days = [d for d in candidate_days if d not in already_have and d not in known_holidays]

    print(f"Backfill range: {start} to {end} ({len(candidate_days)} weekdays)")
    if already_have:
        print(f"Already have {len(already_have)} days saved -- skipping those.")
    if known_holidays:
        print(f"Skipping {len(known_holidays & set(candidate_days))} known holidays in this range.")
    print(f"{len(todo_days)} left to fetch, one request every {REQUEST_DELAY_SECONDS}s.\n")

    if not todo_days:
        print("Nothing left to backfill. Scoring with existing history...")
        rankings = write_results(existing)
        print(f"{len(rankings)} stocks scored. Wrote {scanner.RESULTS_FILE}")
        return

    session = scanner.new_session()
    pending_frames = []
    consecutive_failures = 0
    hist = existing
    newly_found_holidays = []

    for i, day in enumerate(todo_days, 1):
        try:
            df = scanner.fetch_day(day, session=session)
        except Exception as e:
            print(f"  [{i}/{len(todo_days)}] {day}  ERROR: {e}")
            df = None

        if df is not None and not df.empty:
            pending_frames.append(df)
            consecutive_failures = 0
            print(f"  [{i}/{len(todo_days)}] {day}  ok, {len(df)} stocks")
        else:
            consecutive_failures += 1
            newly_found_holidays.append(day)
            print(f"  [{i}/{len(todo_days)}] {day}  no data (holiday, or not published)")

        if len(pending_frames) >= CHECKPOINT_EVERY:
            hist = save_history(hist, pending_frames)
            print(f"  --- checkpoint: {hist['date'].nunique()} trading days saved to disk ---")
            pending_frames = []

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"\n{consecutive_failures} failures in a row -- could be a genuine holiday "
                  "cluster, or NSE rate-limiting. Not caching this streak as confirmed holidays "
                  "(they'll be retried next run, just in case). Saving what we have.")
            # Don't trust the trailing streak that caused the abort — it's
            # ambiguous whether it's holidays or a real block. Only the
            # no-data days BEFORE this streak are safe to cache permanently.
            newly_found_holidays = newly_found_holidays[:-consecutive_failures]
            break

        time.sleep(REQUEST_DELAY_SECONDS)

    if pending_frames:
        hist = save_history(hist, pending_frames)

    if newly_found_holidays:
        known_holidays |= set(newly_found_holidays)
        save_known_holidays(known_holidays)
        print(f"Cached {len(newly_found_holidays)} newly-confirmed holiday date(s) "
              "-- these won't be retried on future runs.")

    if hist is None or hist.empty:
        print("\nNo data was collected at all. NSE likely blocked these requests "
              "(common from cloud/CI IP ranges). See the troubleshooting note at "
              "the bottom of this file.")
        sys.exit(1)

    print(f"\nHistory now spans {hist['date'].nunique()} trading days, "
          f"{hist['symbol'].nunique()} unique symbols.")
    print("Scoring the universe...")
    rankings = write_results(hist)
    print(f"{len(rankings)} stocks passed filters and were scored. Wrote {scanner.RESULTS_FILE}")
    print("\nDone. Commit data/history.parquet and results.json to your repo -- "
          "the daily workflow takes over from here, adding one day at a time.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill NSE bhavcopy history for the swing scanner.")
    parser.add_argument("--days", type=int, default=400,
                         help="How many calendar days back to pull (default 400, "
                              "roughly a year with buffer for holidays).")
    args = parser.parse_args()
    backfill(args.days)

# ---------------------------------------------------------------------------
# Troubleshooting
# ---------------------------------------------------------------------------
# If every request fails immediately:
#   - NSE sometimes blocks datacenter/cloud IPs (including GitHub Actions runners
#     and many VPS providers) more aggressively than residential ones. Try running
#     this once from your own laptop/home connection to get the initial backfill,
#     then let the daily GitHub Actions workflow (which only needs ONE request a
#     day, not hundreds in a burst) take over -- a single daily request is far
#     less likely to be blocked than a rapid-fire backfill.
#   - If you get blocked partway through, that's fine -- progress is checkpointed
#     every 15 successful days. Just re-run the same command later; it automatically
#     skips days it already has and continues from there.
#   - If you still get blocked everywhere, increase REQUEST_DELAY_SECONDS to 3-5.