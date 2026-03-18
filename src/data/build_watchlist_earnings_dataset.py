import csv
import random
import time
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Set, Tuple

import pandas as pd

from src.data.earnings_api import get_earnings_calendar
from src.data.price_api import compute_earnings_reaction


def load_watchlist(path: str = "watchlist.txt") -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip()]


def save_dataset_to_csv(dataset: List[Dict[str, Any]], output_path: str) -> None:
    fieldnames = [
        "symbol",
        "date",
        "year",
        "quarter",
        "pre_close",
        "post_close",
        "reaction_1d",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)


def resolve_date_range(years: int = 3) -> Tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=365 * years)
    return start, end


def month_chunks(start: date, end: date) -> List[Tuple[date, date]]:
    chunks = []
    current = start

    while current <= end:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)

        chunk_end = min(end, next_month - timedelta(days=1))
        chunks.append((current, chunk_end))
        current = next_month

    return chunks


def fetch_watchlist_earnings_events(
    watchlist: Set[str],
    years: int = 3,
    per_chunk_sleep_min: float = 1.0,
    per_chunk_sleep_max: float = 2.0,
) -> List[Dict[str, Any]]:
    start, end = resolve_date_range(years=years)
    chunks = month_chunks(start, end)

    print(f"Fetching earnings calendar from {start} to {end}")
    print(f"Total month chunks: {len(chunks)}\n")

    all_events: List[Dict[str, Any]] = []
    seen = set()

    for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(
            f"[Calendar {idx}/{len(chunks)}] "
            f"{chunk_start.isoformat()} -> {chunk_end.isoformat()}"
        )

        try:
            raw_events = get_earnings_calendar(
                chunk_start.isoformat(),
                chunk_end.isoformat()
            )
        except Exception as e:
            print(f"  -> failed chunk: {e}")
            sleep_seconds = random.uniform(per_chunk_sleep_min, per_chunk_sleep_max)
            time.sleep(sleep_seconds)
            continue

        kept = 0

        for item in raw_events:
            symbol = (item.get("symbol") or "").upper()
            if symbol not in watchlist:
                continue

            event_date = item.get("date")
            year = item.get("year")
            quarter = item.get("quarter")

            key = (symbol, event_date, year, quarter)
            if key in seen:
                continue

            seen.add(key)
            all_events.append({
                "symbol": symbol,
                "date": event_date,
                "year": year,
                "quarter": quarter,
            })
            kept += 1

        print(f"  -> kept {kept} watchlist events")
        sleep_seconds = random.uniform(per_chunk_sleep_min, per_chunk_sleep_max)
        time.sleep(sleep_seconds)

    all_events.sort(key=lambda x: (x["symbol"], x["date"]))
    return all_events


def build_watchlist_earnings_dataset(
    watchlist_path: str = "watchlist.txt",
    years: int = 3,
    autosave_every_n_events: int = 100,
    per_chunk_sleep_min: float = 1.0,
    per_chunk_sleep_max: float = 2.0,
    per_event_sleep_min: float = 0.2,
    per_event_sleep_max: float = 0.6,
    chunk_cooldown_every: int = 12,
    chunk_cooldown_min: int = 10,
    chunk_cooldown_max: int = 20,
) -> List[Dict[str, Any]]:
    tickers = load_watchlist(watchlist_path)
    watchlist = set(tickers)

    print(f"Loaded {len(tickers)} tickers from {watchlist_path}.\n")

    # Step 1: fetch all calendar events once
    start, end = resolve_date_range(years=years)
    chunks = month_chunks(start, end)

    all_events: List[Dict[str, Any]] = []
    seen = set()

    print(f"Fetching earnings calendar once for all watchlist names...")
    print(f"Range: {start} -> {end}")
    print(f"Chunks: {len(chunks)}\n")

    for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(
            f"[Calendar {idx}/{len(chunks)}] "
            f"{chunk_start.isoformat()} -> {chunk_end.isoformat()}"
        )

        try:
            raw_events = get_earnings_calendar(
                chunk_start.isoformat(),
                chunk_end.isoformat()
            )
        except Exception as e:
            print(f"  -> failed chunk: {e}")
            sleep_seconds = random.uniform(per_chunk_sleep_min, per_chunk_sleep_max)
            time.sleep(sleep_seconds)
            continue

        kept = 0
        for item in raw_events:
            symbol = (item.get("symbol") or "").upper()
            if symbol not in watchlist:
                continue

            event_date = item.get("date")
            year = item.get("year")
            quarter = item.get("quarter")

            key = (symbol, event_date, year, quarter)
            if key in seen:
                continue

            seen.add(key)
            all_events.append({
                "symbol": symbol,
                "date": event_date,
                "year": year,
                "quarter": quarter,
            })
            kept += 1

        print(f"  -> kept {kept} watchlist events")

        sleep_seconds = random.uniform(per_chunk_sleep_min, per_chunk_sleep_max)
        time.sleep(sleep_seconds)

        if idx % chunk_cooldown_every == 0 and idx < len(chunks):
            cooldown = random.randint(chunk_cooldown_min, chunk_cooldown_max)
            print(f"  -> chunk cooldown {cooldown} seconds\n")
            time.sleep(cooldown)

    all_events.sort(key=lambda x: (x["symbol"], x["date"]))

    print("\n" + "=" * 60)
    print("Finished fetching calendar events.")
    print(f"Unique watchlist events found: {len(all_events)}")
    print("=" * 60 + "\n")

    # Step 2: compute price reaction per event
    dataset: List[Dict[str, Any]] = []
    failed_events: List[str] = []

    today = pd.Timestamp.today().normalize()

    total_events = len(all_events)

    for i, event in enumerate(all_events, start=1):
        symbol = event["symbol"]
        earnings_date = event["date"]

        print(f"[{i}/{total_events}] processing {symbol} {earnings_date}")

        if not earnings_date:
            failed_events.append(f"{symbol}: missing date")
            continue

        try:
            earnings_ts = pd.Timestamp(earnings_date).normalize()
        except Exception:
            print(f"  -> invalid date: {earnings_date}")
            failed_events.append(f"{symbol}: invalid date {earnings_date}")
            continue

        # future event: skip
        if earnings_ts > today:
            print(f"  -> skipping future event")
            continue

        # too recent: keep event, but reaction unknown for now
        if earnings_ts >= today - pd.Timedelta(days=1):
            price_info = {
                "pre_close": None,
                "post_close": None,
                "reaction_1d": None,
            }
            print("  -> too recent, reaction_1d set to None")
        else:
            try:
                price_info = compute_earnings_reaction(symbol, earnings_date)
            except Exception as e:
                print(f"  -> price fetch failed: {e}")
                failed_events.append(f"{symbol} {earnings_date}: {e}")
                price_info = {
                    "pre_close": None,
                    "post_close": None,
                    "reaction_1d": None,
                }

        reaction = price_info.get("reaction_1d")

        record = {
            "symbol": symbol,
            "date": earnings_date,
            "year": event.get("year"),
            "quarter": event.get("quarter"),
            "pre_close": price_info.get("pre_close"),
            "post_close": price_info.get("post_close"),
            "reaction_1d": round(reaction, 4) if reaction is not None else None,
        }

        dataset.append(record)

        if autosave_every_n_events and i % autosave_every_n_events == 0:
            temp_output = f"watchlist_earnings_research_{years}y_partial.csv"
            dataset.sort(key=lambda x: (x["symbol"], x["date"]))
            save_dataset_to_csv(dataset, temp_output)
            print(f"  -> autosaved partial dataset to {temp_output}")

        sleep_seconds = random.uniform(per_event_sleep_min, per_event_sleep_max)
        time.sleep(sleep_seconds)

    dataset.sort(key=lambda x: (x["symbol"], x["date"]))

    print("\n" + "=" * 60)
    print("Finished building dataset.")
    print(f"Total rows: {len(dataset)}")
    print(f"Failed events: {len(failed_events)}")
    if failed_events:
        print("Some failed events:")
        for msg in failed_events[:20]:
            print(f"  - {msg}")
        if len(failed_events) > 20:
            print(f"  ... and {len(failed_events) - 20} more")
    print("=" * 60 + "\n")

    return dataset


if __name__ == "__main__":
    years_str = input("Enter number of years (default 3): ").strip()

    years = 3
    if years_str:
        try:
            years = int(years_str)
        except ValueError:
            print("[WARN] Invalid input, using default = 3")

    dataset = build_watchlist_earnings_dataset(
        years=years,
        autosave_every_n_events=100,
        per_chunk_sleep_min=1.0,
        per_chunk_sleep_max=2.0,
        per_event_sleep_min=0.2,
        per_event_sleep_max=0.6,
        chunk_cooldown_every=12,
        chunk_cooldown_min=10,
        chunk_cooldown_max=20,
    )

    print(f"\nBuilt total {len(dataset)} rows.\n")

    output_name = f"watchlist_earnings_research_{years}y.csv"
    save_dataset_to_csv(dataset, output_name)

    print(f"Saved dataset to: {output_name}")