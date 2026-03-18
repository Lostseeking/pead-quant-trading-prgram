import csv
from typing import List, Dict, Any

import pandas as pd

from archive.historical_earnings import get_historical_earnings_events
from src.data.price_api import compute_earnings_reaction


def build_earnings_research_dataset(symbol: str, years: int = 3) -> List[Dict[str, Any]]:
    """
    Build a historical earnings research dataset for one symbol.

    Steps:
    1. Get historical earnings events
    2. For each event, compute price reaction
    3. Merge both into one record list

    Safety rules:
    - Skip future earnings dates
    - Skip very recent earnings dates if next trading day's reaction may not exist yet
    """
    symbol = symbol.upper()

    events = get_historical_earnings_events(symbol, years=years)
    dataset = []

    today = pd.Timestamp.today().normalize()

    for event in events:
        earnings_date = event.get("date")
        if not earnings_date:
            continue

        try:
            earnings_ts = pd.Timestamp(earnings_date).normalize()
        except Exception:
            print(f"[WARN] invalid earnings date for {symbol}: {earnings_date}")
            continue

        # 1) 如果财报日期在未来，直接跳过
        if earnings_ts > today:
            print(f"[INFO] skipping future earnings for {symbol} on {earnings_date}")
            continue

        # 2) 如果财报日期太新，可能 next-day price 还没出来，先不给 reaction
        #    这里用 2 天缓冲，比较稳
        if earnings_ts >= today - pd.Timedelta(days=1):
            print(f"[INFO] skipping too-recent earnings for {symbol} on {earnings_date}")
            price_info = {
                "symbol": symbol,
                "earnings_date": earnings_date,
                "pre_close": None,
                "post_close": None,
                "reaction_1d": None,
            }
        else:
            try:
                price_info = compute_earnings_reaction(symbol, earnings_date)
            except Exception as e:
                print(f"[WARN] price fetch failed for {symbol} on {earnings_date}: {e}")
                price_info = {
                    "symbol": symbol,
                    "earnings_date": earnings_date,
                    "pre_close": None,
                    "post_close": None,
                    "reaction_1d": None,
                }

        reaction = price_info.get("reaction_1d")

        record = {
            "symbol": symbol,
            "date": event.get("date"),
            "year": event.get("year"),
            "quarter": event.get("quarter"),
            "pre_close": price_info.get("pre_close"),
            "post_close": price_info.get("post_close"),
            "reaction_1d": round(reaction, 4) if reaction is not None else None,
        }

        dataset.append(record)

    dataset.sort(key=lambda x: x["date"], reverse=True)
    return dataset


def save_dataset_to_csv(dataset: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save the research dataset to a CSV file.
    """
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


if __name__ == "__main__":
    symbol = input("Enter ticker: ").strip().upper()
    years_str = input("Enter number of years (default 3): ").strip()

    years = 3
    if years_str:
        try:
            years = int(years_str)
        except ValueError:
            print("[WARN] invalid years input, using default = 3")

    dataset = build_earnings_research_dataset(symbol, years=years)

    print(f"\nBuilt {len(dataset)} records for {symbol}.\n")

    for row in dataset:
        reaction = row["reaction_1d"]
        reaction_str = f"{reaction:.2%}" if reaction is not None else "N/A"

        print(
            f"{row['date']} | {row['symbol']} | "
            f"Q{row['quarter']} {row['year']} | "
            f"pre={row['pre_close']} | post={row['post_close']} | "
            f"reaction_1d={reaction_str}"
        )

    output_name = f"{symbol}_earnings_research_{years}y.csv"
    save_dataset_to_csv(dataset, output_name)

    print(f"\nSaved dataset to: {output_name}")