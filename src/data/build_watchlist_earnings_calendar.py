import time
import requests
import pandas as pd
from config import FMP_API_KEY

API_KEY = FMP_API_KEY
WATCHLIST_FILE = "watchlist.txt"
OUTPUT_FILE = "data/raw/watchlist_earnings_calendar.csv"

BASE_URL = "https://financialmodelingprep.com/stable/earnings"

FROM_DATE = "2018-01-01"
TO_DATE = "2026-12-31"

MIN_EVENT_GAP_DAYS = 45


def load_watchlist():
    with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
        return [x.strip().upper() for x in f if x.strip()]


def safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def fetch_earnings(symbol: str):
    params = {
        "symbol": symbol,
        "from": FROM_DATE,
        "to": TO_DATE,
        "apikey": API_KEY,
    }

    try:
        r = requests.get(BASE_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        if not isinstance(data, list):
            print(f"  unexpected response for {symbol}: {data}")
            return []

        rows = []
        for row in data:
            rows.append({
                "symbol": symbol,
                "date": row.get("date"),
                "epsActual": safe_float(row.get("epsActual")),
                "epsEstimate": safe_float(row.get("epsEstimated")),
                "revenueActual": safe_float(row.get("revenueActual")),
                "revenueEstimate": safe_float(row.get("revenueEstimated")),
                "lastUpdated": row.get("lastUpdated"),
            })

        return rows

    except Exception as e:
        print(f"  fetch failed for {symbol}: {e}")
        return []


def dedup_nearby_actual_rows(group: pd.DataFrame) -> pd.DataFrame:

    if group.empty:
        return group

    group = group.sort_values(["date", "lastUpdated"]).reset_index(drop=True)

    kept_indices = []
    last_kept_idx = None

    for idx, row in group.iterrows():
        if last_kept_idx is None:
            kept_indices.append(idx)
            last_kept_idx = idx
            continue

        prev = group.loc[last_kept_idx]
        day_gap = (row["date"] - prev["date"]).days

        if day_gap < MIN_EVENT_GAP_DAYS:
            kept_indices[-1] = idx
            last_kept_idx = idx
        else:
            kept_indices.append(idx)
            last_kept_idx = idx

    return group.loc[kept_indices].copy()


def main():
    tickers = load_watchlist()
    print(f"Loaded {len(tickers)} tickers from {WATCHLIST_FILE}")

    all_rows = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{total}] {ticker}")
        rows = fetch_earnings(ticker)

        if rows:
            print(f"  raw rows: {len(rows)}")
            all_rows.extend(rows)
        else:
            print("  no earnings found")

        time.sleep(0.25)

    df = pd.DataFrame(all_rows)

    if df.empty:
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"\nDone. Saved 0 rows to {OUTPUT_FILE}")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["lastUpdated"] = pd.to_datetime(df["lastUpdated"], errors="coerce")

    df = df[df["date"].notna()].copy()
    print(f"\nAfter removing null dates: {len(df)}")

    df = df[df["epsActual"].notna() | df["revenueActual"].notna()].copy()
    print(f"After removing future-only rows: {len(df)}")

    df = df.sort_values(["symbol", "date", "lastUpdated"])
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
    print(f"After same-date dedup: {len(df)}")

    cleaned_groups = []
    for symbol, group in df.groupby("symbol", group_keys=False):
        cleaned = dedup_nearby_actual_rows(group)
        cleaned_groups.append(cleaned)

    df = pd.concat(cleaned_groups, ignore_index=True)

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    print(f"Final cleaned rows: {len(df)}")

    check_symbols = ["AMD", "NVDA", "TSM", "AVGO", "SNPS", "CDNS"]
    print("\nSample symbol counts after cleaning:")
    for s in check_symbols:
        cnt = int((df["symbol"] == s).sum())
        print(f"  {s}: {cnt}")

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nDone. Saved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()