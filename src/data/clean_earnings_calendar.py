import pandas as pd

INPUT_FILE = "data/raw/watchlist_earnings_calendar.csv"
OUTPUT_FILE = "data/raw/watchlist_earnings_calendar_clean.csv"

df = pd.read_csv(INPUT_FILE)

print("Raw rows:", len(df))

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df[df["date"] >= "2018-01-01"]

print("After 2018 filter:", len(df))

df = df[df["epsActual"].notna() | df["revenueActual"].notna()]

print("After removing future earnings:", len(df))

df = df.sort_values(["symbol", "date", "lastUpdated"])
df = df.drop_duplicates(subset=["symbol", "date"], keep="last")

print("After same-date dedup:", len(df))

df = df.sort_values(["symbol", "date"])

df.to_csv(OUTPUT_FILE, index=False)

print("Saved cleaned dataset:", len(df))