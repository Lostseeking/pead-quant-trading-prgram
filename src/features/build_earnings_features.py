import time
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf


INPUT_FILE = "data/processed/earnings_research_dataset.csv"
OUTPUT_FILE = "data/processed/earnings_features_2018_2026.csv"

SUE_MIN_HISTORY = 4
SLEEP_SECONDS = 0.1


def safe_float(x):
    if pd.isna(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def compute_price_features_for_entry(symbol: str, entry_date_str: str):

    if pd.isna(entry_date_str):
        return None

    try:
        entry_date = pd.Timestamp(entry_date_str).normalize()
    except Exception:
        return None

    start = (entry_date - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    end = (entry_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=False,
        )
    except Exception as e:
        raise Exception(f"yfinance download failed: {e}")

    if df is None or df.empty or len(df) < 5:
        return None

    df = flatten_yfinance_columns(df)
    df = df.reset_index()

    if "Date" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    required_cols = {"Open", "Close", "Volume"}
    if not required_cols.issubset(set(df.columns)):
        raise Exception(f"missing columns in yfinance output: {df.columns.tolist()}")

    event_idx = None
    for i in range(len(df)):
        if df.iloc[i]["Date"] >= entry_date:
            event_idx = i
            break

    if event_idx is None or event_idx == 0:
        return None

    prev_idx = event_idx - 1

    try:
        close_prev_day = float(df.iloc[prev_idx]["Close"])
        open_day1 = float(df.iloc[event_idx]["Open"])
        volume_day1 = float(df.iloc[event_idx]["Volume"])

        start_hist_idx = max(0, prev_idx - 19)
        avg_volume_20d = float(
            df.iloc[start_hist_idx:prev_idx + 1]["Volume"].astype(float).mean()
        )
    except Exception as e:
        raise Exception(f"failed to extract scalar price fields: {e}")

    gap_pct = None
    volume_ratio = None

    if close_prev_day is not None and close_prev_day != 0:
        gap_pct = (open_day1 - close_prev_day) / close_prev_day

    if avg_volume_20d is not None and avg_volume_20d != 0:
        volume_ratio = volume_day1 / avg_volume_20d

    return {
        "close_prev_day": close_prev_day,
        "open_day1": open_day1,
        "gap_pct": gap_pct,
        "volume_day1": volume_day1,
        "avg_volume_20d": avg_volume_20d,
        "volume_ratio": volume_ratio,
    }


def add_sue_feature(df: pd.DataFrame, min_history: int = 4) -> pd.DataFrame:

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    out["eps_actual"] = pd.to_numeric(out["eps_actual"], errors="coerce")
    out["eps_estimate"] = pd.to_numeric(out["eps_estimate"], errors="coerce")

    out["eps_surprise_raw"] = out["eps_actual"] - out["eps_estimate"]

    def compute_group_sue(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy().sort_values("date")

        past_std = (
            g["eps_surprise_raw"]
            .rolling(window=min_history, min_periods=min_history)
            .std()
            .shift(1)
        )

        g["sue"] = g["eps_surprise_raw"] / past_std
        g.loc[(past_std == 0) | (past_std.isna()), "sue"] = np.nan
        return g

    out = (
        out.groupby("symbol", group_keys=False)
        .apply(compute_group_sue)
        .reset_index(drop=True)
    )

    return out


def normalize_input_schema(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    if "earnings_date" in out.columns and "date" not in out.columns:
        out["date"] = out["earnings_date"]

    rename_map = {}

    if "epsActual" in out.columns:
        rename_map["epsActual"] = "eps_actual"
    if "epsEstimate" in out.columns:
        rename_map["epsEstimate"] = "eps_estimate"
    if "revenueActual" in out.columns:
        rename_map["revenueActual"] = "rev_actual"
    if "revenueEstimate" in out.columns:
        rename_map["revenueEstimate"] = "rev_estimate"

    out = out.rename(columns=rename_map)

    required_defaults = {
        "eps_actual": np.nan,
        "eps_estimate": np.nan,
        "rev_actual": np.nan,
        "rev_estimate": np.nan,
        "eps_surprise_pct": np.nan,
        "rev_surprise_pct": np.nan,
        "release_time": None,
        "entry_date": None,
        "ret_1d": np.nan,
        "ret_5d": np.nan,
        "ret_10d": np.nan,
        "ret_20d": np.nan,
    }

    for col, default_val in required_defaults.items():
        if col not in out.columns:
            out[col] = default_val

    if "guidance_flag" not in out.columns:
        out["guidance_flag"] = 0

    return out


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} base rows from {INPUT_FILE}")

    df = normalize_input_schema(df)

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")

    df = df[df["symbol"].notna() & df["date"].notna()].copy()

    results = []
    total = len(df)

    for i, row in df.iterrows():
        symbol = str(row["symbol"]).strip().upper()
        date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")

        entry_date = None
        if pd.notna(row.get("entry_date")):
            entry_date = pd.Timestamp(row["entry_date"]).strftime("%Y-%m-%d")

        print(f"[{i+1}/{total}] processing {symbol} {date}")

        new_row = row.to_dict()

        eps_actual = safe_float(new_row.get("eps_actual"))
        eps_estimate = safe_float(new_row.get("eps_estimate"))
        rev_actual = safe_float(new_row.get("rev_actual"))
        rev_estimate = safe_float(new_row.get("rev_estimate"))

        if pd.isna(new_row.get("eps_surprise_pct")):
            if eps_actual is not None and eps_estimate is not None and eps_estimate != 0:
                new_row["eps_surprise_pct"] = (eps_actual - eps_estimate) / abs(eps_estimate)
            else:
                new_row["eps_surprise_pct"] = np.nan

        if pd.isna(new_row.get("rev_surprise_pct")):
            if rev_actual is not None and rev_estimate is not None and rev_estimate != 0:
                new_row["rev_surprise_pct"] = (rev_actual - rev_estimate) / abs(rev_estimate)
            else:
                new_row["rev_surprise_pct"] = np.nan

        try:
            if entry_date is not None:
                price = compute_price_features_for_entry(symbol, entry_date)
            else:
                price = None
        except Exception as e:
            print(f"  price fetch failed for {symbol} {date}: {e}")
            price = None

        if price is not None:
            new_row.update(price)
        else:
            new_row.update({
                "close_prev_day": np.nan,
                "open_day1": np.nan,
                "gap_pct": np.nan,
                "volume_day1": np.nan,
                "avg_volume_20d": np.nan,
                "volume_ratio": np.nan,
            })

        try:
            new_row["guidance_flag"] = int(new_row.get("guidance_flag", 0))
        except Exception:
            new_row["guidance_flag"] = 0

        results.append(new_row)

        time.sleep(SLEEP_SECONDS)

    out_df = pd.DataFrame(results)

    out_df = add_sue_feature(out_df, min_history=SUE_MIN_HISTORY)

    preferred_cols = [
        "symbol",
        "date",
        "release_time",
        "entry_date",

        "eps_actual",
        "eps_estimate",
        "eps_surprise_raw",
        "eps_surprise_pct",
        "sue",

        "rev_actual",
        "rev_estimate",
        "rev_surprise_pct",

        "close_prev_day",
        "open_day1",
        "gap_pct",
        "volume_day1",
        "avg_volume_20d",
        "volume_ratio",

        "guidance_flag",

        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
    ]

    existing_cols = [c for c in preferred_cols if c in out_df.columns]
    other_cols = [c for c in out_df.columns if c not in existing_cols]
    out_df = out_df[existing_cols + other_cols]

    out_df = out_df.sort_values(["symbol", "date"]).reset_index(drop=True)

    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nSaved {len(out_df)} rows to {OUTPUT_FILE}")
    print("\nNon-null summary:")
    for c in ["eps_surprise_pct", "sue", "rev_surprise_pct", "gap_pct", "volume_ratio"]:
        if c in out_df.columns:
            print(f"  {c}: {out_df[c].notna().sum()}")


if __name__ == "__main__":
    main()