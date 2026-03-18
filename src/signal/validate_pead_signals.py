from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd
import yfinance as yf


USE_FULL_DAY_REACTION_FEATURES = True
FORWARD_WINDOWS = [1, 3, 5, 10, 20]

INPUT_SIGNALS_FILE = "data/processed/pead_signals_2018_2026.csv"
OUTPUT_VALIDATED_FILE = "data/processed/validated_signals_2018_2026.csv"
OUTPUT_SUMMARY_ALL_FILE = "data/processed/validated_summary_all_2018_2026.csv"
OUTPUT_SUMMARY_SIDE_FILE = "data/processed/validated_summary_by_side_2018_2026.csv"
OUTPUT_SUMMARY_YEAR_FILE = "data/processed/validated_summary_by_year_2018_2026.csv"
OUTPUT_SUMMARY_YEAR_SIDE_FILE = "data/processed/validated_summary_by_year_and_side_2018_2026.csv"
OUTPUT_ENTRY_LAG_DETAIL_FILE = "data/processed/entry_lag_detail_2018_2026.csv"
OUTPUT_ENTRY_LAG_SUMMARY_FILE = "data/processed/entry_lag_summary_2018_2026.csv"


# =========================
# Data structures
# =========================

@dataclass
class BacktestConfig:
    use_full_day_reaction_features: bool = True
    forward_windows: tuple[int, ...] = (1, 3, 5, 10, 20)


# =========================
# Helpers
# =========================

def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


# =========================
# Price download
# =========================

def download_price_history(
    symbols: Iterable[str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:

    price_map: Dict[str, pd.DataFrame] = {}

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    today = pd.Timestamp.today().normalize()

    if end_ts > today:
        print(f"[info] requested end date {end_ts.date()} is in the future; capped to {today.date()}")
        end_ts = today

    if start_ts > end_ts:
        print(f"[warn] invalid date range after capping: start={start_ts.date()}, end={end_ts.date()}")
        return price_map

    for symbol in sorted(set(symbols)):
        try:
            df = yf.download(
                symbol,
                start=start_ts.strftime("%Y-%m-%d"),
                end=end_ts.strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as e:
            print(f"[warn] price download failed for {symbol}: {e}")
            continue

        if df.empty:
            print(f"[warn] no price data for {symbol}")
            continue

        df = flatten_yfinance_columns(df)
        df = df.rename(columns=str.title)

        needed = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[warn] {symbol} missing columns: {missing}")
            continue

        df = df[needed].copy()
        df.index = pd.to_datetime(df.index).normalize()
        df = df.sort_index()

        price_map[symbol] = df

    return price_map


# =========================
# Trading-day helpers
# =========================

def next_trading_day(price_df: pd.DataFrame, current_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    idx = price_df.index
    future_days = idx[idx > current_date]
    if len(future_days) == 0:
        return None
    return future_days[0]


def nth_trading_day_after(
    price_df: pd.DataFrame,
    start_date: pd.Timestamp,
    n: int,
) -> Optional[pd.Timestamp]:
    if n < 1:
        raise ValueError("n must be >= 1")

    idx = price_df.index
    future_days = idx[idx > start_date]
    if len(future_days) < n:
        return None
    return future_days[n - 1]


# =========================
# Signal timing logic
# =========================

def infer_entry_date_no_lookahead(
    price_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    use_full_day_reaction_features: bool = True,
) -> Optional[pd.Timestamp]:
    _ = use_full_day_reaction_features
    return next_trading_day(price_df, signal_date)


# =========================
# Forward return calculation
# =========================

def compute_forward_open_to_open_returns(
    price_df: pd.DataFrame,
    entry_date: pd.Timestamp,
    windows: Iterable[int],
) -> Dict[str, float]:

    result: Dict[str, float] = {}

    if entry_date not in price_df.index:
        for w in windows:
            result[f"ret_{w}d"] = math.nan
        return result

    entry_open = float(price_df.loc[entry_date, "Open"])

    for w in windows:
        exit_date = nth_trading_day_after(price_df, entry_date, w)
        if exit_date is None:
            result[f"ret_{w}d"] = math.nan
            continue

        exit_open = float(price_df.loc[exit_date, "Open"])
        result[f"ret_{w}d"] = (exit_open - entry_open) / entry_open

    return result


def compute_forward_open_to_close_returns(
    price_df: pd.DataFrame,
    entry_date: pd.Timestamp,
    windows: Iterable[int],
) -> Dict[str, float]:

    result: Dict[str, float] = {}

    if entry_date not in price_df.index:
        for w in windows:
            result[f"ret_close_{w}d"] = math.nan
        return result

    entry_open = float(price_df.loc[entry_date, "Open"])

    for w in windows:
        exit_date = nth_trading_day_after(price_df, entry_date, w)
        if exit_date is None:
            result[f"ret_close_{w}d"] = math.nan
            continue

        exit_close = float(price_df.loc[exit_date, "Close"])
        result[f"ret_close_{w}d"] = (exit_close - entry_open) / entry_open

    return result


# =========================
# Main validation builder
# =========================

def build_validation_frame(
    signals_df: pd.DataFrame,
    price_map: Dict[str, pd.DataFrame],
    config: BacktestConfig,
) -> pd.DataFrame:

    required_cols = {"symbol", "date", "score", "signal"}
    missing = required_cols - set(signals_df.columns)
    if missing:
        raise ValueError(f"signals_df missing columns: {missing}")

    df = signals_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df[df["date"].notna()].copy()

    rows = []

    for row in df.itertuples(index=False):
        symbol = row.symbol
        signal_date = pd.Timestamp(row.date).normalize()

        price_df = price_map.get(symbol)
        if price_df is None or price_df.empty:
            continue

        entry_date = infer_entry_date_no_lookahead(
            price_df=price_df,
            signal_date=signal_date,
            use_full_day_reaction_features=config.use_full_day_reaction_features,
        )

        if entry_date is None or entry_date not in price_df.index:
            continue

        entry_open = float(price_df.loc[entry_date, "Open"])

        ret_open = compute_forward_open_to_open_returns(
            price_df=price_df,
            entry_date=entry_date,
            windows=config.forward_windows,
        )

        ret_close = compute_forward_open_to_close_returns(
            price_df=price_df,
            entry_date=entry_date,
            windows=config.forward_windows,
        )

        record = row._asdict()
        record["signal_date"] = signal_date
        record["entry_date"] = entry_date
        record["entry_open"] = entry_open
        record["entry_year"] = entry_date.year

        record.update(ret_open)
        record.update(ret_close)

        for w in config.forward_windows:
            raw_ret = record.get(f"ret_{w}d", math.nan)
            if pd.isna(raw_ret):
                record[f"pnl_{w}d"] = math.nan
            else:
                if record["signal"] == "LONG":
                    record[f"pnl_{w}d"] = raw_ret
                elif record["signal"] == "SHORT":
                    record[f"pnl_{w}d"] = -raw_ret
                else:
                    record[f"pnl_{w}d"] = math.nan

        rows.append(record)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["entry_date", "symbol"]).reset_index(drop=True)

    return out


# =========================
# Summary stats
# =========================

def summarize_validation_results(
    validation_df: pd.DataFrame,
    windows: Iterable[int],
) -> pd.DataFrame:

    summaries = []

    for w in windows:
        pnl_col = f"pnl_{w}d"
        if pnl_col not in validation_df.columns:
            continue

        s = validation_df[pnl_col].dropna()
        if s.empty:
            continue

        summary = {
            "window": f"{w}d",
            "count": int(s.shape[0]),
            "mean_pnl": float(s.mean()),
            "median_pnl": float(s.median()),
            "win_rate": float((s > 0).mean()),
        }
        summaries.append(summary)

    return pd.DataFrame(summaries)


def summarize_by_side(
    validation_df: pd.DataFrame,
    windows: Iterable[int],
) -> pd.DataFrame:

    summaries = []

    for side in ["LONG", "SHORT"]:
        side_df = validation_df[validation_df["signal"] == side].copy()
        if side_df.empty:
            continue

        for w in windows:
            ret_col = f"ret_{w}d"
            if ret_col not in side_df.columns:
                continue

            s = side_df[ret_col].dropna()
            if s.empty:
                continue

            if side == "LONG":
                win_rate = float((s > 0).mean())
            else:
                win_rate = float((s < 0).mean())

            summaries.append({
                "side": side,
                "window": f"{w}d",
                "count": int(s.shape[0]),
                "mean_raw_return": float(s.mean()),
                "median_raw_return": float(s.median()),
                "win_rate": win_rate,
            })

    return pd.DataFrame(summaries)


def summarize_by_year(
    validation_df: pd.DataFrame,
    windows: Iterable[int],
) -> pd.DataFrame:

    summaries = []

    if validation_df.empty:
        return pd.DataFrame(
            columns=["year", "window", "count", "mean_pnl", "median_pnl", "win_rate"]
        )

    for year in sorted(validation_df["entry_year"].dropna().unique()):
        year_df = validation_df[validation_df["entry_year"] == year].copy()
        if year_df.empty:
            continue

        for w in windows:
            pnl_col = f"pnl_{w}d"
            if pnl_col not in year_df.columns:
                continue

            s = year_df[pnl_col].dropna()
            if s.empty:
                continue

            summaries.append({
                "year": int(year),
                "window": f"{w}d",
                "count": int(s.shape[0]),
                "mean_pnl": float(s.mean()),
                "median_pnl": float(s.median()),
                "win_rate": float((s > 0).mean()),
            })

    return pd.DataFrame(summaries).sort_values(["year", "window"]).reset_index(drop=True)


def summarize_by_year_and_side(
    validation_df: pd.DataFrame,
    windows: Iterable[int],
) -> pd.DataFrame:

    summaries = []

    if validation_df.empty:
        return pd.DataFrame(
            columns=["year", "side", "window", "count", "mean_raw_return", "median_raw_return", "win_rate"]
        )

    for year in sorted(validation_df["entry_year"].dropna().unique()):
        year_df = validation_df[validation_df["entry_year"] == year].copy()
        if year_df.empty:
            continue

        for side in ["LONG", "SHORT"]:
            side_df = year_df[year_df["signal"] == side].copy()
            if side_df.empty:
                continue

            for w in windows:
                ret_col = f"ret_{w}d"
                if ret_col not in side_df.columns:
                    continue

                s = side_df[ret_col].dropna()
                if s.empty:
                    continue

                if side == "LONG":
                    win_rate = float((s > 0).mean())
                else:
                    win_rate = float((s < 0).mean())

                summaries.append({
                    "year": int(year),
                    "side": side,
                    "window": f"{w}d",
                    "count": int(s.shape[0]),
                    "mean_raw_return": float(s.mean()),
                    "median_raw_return": float(s.median()),
                    "win_rate": win_rate,
                })

    return pd.DataFrame(summaries).sort_values(["year", "side", "window"]).reset_index(drop=True)


def build_entry_lag_table(
    signals_df: pd.DataFrame,
    price_map: Dict[str, pd.DataFrame],
    entry_lags: Iterable[int] = (1, 2, 3, 5),
    hold_windows: Iterable[int] = (5, 10, 20),
) -> pd.DataFrame:

    required_cols = {"symbol", "date", "score", "signal"}
    missing = required_cols - set(signals_df.columns)
    if missing:
        raise ValueError(f"signals_df missing columns: {missing}")

    df = signals_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df[df["date"].notna()].copy()

    # long-only
    df = df[df["signal"] == "LONG"].copy()

    rows = []

    for row in df.itertuples(index=False):
        symbol = row.symbol
        signal_date = pd.Timestamp(row.date).normalize()

        price_df = price_map.get(symbol)
        if price_df is None or price_df.empty:
            continue

        price_df = price_df.copy()
        price_df.index = pd.to_datetime(price_df.index).normalize()
        price_df = price_df.sort_index()

        for lag in entry_lags:
            entry_date = nth_trading_day_after(price_df, signal_date, lag)
            if entry_date is None:
                continue

            entry_date = pd.Timestamp(entry_date).normalize()

            if entry_date not in price_df.index:
                continue

            try:
                entry_open = float(price_df.loc[entry_date, "Open"])
            except Exception:
                continue

            for hold in hold_windows:
                exit_date = nth_trading_day_after(price_df, entry_date, hold)
                if exit_date is None:
                    continue

                exit_date = pd.Timestamp(exit_date).normalize()

                if exit_date not in price_df.index:
                    continue

                try:
                    exit_close = float(price_df.loc[exit_date, "Close"])
                except Exception:
                    continue

                pnl = (exit_close - entry_open) / entry_open

                rows.append({
                    "symbol": symbol,
                    "signal_date": signal_date,
                    "score": row.score,
                    "signal": row.signal,
                    "entry_lag": lag,
                    "hold_days": hold,
                    "entry_date": entry_date,
                    "entry_year": entry_date.year,
                    "exit_date": exit_date,
                    "entry_open": entry_open,
                    "exit_close": exit_close,
                    "pnl": pnl,
                })

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["entry_lag", "hold_days", "entry_date", "symbol"]
        ).reset_index(drop=True)

    return out


def summarize_entry_lag_table(entry_lag_df: pd.DataFrame) -> pd.DataFrame:

    if entry_lag_df.empty:
        return pd.DataFrame(
            columns=[
                "entry_lag",
                "hold_days",
                "count",
                "mean_pnl",
                "median_pnl",
                "win_rate",
            ]
        )

    summary = (
        entry_lag_df
        .groupby(["entry_lag", "hold_days"])["pnl"]
        .agg(
            count="count",
            mean_pnl="mean",
            median_pnl="median",
            win_rate=lambda s: (s > 0).mean(),
        )
        .reset_index()
        .sort_values(["entry_lag", "hold_days"])
        .reset_index(drop=True)
    )

    return summary


def summarize_entry_lag_by_year(entry_lag_df: pd.DataFrame) -> pd.DataFrame:
    if entry_lag_df.empty:
        return pd.DataFrame(
            columns=[
                "entry_year",
                "entry_lag",
                "hold_days",
                "count",
                "mean_pnl",
                "median_pnl",
                "win_rate",
            ]
        )

    summary = (
        entry_lag_df
        .groupby(["entry_year", "entry_lag", "hold_days"])["pnl"]
        .agg(
            count="count",
            mean_pnl="mean",
            median_pnl="median",
            win_rate=lambda s: (s > 0).mean(),
        )
        .reset_index()
        .sort_values(["entry_year", "entry_lag", "hold_days"])
        .reset_index(drop=True)
    )

    return summary


# =========================
# Example usage
# =========================

if __name__ == "__main__":

    signals = pd.read_csv(INPUT_SIGNALS_FILE)

    # 只测试 score = 4
    signals = signals[signals["score"] == 4].copy()

    min_date = pd.to_datetime(signals["date"], errors="coerce").min() - pd.Timedelta(days=40)
    max_date = pd.to_datetime(signals["date"], errors="coerce").max() + pd.Timedelta(days=60)

    symbols = signals["symbol"].dropna().astype(str).str.upper().unique().tolist()

    price_map = download_price_history(
        symbols=symbols,
        start=min_date.strftime("%Y-%m-%d"),
        end=max_date.strftime("%Y-%m-%d"),
    )

    config = BacktestConfig(
        use_full_day_reaction_features=USE_FULL_DAY_REACTION_FEATURES,
        forward_windows=tuple(FORWARD_WINDOWS),
    )

    validation_df = build_validation_frame(
        signals_df=signals,
        price_map=price_map,
        config=config,
    )

    summary_all = summarize_validation_results(
        validation_df=validation_df,
        windows=FORWARD_WINDOWS,
    )

    summary_side = summarize_by_side(
        validation_df=validation_df,
        windows=FORWARD_WINDOWS,
    )

    summary_year = summarize_by_year(
        validation_df=validation_df,
        windows=FORWARD_WINDOWS,
    )

    summary_year_side = summarize_by_year_and_side(
        validation_df=validation_df,
        windows=FORWARD_WINDOWS,
    )

    validation_df.to_csv(OUTPUT_VALIDATED_FILE, index=False)
    summary_all.to_csv(OUTPUT_SUMMARY_ALL_FILE, index=False)
    summary_side.to_csv(OUTPUT_SUMMARY_SIDE_FILE, index=False)
    summary_year.to_csv(OUTPUT_SUMMARY_YEAR_FILE, index=False)
    summary_year_side.to_csv(OUTPUT_SUMMARY_YEAR_SIDE_FILE, index=False)

    print("\nSaved:")
    print(f"  {OUTPUT_VALIDATED_FILE}")
    print(f"  {OUTPUT_SUMMARY_ALL_FILE}")
    print(f"  {OUTPUT_SUMMARY_SIDE_FILE}")
    print(f"  {OUTPUT_SUMMARY_YEAR_FILE}")
    print(f"  {OUTPUT_SUMMARY_YEAR_SIDE_FILE}")

    print("\nOverall strategy summary:")
    if summary_all.empty:
        print("No summary rows.")
    else:
        print(summary_all.to_string(index=False))

    print("\nBy side summary:")
    if summary_side.empty:
        print("No side summary rows.")
    else:
        print(summary_side.to_string(index=False))

    print("\nBy year summary:")
    if summary_year.empty:
        print("No year summary rows.")
    else:
        print(summary_year.to_string(index=False))

    print("\nBy year and side summary:")
    if summary_year_side.empty:
        print("No year-side summary rows.")
    else:
        print(summary_year_side.to_string(index=False))

    # =========================
    # Entry lag test (LONG only)
    # =========================
    entry_lag_df = build_entry_lag_table(
        signals_df=signals,
        price_map=price_map,
        entry_lags=(1, 2, 3, 5),
        hold_windows=(3, 5, 10, 20),
    )

    entry_lag_summary = summarize_entry_lag_table(entry_lag_df)
    entry_lag_summary_by_year = summarize_entry_lag_by_year(entry_lag_df)

    OUTPUT_ENTRY_LAG_YEAR_SUMMARY_FILE = "entry_lag_summary_by_year_2018_2026.csv"

    entry_lag_df.to_csv(OUTPUT_ENTRY_LAG_DETAIL_FILE, index=False)
    entry_lag_summary.to_csv(OUTPUT_ENTRY_LAG_SUMMARY_FILE, index=False)
    entry_lag_summary_by_year.to_csv(OUTPUT_ENTRY_LAG_YEAR_SUMMARY_FILE, index=False)

    print("\nSaved:")
    print(f"  {OUTPUT_ENTRY_LAG_DETAIL_FILE}")
    print(f"  {OUTPUT_ENTRY_LAG_SUMMARY_FILE}")
    print(f"  {OUTPUT_ENTRY_LAG_YEAR_SUMMARY_FILE}")

    print("\nEntry lag summary:")
    if entry_lag_summary.empty:
        print("No entry lag rows.")
    else:
        print(entry_lag_summary.to_string(index=False))

    print("\nEntry lag summary by year:")
    if entry_lag_summary_by_year.empty:
        print("No entry lag year rows.")
    else:
        print(entry_lag_summary_by_year.to_string(index=False))