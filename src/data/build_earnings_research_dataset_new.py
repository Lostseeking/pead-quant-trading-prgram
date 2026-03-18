import time
import math
import pandas as pd
import yfinance as yf

INPUT_FILE = "data/raw/watchlist_earnings_calendar_clean.csv"
OUTPUT_FILE = "data/processed/earnings_research_dataset.csv"

PRICE_BUFFER_BEFORE_DAYS = 30
PRICE_BUFFER_AFTER_DAYS = 40

HOLD_DAYS_LIST = [1, 5, 10, 20]


def safe_float(x):
    try:
        if pd.isna(x) or x == "":
            return None
        return float(x)
    except Exception:
        return None


def calc_eps_surprise_pct(eps_actual, eps_estimate):
    eps_actual = safe_float(eps_actual)
    eps_estimate = safe_float(eps_estimate)

    if eps_actual is None or eps_estimate is None:
        return None
    if eps_estimate == 0:
        return None

    return (eps_actual - eps_estimate) / abs(eps_estimate)


def calc_rev_surprise_pct(rev_actual, rev_estimate):
    rev_actual = safe_float(rev_actual)
    rev_estimate = safe_float(rev_estimate)

    if rev_actual is None or rev_estimate is None:
        return None
    if rev_estimate == 0:
        return None

    return (rev_actual - rev_estimate) / abs(rev_estimate)


def normalize_release_time(x):
    """
    标准化财报发布时间:
    - bmo / before market open -> BMO
    - amc / after market close -> AMC
    - 其他 / 缺失 -> AMC（保守默认）
    """
    if pd.isna(x):
        return "AMC"

    s = str(x).strip().lower()

    if s in {"bmo", "before market open"}:
        return "BMO"

    if s in {"amc", "after market close"}:
        return "AMC"

    # 很多数据源会给空值或其他值，默认按 AMC 处理更稳妥
    return "AMC"


def get_price_table(symbol, start_date, end_date):
    """
    下载单个股票的历史日线数据。
    返回 index 为交易日的 DataFrame。
    """
    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # 兼容 yfinance 某些版本的列结构
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

        # 优先用 Adj Close，更适合回测
        if "Adj Close" in df.columns:
            df["px"] = pd.to_numeric(df["Adj Close"], errors="coerce")
        else:
            df["px"] = pd.to_numeric(df["Close"], errors="coerce")

        df["Volume"] = pd.to_numeric(df.get("Volume"), errors="coerce")

        df = df[["Date", "px", "Volume"]].dropna(subset=["Date", "px"]).copy()
        df = df.sort_values("Date").reset_index(drop=True)

        return df

    except Exception as e:
        print(f"  price fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def find_first_trading_day_on_or_after(trading_days, target_date):
    """
    找到 >= target_date 的第一个交易日
    """
    candidates = trading_days[trading_days >= target_date]
    if len(candidates) == 0:
        return None
    return candidates[0]


def find_first_trading_day_after(trading_days, target_date):
    """
    找到 > target_date 的第一个交易日
    """
    candidates = trading_days[trading_days > target_date]
    if len(candidates) == 0:
        return None
    return candidates[0]


def get_entry_date(trading_days, earnings_date, release_time):
    """
    BMO: 当天若是交易日，则当天进场；否则下一个交易日
    AMC: 下一交易日进场
    """
    if release_time == "BMO":
        return find_first_trading_day_on_or_after(trading_days, earnings_date)

    return find_first_trading_day_after(trading_days, earnings_date)


def compute_forward_return(price_df, entry_date, hold_days):
    """
    ret_n = close(entry+n) / close(entry) - 1
    这里的 n 是“交易日”数量
    """
    if price_df.empty or entry_date is None:
        return None

    trading_days = price_df["Date"].values

    entry_idx_list = price_df.index[price_df["Date"] == entry_date].tolist()
    if not entry_idx_list:
        return None

    entry_idx = entry_idx_list[0]
    exit_idx = entry_idx + hold_days

    if exit_idx >= len(price_df):
        return None

    entry_px = safe_float(price_df.loc[entry_idx, "px"])
    exit_px = safe_float(price_df.loc[exit_idx, "px"])

    if entry_px is None or exit_px is None or entry_px == 0:
        return None

    return exit_px / entry_px - 1


def build_dataset_for_symbol(symbol_df, price_df):
    """
    对单个 symbol 的 earnings 事件表构建 reaction dataset
    """
    results = []

    if symbol_df.empty:
        return results

    trading_days = price_df["Date"].values if not price_df.empty else []

    for _, row in symbol_df.iterrows():
        earnings_date = row["date"]
        release_time = normalize_release_time(row.get("time"))

        entry_date = get_entry_date(trading_days, earnings_date, release_time)

        eps_surprise_pct = calc_eps_surprise_pct(
            row.get("epsActual"),
            row.get("epsEstimate"),
        )

        rev_surprise_pct = calc_rev_surprise_pct(
            row.get("revenueActual"),
            row.get("revenueEstimate"),
        )

        out = {
            "symbol": row["symbol"],
            "earnings_date": earnings_date,
            "release_time": release_time,
            "entry_date": entry_date,
            "epsActual": safe_float(row.get("epsActual")),
            "epsEstimate": safe_float(row.get("epsEstimate")),
            "revenueActual": safe_float(row.get("revenueActual")),
            "revenueEstimate": safe_float(row.get("revenueEstimate")),
            "eps_surprise_pct": eps_surprise_pct,
            "rev_surprise_pct": rev_surprise_pct,
        }

        for n in HOLD_DAYS_LIST:
            out[f"ret_{n}d"] = compute_forward_return(price_df, entry_date, n)

        results.append(out)

    return results


def main():
    print(f"Loading cleaned earnings file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} rows")

    # 统一列名 / 日期类型
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    if "time" not in df.columns:
        df["time"] = None

    # 只保留有 symbol/date 的记录
    df = df[df["symbol"].notna() & df["date"].notna()].copy()

    # 再保险：去重一次
    df = df.sort_values(["symbol", "date"])
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last").copy()

    print(f"After final sanity dedup: {len(df)} rows")

    symbols = df["symbol"].dropna().unique().tolist()
    total = len(symbols)

    all_results = []

    for i, symbol in enumerate(symbols, start=1):
        symbol_df = df[df["symbol"] == symbol].copy().sort_values("date")

        min_event_date = symbol_df["date"].min()
        max_event_date = symbol_df["date"].max()

        price_start = (min_event_date - pd.Timedelta(days=PRICE_BUFFER_BEFORE_DAYS)).strftime("%Y-%m-%d")
        price_end = (max_event_date + pd.Timedelta(days=PRICE_BUFFER_AFTER_DAYS)).strftime("%Y-%m-%d")

        print(f"[{i}/{total}] {symbol}  events={len(symbol_df)}  price_range={price_start} -> {price_end}")

        price_df = get_price_table(symbol, price_start, price_end)

        if price_df.empty:
            print(f"  no price data for {symbol}")
            # 即使没价格，也保留事件记录，return 留空
            for _, row in symbol_df.iterrows():
                release_time = normalize_release_time(row.get("time"))
                all_results.append({
                    "symbol": row["symbol"],
                    "earnings_date": row["date"],
                    "release_time": release_time,
                    "entry_date": None,
                    "epsActual": safe_float(row.get("epsActual")),
                    "epsEstimate": safe_float(row.get("epsEstimate")),
                    "revenueActual": safe_float(row.get("revenueActual")),
                    "revenueEstimate": safe_float(row.get("revenueEstimate")),
                    "eps_surprise_pct": calc_eps_surprise_pct(row.get("epsActual"), row.get("epsEstimate")),
                    "rev_surprise_pct": calc_rev_surprise_pct(row.get("revenueActual"), row.get("revenueEstimate")),
                    "ret_1d": None,
                    "ret_5d": None,
                    "ret_10d": None,
                    "ret_20d": None,
                })
            continue

        symbol_results = build_dataset_for_symbol(symbol_df, price_df)
        all_results.extend(symbol_results)

        time.sleep(0.1)

    out_df = pd.DataFrame(all_results)

    if out_df.empty:
        print("No results generated.")
        out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        return

    out_df["earnings_date"] = pd.to_datetime(out_df["earnings_date"], errors="coerce")
    out_df["entry_date"] = pd.to_datetime(out_df["entry_date"], errors="coerce")

    out_df = out_df.sort_values(["symbol", "earnings_date"]).reset_index(drop=True)

    print("\nDataset summary:")
    print(f"Rows: {len(out_df)}")
    print(f"ret_1d non-null: {out_df['ret_1d'].notna().sum()}")
    print(f"ret_5d non-null: {out_df['ret_5d'].notna().sum()}")
    print(f"ret_10d non-null: {out_df['ret_10d'].notna().sum()}")
    print(f"ret_20d non-null: {out_df['ret_20d'].notna().sum()}")

    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()