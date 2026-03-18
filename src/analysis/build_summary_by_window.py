from pathlib import Path
import argparse
import math
import sys

import pandas as pd


ENTRY_DAY_CANDIDATES = [
    "entry_day",
    "entry_lag",
    "entry_delay",
    "lag",
]

HOLDING_DAY_CANDIDATES = [
    "hold_days",
    "holding_days",
    "window",
    "holding_period",
]

DATE_CANDIDATES = [
    "date",
    "signal_date",
    "entry_date",
    "earnings_date",
    "report_date",
]

RET_CANDIDATES = [
    "ret",
    "return",
    "pnl",
    "trade_return",
    "holding_return",
]


def detect_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def compute_metrics_from_series(series: pd.Series) -> dict:
    rets = pd.to_numeric(series, errors="coerce").dropna()
    n = len(rets)

    if n == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "sharpe": None,
            "t_stat": None,
            "win_rate": None,
        }

    mean = rets.mean()
    median = rets.median()
    std = rets.std(ddof=1) if n > 1 else 0.0
    win_rate = (rets > 0).mean()

    sharpe = mean / std if std and std > 0 else None
    t_stat = mean / (std / math.sqrt(n)) if std and std > 0 and n > 1 else None

    return {
        "count": int(n),
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "sharpe": float(sharpe) if sharpe is not None else None,
        "t_stat": float(t_stat) if t_stat is not None else None,
        "win_rate": float(win_rate),
    }


def build_overall_summary(
    df: pd.DataFrame,
    entry_col: str,
    hold_col: str,
    ret_col: str,
) -> pd.DataFrame:
    rows = []

    for (entry_value, hold_value), group in df.groupby([entry_col, hold_col]):
        metrics = compute_metrics_from_series(group[ret_col])
        row = {
            "entry_day": entry_value,
            "hold_days": hold_value,
            "ret_col": ret_col,
            **metrics,
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    result = result.sort_values(["entry_day", "hold_days"]).reset_index(drop=True)
    return result


def build_yearly_summary(
    df: pd.DataFrame,
    entry_col: str,
    hold_col: str,
    date_col: str,
    ret_col: str,
) -> pd.DataFrame:
    working = df.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.dropna(subset=[date_col])
    working["year"] = working[date_col].dt.year

    rows = []

    for (entry_value, hold_value, year), group in working.groupby([entry_col, hold_col, "year"]):
        metrics = compute_metrics_from_series(group[ret_col])
        row = {
            "entry_day": entry_value,
            "hold_days": hold_value,
            "year": int(year),
            "ret_col": ret_col,
            **metrics,
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    result = result.sort_values(["entry_day", "hold_days", "year"]).reset_index(drop=True)
    return result


def print_pretty_summary(summary_df: pd.DataFrame) -> None:
    pretty = summary_df.copy()

    for col in ["mean", "median", "std", "win_rate"]:
        if col in pretty.columns:
            pretty[col] = pretty[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")

    for col in ["sharpe", "t_stat"]:
        if col in pretty.columns:
            pretty[col] = pretty[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    print("\n=== Summary by Entry Day and Holding Window ===")
    print(pretty.to_string(index=False))


def print_top_combinations(summary_df: pd.DataFrame, top_n: int = 5) -> None:
    valid = summary_df.dropna(subset=["t_stat", "mean"]).copy()

    if valid.empty:
        print("\nNo valid rows available for ranking.")
        return

    top_t = valid.sort_values(["t_stat", "mean"], ascending=False).head(top_n)
    top_mean = valid.sort_values(["mean", "t_stat"], ascending=False).head(top_n)

    def _format(df_: pd.DataFrame) -> pd.DataFrame:
        out = df_.copy()
        for col in ["mean", "median", "std", "win_rate"]:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        for col in ["sharpe", "t_stat"]:
            out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        return out

    print("\n=== Top combinations by t-stat ===")
    print(_format(top_t).to_string(index=False))

    print("\n=== Top combinations by mean return ===")
    print(_format(top_mean).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build summary table across all entry_day and holding windows from long-format trade results."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/entry_lag_detail_2018_2026.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--entry-col",
        type=str,
        default=None,
        help="Entry day column name, e.g. entry_lag",
    )
    parser.add_argument(
        "--hold-col",
        type=str,
        default=None,
        help="Holding window column name, e.g. hold_days",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default=None,
        help="Date column name, e.g. signal_date or entry_date",
    )
    parser.add_argument(
        "--ret-col",
        type=str,
        default=None,
        help="Return column name, e.g. pnl",
    )
    parser.add_argument(
        "--save-summary",
        type=str,
        default="data/processed/summary_by_window_and_entry_day.csv",
        help="Output path for overall summary CSV",
    )
    parser.add_argument(
        "--save-yearly",
        type=str,
        default="data/processed/yearly_summary_by_window_and_entry_day.csv",
        help="Output path for yearly summary CSV",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    if df.empty:
        print("Input CSV is empty.")
        sys.exit(1)

    entry_col = args.entry_col or detect_column(df.columns.tolist(), ENTRY_DAY_CANDIDATES)
    hold_col = args.hold_col or detect_column(df.columns.tolist(), HOLDING_DAY_CANDIDATES)
    date_col = args.date_col or detect_column(df.columns.tolist(), DATE_CANDIDATES)
    ret_col = args.ret_col or detect_column(df.columns.tolist(), RET_CANDIDATES)

    missing = []
    if entry_col is None:
        missing.append("entry column")
    if hold_col is None:
        missing.append("holding window column")
    if date_col is None:
        missing.append("date column")
    if ret_col is None:
        missing.append("return column")

    if missing:
        print(f"Could not detect: {', '.join(missing)}")
        print("Available columns:")
        for c in df.columns:
            print(f"  - {c}")
        print("\nYou can rerun manually, for example:")
        print(
            "python build_summary_by_window.py "
            "--input entry_lag_detail_2018_2026.csv "
            "--entry-col entry_lag --hold-col hold_days --date-col signal_date --ret-col pnl"
        )
        sys.exit(1)

    print(f"Reading input: {input_path}")
    print(f"Using entry column: {entry_col}")
    print(f"Using holding column: {hold_col}")
    print(f"Using date column: {date_col}")
    print(f"Using return column: {ret_col}")

    overall_summary = build_overall_summary(
        df,
        entry_col=entry_col,
        hold_col=hold_col,
        ret_col=ret_col,
    )

    yearly_summary = build_yearly_summary(
        df,
        entry_col=entry_col,
        hold_col=hold_col,
        date_col=date_col,
        ret_col=ret_col,
    )

    print_pretty_summary(overall_summary)
    print_top_combinations(overall_summary, top_n=5)

    overall_path = Path(args.save_summary)
    yearly_path = Path(args.save_yearly)

    overall_summary.to_csv(overall_path, index=False)
    yearly_summary.to_csv(yearly_path, index=False)

    print(f"\nSaved overall summary to: {overall_path}")
    print(f"Saved yearly summary to: {yearly_path}")


if __name__ == "__main__":
    main()