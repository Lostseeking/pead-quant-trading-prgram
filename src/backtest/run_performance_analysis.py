from pathlib import Path
import argparse
import sys

import pandas as pd

from performance_metrics import compute_metrics, compute_yearly_metrics, format_metrics


DEFAULT_INPUTS = [
    "data/processed/validated_signals_2018_2026.csv",
    "data/processed/entry_lag_detail_2018_2026.csv",
    "data/processed/pead_signals_2018_2026.csv",
    "data/processed/pead_signals_3y.csv",
]


DATE_CANDIDATES = [
    "date",
    "entry_date",
    "exit_date",
    "earnings_date",
    "report_date",
]


RET_CANDIDATES = [
    "ret",
    "return",
    "pnl",
    "trade_return",
    "holding_return",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "1d",
    "3d",
    "5d",
    "10d",
    "20d",
    "pnl_1d",
    "pnl_3d",
    "pnl_5d",
    "pnl_10d",
    "pnl_20d",
]


def detect_existing_file() -> Path | None:
    for name in DEFAULT_INPUTS:
        p = Path(name)
        if p.exists():
            return p
    return None


def detect_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def print_available_columns(df: pd.DataFrame) -> None:
    print("\nAvailable columns:")
    for c in df.columns:
        print(f"  - {c}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run performance analysis on an earnings_scanner CSV.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV. If omitted, script tries common filenames automatically.",
    )
    parser.add_argument(
        "--ret-col",
        type=str,
        default=None,
        help="Return column name, e.g. ret / ret_20d / pnl_5d",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default=None,
        help="Date column name, e.g. date / entry_date / earnings_date",
    )
    parser.add_argument(
        "--save-yearly",
        type=str,
        default="data/processed/yearly_performance_metrics.csv",
        help="Output CSV path for yearly metrics",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else detect_existing_file()

    if input_path is None or not input_path.exists():
        print("Could not find an input CSV automatically.")
        print("Please specify one manually, for example:")
        print("  python run_performance_analysis.py --input validated_signals_2018_2026.csv --ret-col ret --date-col date")
        sys.exit(1)

    print(f"Reading input: {input_path}")
    df = pd.read_csv(input_path)

    if df.empty:
        print("Input CSV is empty.")
        sys.exit(1)

    date_col = args.date_col or detect_column(df.columns.tolist(), DATE_CANDIDATES)
    ret_col = args.ret_col or detect_column(df.columns.tolist(), RET_CANDIDATES)

    if ret_col is None:
        print("Could not automatically detect return column.")
        print_available_columns(df)
        print("\nPlease rerun with --ret-col, for example:")
        print("  python run_performance_analysis.py --input your_file.csv --ret-col ret --date-col date")
        sys.exit(1)

    if date_col is None:
        print("Could not automatically detect date column.")
        print_available_columns(df)
        print("\nPlease rerun with --date-col, for example:")
        print("  python run_performance_analysis.py --input your_file.csv --ret-col ret --date-col entry_date")
        sys.exit(1)

    print(f"Using date column: {date_col}")
    print(f"Using return column: {ret_col}")

    overall = compute_metrics(df, ret_col=ret_col)
    if overall is None:
        print("No valid returns found after cleaning the return column.")
        sys.exit(1)

    print()
    print(format_metrics(overall))

    yearly = compute_yearly_metrics(df, date_col=date_col, ret_col=ret_col)

    if yearly.empty:
        print("\nNo yearly metrics could be computed.")
        sys.exit(0)

    pretty_yearly = yearly.copy()
    for col in ["mean", "median", "std", "win_rate"]:
        pretty_yearly[col] = pretty_yearly[col].map(lambda x: f"{x:.2%}")
    for col in ["sharpe", "t_stat"]:
        pretty_yearly[col] = pretty_yearly[col].map(lambda x: f"{x:.4f}")

    print("\n=== Yearly Performance Metrics ===")
    print(pretty_yearly.to_string(index=False))

    save_path = Path(args.save_yearly)
    yearly.to_csv(save_path, index=False)
    print(f"\nSaved yearly metrics to: {save_path}")


if __name__ == "__main__":
    main()