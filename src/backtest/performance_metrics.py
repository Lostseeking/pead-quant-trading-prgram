import math
from typing import Optional, Dict, Any

import pandas as pd


def compute_metrics(df: pd.DataFrame, ret_col: str = "ret") -> Optional[Dict[str, Any]]:
    """
    Compute core performance statistics for a return column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a return column.
    ret_col : str
        Name of the return column.

    Returns
    -------
    dict or None
        Dictionary of metrics, or None if no valid returns are found.
    """
    if ret_col not in df.columns:
        raise KeyError(f"Return column '{ret_col}' not found in DataFrame.")

    rets = pd.to_numeric(df[ret_col], errors="coerce").dropna()

    n = len(rets)
    if n == 0:
        return None

    mean = rets.mean()
    median = rets.median()
    std = rets.std(ddof=1) if n > 1 else 0.0
    win_rate = (rets > 0).mean()

    sharpe = mean / std if std > 0 else 0.0
    t_stat = mean / (std / math.sqrt(n)) if std > 0 and n > 1 else 0.0

    return {
        "count": int(n),
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "sharpe": float(sharpe),
        "t_stat": float(t_stat),
        "win_rate": float(win_rate),
    }


def compute_yearly_metrics(
    df: pd.DataFrame,
    date_col: str = "date",
    ret_col: str = "ret",
) -> pd.DataFrame:
    """
    Compute yearly performance metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_col : str
        Date column name.
    ret_col : str
        Return column name.

    Returns
    -------
    pd.DataFrame
        Yearly metrics table.
    """
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in DataFrame.")

    working = df.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.dropna(subset=[date_col])

    working["year"] = working[date_col].dt.year

    rows = []
    for year, group in working.groupby("year"):
        metrics = compute_metrics(group, ret_col=ret_col)
        if metrics is not None:
            metrics["year"] = int(year)
            rows.append(metrics)

    if not rows:
        return pd.DataFrame(columns=["year", "count", "mean", "median", "std", "sharpe", "t_stat", "win_rate"])

    yearly_df = pd.DataFrame(rows)
    yearly_df = yearly_df[["year", "count", "mean", "median", "std", "sharpe", "t_stat", "win_rate"]]
    yearly_df = yearly_df.sort_values("year").reset_index(drop=True)
    return yearly_df


def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    Format a metrics dictionary into a readable text block.
    """
    if metrics is None:
        return "No valid returns found."

    lines = [
        "=== Overall Performance Summary ===",
        f"count   : {metrics['count']}",
        f"mean    : {metrics['mean']:.6f}  ({metrics['mean']:.2%})",
        f"median  : {metrics['median']:.6f}  ({metrics['median']:.2%})",
        f"std     : {metrics['std']:.6f}  ({metrics['std']:.2%})",
        f"sharpe  : {metrics['sharpe']:.4f}",
        f"t_stat  : {metrics['t_stat']:.4f}",
        f"win_rate: {metrics['win_rate']:.4f}  ({metrics['win_rate']:.2%})",
    ]
    return "\n".join(lines)