# price_api.py

from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any

import pandas as pd
import yfinance as yf


def _to_date(value) -> date:
    """
    Convert input to datetime.date.
    Supports date, datetime, or YYYY-MM-DD string.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    raise TypeError(f"Unsupported date type: {type(value)}")


def _today_date() -> date:
    return pd.Timestamp.today().normalize().date()


def _get_history_window(symbol: str, start_date: date, end_date: date):
    """
    Download daily price history for a symbol in [start_date, end_date).

    Safety:
    - Cap end_date to today + 1 day boundary logic for yfinance
    - If requested window is entirely in the future, return empty DataFrame
    - If start_date >= end_date after capping, return empty DataFrame
    - Catch yfinance exceptions and return empty DataFrame
    """
    today = _today_date()

    if start_date > today:
        return pd.DataFrame()

    if end_date > today + timedelta(days=1):
        end_date = today + timedelta(days=1)

    if start_date >= end_date:
        return pd.DataFrame()

    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval="1d",
            auto_adjust=False
        )
        return data
    except Exception:
        return pd.DataFrame()


def get_previous_trading_close(symbol: str, target_date) -> Optional[float]:
    """
    Get the most recent trading day's close strictly before target_date.
    Example:
        target_date = earnings date
        return = previous trading day's close
    """
    target_date = _to_date(target_date)

    # Look back enough days to cross weekends/holidays.
    start_date = target_date - timedelta(days=10)
    end_date = target_date

    data = _get_history_window(symbol, start_date, end_date)
    if data.empty:
        return None

    return float(data["Close"].iloc[-1])


def get_next_trading_close(symbol: str, target_date) -> Optional[float]:
    """
    Get the first trading day's close strictly after target_date.
    Example:
        target_date = earnings date
        return = next trading day's close

    If target_date is too recent and next trading day data does not exist yet,
    return None instead of forcing a future request.
    """
    target_date = _to_date(target_date)
    today = _today_date()

    if target_date >= today:
        return None

    start_date = target_date + timedelta(days=1)
    end_date = target_date + timedelta(days=11)

    data = _get_history_window(symbol, start_date, end_date)
    if data.empty:
        return None

    return float(data["Close"].iloc[0])


def get_trading_close_on_or_after(symbol: str, target_date) -> Optional[float]:
    """
    Get the first available trading close on or after target_date.
    Useful when you want the close of the earnings day if it is a trading day,
    otherwise the next available trading day.
    """
    target_date = _to_date(target_date)
    today = _today_date()

    if target_date > today:
        return None

    start_date = target_date
    end_date = target_date + timedelta(days=10)

    data = _get_history_window(symbol, start_date, end_date)
    if data.empty:
        return None

    return float(data["Close"].iloc[0])


def get_trading_close_on_or_before(symbol: str, target_date) -> Optional[float]:
    """
    Get the last available trading close on or before target_date.
    Useful when target_date may fall on a weekend/holiday.
    """
    target_date = _to_date(target_date)

    start_date = target_date - timedelta(days=10)
    end_date = target_date + timedelta(days=1)

    data = _get_history_window(symbol, start_date, end_date)
    if data.empty:
        return None

    return float(data["Close"].iloc[-1])


def compute_simple_return(price_before: Optional[float], price_after: Optional[float]) -> Optional[float]:
    """
    Compute simple return: (after - before) / before
    """
    if price_before is None or price_after is None:
        return None
    if price_before == 0:
        return None
    return (price_after - price_before) / price_before


def compute_earnings_reaction(symbol: str, earnings_date) -> Dict[str, Any]:
    """
    Compute a simple 1D post-earnings reaction.

    Assumption:
    - pre_close  = previous trading day's close before earnings_date
    - post_close = next trading day's close after earnings_date

    If the earnings date is too recent / future and next trading close is unavailable,
    post_close and reaction_1d will be None.
    """
    earnings_date = _to_date(earnings_date)

    pre_close = get_previous_trading_close(symbol, earnings_date)
    post_close = get_next_trading_close(symbol, earnings_date)
    reaction_1d = compute_simple_return(pre_close, post_close)

    return {
        "symbol": symbol,
        "earnings_date": earnings_date.isoformat(),
        "pre_close": pre_close,
        "post_close": post_close,
        "reaction_1d": reaction_1d,
    }


def compute_multi_horizon_reaction(symbol: str, earnings_date) -> Dict[str, Any]:
    """
    Optional richer version for research:
    - pre_close: previous trading day's close
    - day0_close: close on or after earnings_date
    - day1_close: first trading day's close after earnings_date
    - reaction_day0: day0_close / pre_close - 1
    - reaction_day1: day1_close / pre_close - 1
    """
    earnings_date = _to_date(earnings_date)

    pre_close = get_previous_trading_close(symbol, earnings_date)
    day0_close = get_trading_close_on_or_after(symbol, earnings_date)
    day1_close = get_next_trading_close(symbol, earnings_date)

    reaction_day0 = compute_simple_return(pre_close, day0_close)
    reaction_day1 = compute_simple_return(pre_close, day1_close)

    return {
        "symbol": symbol,
        "earnings_date": earnings_date.isoformat(),
        "pre_close": pre_close,
        "day0_close": day0_close,
        "day1_close": day1_close,
        "reaction_day0": reaction_day0,
        "reaction_day1": reaction_day1,
    }