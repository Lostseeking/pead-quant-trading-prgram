# historical_earnings.py

from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
from src.data.earnings_api import get_historical_earnings

from src.data.earnings_api import get_earnings_calendar


def _normalize_date(value) -> date:
    """
    Convert input to datetime.date.
    Supports:
    - date
    - datetime
    - 'YYYY-MM-DD' string
    """
    if value is None:
        return None

    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, str):
        return datetime.fromisoformat(value).date()

    raise TypeError(f"Unsupported date type: {type(value)}")


def _resolve_date_range(start_date=None, end_date=None, years=3):
    """
    Resolve the final [start_date, end_date] range.

    Rules:
    - If end_date is None, use today.
    - If start_date is None, go back `years` years approximately.
    """
    end = _normalize_date(end_date) if end_date is not None else date.today()

    if start_date is not None:
        start = _normalize_date(start_date)
    else:
        start = end - timedelta(days=365 * years)

    if start > end:
        raise ValueError("start_date cannot be later than end_date")

    return start, end


def _month_chunks(start: date, end: date):
    """
    Split a long date range into month-sized chunks:
    [(chunk_start, chunk_end), ...]
    """
    chunks = []
    current = start

    while current <= end:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)

        chunk_end = min(end, next_month - timedelta(days=1))
        chunks.append((current, chunk_end))
        current = next_month

    return chunks


def _fetch_earnings_calendar_chunk(start_date: date, end_date: date) -> List[Dict[str, Any]]:
    """
    Fetch raw earnings calendar data for one chunk.
    """
    return get_earnings_calendar(start_date.isoformat(), end_date.isoformat())


def _filter_symbol_events(raw_events: List[Dict[str, Any]], symbol: str) -> List[Dict[str, Any]]:
    """
    Keep only events for the target symbol.
    """
    symbol = symbol.upper()
    return [
        item for item in raw_events
        if (item.get("symbol") or "").upper() == symbol
    ]


def _deduplicate_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate events by (symbol, date, year, quarter).
    This matters because chunked queries can sometimes overlap logically.
    """
    seen = set()
    unique = []

    for item in events:
        key = (
            (item.get("symbol") or "").upper(),
            item.get("date"),
            item.get("year"),
            item.get("quarter"),
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def _build_event_record(symbol: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one raw Finnhub event into a clean, standardized record.
    """
    actual = item.get("epsActual")
    estimate = item.get("epsEstimate")

    result = None
    surprise = None
    surprise_pct = None

    if actual is not None and estimate is not None:
        tol = 1e-6
        if actual > estimate + tol:
            result = "BEAT"
        elif actual < estimate - tol:
            result = "MISS"
        else:
            result = "INLINE"

        surprise = actual - estimate
        if estimate != 0:
            surprise_pct = (surprise / abs(estimate)) * 100

    return {
        "symbol": symbol.upper(),
        "date": item.get("date"),
        "year": item.get("year"),
        "quarter": item.get("quarter"),
        "eps_actual": actual,
        "eps_estimate": estimate,
        "surprise": surprise,
        "surprise_pct": surprise_pct,
        "result": result,
        "revenue_actual": item.get("revenueActual"),
        "revenue_estimate": item.get("revenueEstimate"),
    }

def get_historical_earnings_events(symbol: str, years: int = 3):
    symbol = symbol.upper()

    start, end = _resolve_date_range(years=years)

    all_events = []
    chunks = _month_chunks(start, end)

    for chunk_start, chunk_end in chunks:
        raw_events = _fetch_earnings_calendar_chunk(chunk_start, chunk_end)
        symbol_events = _filter_symbol_events(raw_events, symbol)
        all_events.extend(symbol_events)

    all_events = _deduplicate_events(all_events)

    records = []
    for item in all_events:
        records.append(_build_event_record(symbol, item))

    records.sort(key=lambda x: x["date"], reverse=True)
    return records


if __name__ == "__main__":
    symbol = "PLTR"
    events = get_historical_earnings_events(symbol, years=3)

    print(f"Found {len(events)} historical earnings events for {symbol}.\n")
    for e in events:
        print(
            f"{e['date']} | {e['symbol']} | "
            f"Q{e['quarter']} {e['year']} | "
            f"actual={e['eps_actual']} estimate={e['eps_estimate']} | "
            f"result={e['result']}"
        )