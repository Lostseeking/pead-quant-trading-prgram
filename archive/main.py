import csv
import json
from datetime import datetime, timedelta, timezone, date

import yfinance as yf

from src.data.earnings_api import get_earnings_calendar
from archive.earnings_logic import build_earnings_record


STATE_PATH = "state.json"


def load_watchlist(path="watchlist.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip()]


def parse_earnings_date(cal):
    """
    Get the earnings date from the yfinance calendar.
    """
    if cal is None:
        return None

    try:
        if hasattr(cal, "index") and "Earnings Date" in cal.index:
            val = cal.loc["Earnings Date"][0]
            if isinstance(val, (list, tuple)) and val:
                val = val[0]
            return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except Exception:
        pass

    try:
        if isinstance(cal, dict) and "Earnings Date" in cal:
            val = cal["Earnings Date"]
            if isinstance(val, (list, tuple)) and val:
                val = val[0]
            return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except Exception:
        pass

    return None


def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def make_event_key(item: dict):
    symbol = (item.get("symbol") or "").upper()
    d = item.get("date")
    y = item.get("year")
    q = item.get("quarter")
    return f"{symbol}-{y}Q{q}-{d}"


def detect_recent_releases(watchlist_set, lookback_minutes=60):
    """
    Detect newly released earnings events from today's earnings calendar.
    """
    today_str = date.today().isoformat()
    cal = get_earnings_calendar(today_str, today_str)

    state = load_state()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=lookback_minutes)

    fired = []

    for item in cal:
        symbol = (item.get("symbol") or "").upper()
        if symbol not in watchlist_set:
            continue

        key = make_event_key(item)
        actual = item.get("epsActual")
        est = item.get("epsEstimate")

        prev = state.get(key, {})
        prev_actual = prev.get("eps_actual")

        if (prev_actual is None) and (actual is not None):
            record = build_earnings_record(symbol, item)
            state[key] = record
            fired.append(record)
        else:
            state[key] = {
                **prev,
                "symbol": symbol,
                "date": item.get("date"),
                "year": item.get("year"),
                "quarter": item.get("quarter"),
                "eps_actual": actual if actual is not None else prev.get("eps_actual"),
                "eps_estimate": est if est is not None else prev.get("eps_estimate"),
            }

    save_state(state)

    recent = []
    for e in fired:
        try:
            t = datetime.fromisoformat(e["detected_at"])
            if t >= cutoff:
                recent.append(e)
        except Exception:
            recent.append(e)

    return recent


def main():
    tickers = load_watchlist("watchlist.txt")
    print(f"Loaded {len(tickers)} tickers.\n")

    today = datetime.now().date()
    cutoff = today + timedelta(days=7)

    upcoming = []

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            ed = parse_earnings_date(t.calendar)
        except Exception as e:
            print(f"{ticker}: error fetching calendar ({e})")
            continue

        if ed is None:
            continue

        ed_date = ed.date()

        if today <= ed_date <= cutoff:
            upcoming.append((ed_date, ticker))

    print("\n=== Earnings in next 7 days ===")
    if not upcoming:
        print("(none)")
    else:
        for d, tk in sorted(upcoming):
            days_left = (d - today).days
            print(f"⚠ {tk} in {days_left} day(s) on {d}")

    today_str = today.isoformat()
    out_name = f"earnings_{today_str}.csv"

    with open(out_name, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Ticker", "Earnings Date", "Days Left"])
        for d, tk in sorted(upcoming):
            days_left = (d - today).days
            writer.writerow([tk, d, days_left])

    print(f"\nSaved CSV to: {out_name}")

    watchlist_set = set(tickers)
    fired = detect_recent_releases(watchlist_set, lookback_minutes=60)

    print("\n=== Newly released within last 60 minutes (detected) ===")
    if not fired:
        print("(none)")
    else:
        for e in fired:
            surprise_pct_str = f"{e['surprise_pct']:.2f}%" if e["surprise_pct"] is not None else "N/A"
            print(
                f"{e['symbol']}: actual={e['eps_actual']} est={e['eps_estimate']} "
                f"surprise={e['surprise']} surprise_pct={surprise_pct_str} "
                f"=> {e['result']} (detected_at={e['detected_at']})"
            )

    results_name = f"earnings_results_{today_str}.csv"
    write_header = False

    try:
        with open(results_name, "r", encoding="utf-8") as _:
            pass
    except FileNotFoundError:
        write_header = True

    with open(results_name, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "detected_at",
                "ticker",
                "eps_actual",
                "eps_estimate",
                "surprise",
                "surprise_pct",
                "result",
                "event_date",
                "year",
                "quarter"
            ])

        for e in fired:
            w.writerow([
                e["detected_at"],
                e["symbol"],
                e["eps_actual"],
                e["eps_estimate"],
                e["surprise"],
                e["surprise_pct"],
                e["result"],
                e.get("date"),
                e.get("year"),
                e.get("quarter")
            ])

    if fired:
        print(f"\nSaved results to: {results_name}")


if __name__ == "__main__":
    main()

    