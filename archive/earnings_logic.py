from datetime import datetime


def classify_eps(actual, estimate, tol=1e-6):
    if actual is None or estimate is None:
        return None

    if actual > estimate + tol:
        return "BEAT"
    elif actual < estimate - tol:
        return "MISS"
    else:
        return "INLINE"


def build_earnings_record(symbol, earnings_item):
    """
    Convert raw API earnings data into a standardized record.
    """

    actual = earnings_item.get("epsActual")
    estimate = earnings_item.get("epsEstimate")

    result = classify_eps(actual, estimate)

    surprise = None
    surprise_pct = None

    if actual is not None and estimate is not None:
        surprise = actual - estimate

        if estimate != 0:
            surprise_pct = (surprise / abs(estimate)) * 100

    return {
        "symbol": symbol,
        "date": earnings_item.get("date"),
        "year": earnings_item.get("year"),
        "quarter": earnings_item.get("quarter"),
        "eps_actual": actual,
        "eps_estimate": estimate,
        "surprise": surprise,
        "surprise_pct": surprise_pct,
        "result": result,
        "detected_at": datetime.utcnow().isoformat()
    }