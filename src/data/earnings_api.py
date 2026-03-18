import requests
from config import FINNHUB_API_KEY

def get_latest_earnings(symbol: str):
    url = "https://finnhub.io/api/v1/stock/earnings"
    params = {"symbol": symbol, "token": FINNHUB_API_KEY}
    data = requests.get(url, params=params).json()
    if not data:
        return None
    return data[0]

def get_earnings_calendar(date_from: str, date_to: str):
    """
    Return the lists of earnings for different companys in a date range

    """
    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {"from": date_from, "to": date_to, "token": FINNHUB_API_KEY}

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    return data.get("earningsCalendar", [])

def get_historical_earnings(symbol: str):
    """
    Return historical earnings records for a symbol
    """

    url = "https://finnhub.io/api/v1/stock/earnings"

    params = {
        "symbol": symbol,
        "token": FINNHUB_API_KEY
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()

    return r.json()