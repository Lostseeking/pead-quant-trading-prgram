import os

FMP_API_KEY = os.getenv("FMP_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

if not FMP_API_KEY:
    raise ValueError("Missing FMP_API_KEY environment variable")



