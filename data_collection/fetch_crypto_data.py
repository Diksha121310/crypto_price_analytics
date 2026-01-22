import requests
import pandas as pd
import os
from datetime import datetime

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "SOLUSDT", "DOGEUSDT", "DOTUSDT",
    "LTCUSDT", "TRXUSDT"
]

RAW_PATH = "../data/raw/raw_data.csv"
PROCESSED_PATH = "../data/processed/processed_data.csv"

def fetch_historical_data():
    all_data = []

    for symbol in SYMBOLS:
        print(f"Fetching historical data for {symbol}...")
        url = "https://api.binance.com/api/v3/klines"

        params = {
            "symbol": symbol,
            "interval": "1d",
            "limit": 365
        }

        response = requests.get(url, params=params)
        data = response.json()

        if isinstance(data, dict):
            print(f"Failed for {symbol}")
            continue

        for row in data:
            all_data.append({
                "coin": symbol.replace("USDT", ""),
                "date": datetime.fromtimestamp(row[0] / 1000),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5])
            })

    if not all_data:
        print("No historical data fetched")
        return

    df_raw = pd.DataFrame(all_data)
    df_processed = df_raw.copy()

    # Cleaning
    df_processed.dropna(inplace=True)
    df_processed.sort_values(["coin", "date"], inplace=True)

    os.makedirs("../data/raw", exist_ok=True)
    os.makedirs("../data/processed", exist_ok=True)

    df_raw.to_csv(RAW_PATH, index=False)
    df_processed.to_csv(PROCESSED_PATH, index=False)

    print("Historical data saved successfully")
    print(f"Raw records: {len(df_raw)}")
    print(f"Processed records: {len(df_processed)}")

if __name__ == "__main__":
    fetch_historical_data()
