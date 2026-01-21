import requests
import pandas as pd
from datetime import datetime

COIN_ID = "bitcoin"
VS_CURRENCY = "usd"
DAYS = "365"

url = f"https://api.coingecko.com/api/v3/coins/{COIN_ID}/market_chart"

params = {
    "vs_currency": VS_CURRENCY,
    "days": DAYS
}

response = requests.get(url, params=params)

data = response.json()

prices = data["prices"]
market_caps = data["market_caps"]
volumes = data["total_volumes"]

rows = []

for i in range(len(prices)):
    rows.append({
        "date": datetime.fromtimestamp(prices[i][0] / 1000),
        "price": prices[i][1],
        "market_cap": market_caps[i][1],
        "volume": volumes[i][1]
    })

df = pd.DataFrame(rows)

df.head()

from google.colab import drive
drive.mount('/content/drive')

df.to_csv("btc_raw.csv", index=False)
print("Saved!")

df

df.shape

df.info()

df.isnull().sum()

df = df.dropna()

df.duplicated().sum()

df = df.drop_duplicates()

df = df.sort_values(by="date")

df = df.reset_index(drop=True)

df.head()

df.tail()

df.to_csv("btc_cleaned.csv", index=False)
print("✅ Cleaned data saved!")
