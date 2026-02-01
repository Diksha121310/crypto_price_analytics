import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from pathlib import Path

# SETUP
nltk.download("vader_lexicon")

RAW_PATH = Path("../data/sentiment/raw/news_raw.csv")
PROCESSED_PATH = Path("../data/sentiment/processed/news_processed.csv")

PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

# LOAD DATA
print("📥 Loading raw news data...")
df = pd.read_csv(RAW_PATH)

print(f"📊 Columns found: {list(df.columns)}")

# FIX DATE COLUMN
df = df.rename(columns={"timestamp": "date"})

df["date"] = pd.to_datetime(df["date"], errors="coerce")


df = df[
    (df["date"] >= "2025-01-01") &
    (df["date"] <= "2025-12-31")
]

print(f"📆 Rows after 2025 filter: {len(df)}")

# CREATE HEADLINE FIELD
df["headline"] = (
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["text"].fillna("")
).str.strip()

df = df[df["headline"] != ""]

# COIN EXTRACTION
COIN_MAP = {
    "BTC": r"\b(bitcoin|btc)\b",
    "ETH": r"\b(ethereum|eth)\b",
    "BNB": r"\b(binance coin|bnb)\b",
    "ADA": r"\b(cardano|ada)\b",
    "SOL": r"\b(solana|sol)\b",
    "XRP": r"\b(ripple|xrp)\b",
    "DOGE": r"\b(dogecoin|doge)\b",
    "DOT": r"\b(polkadot|dot)\b",
    "LTC": r"\b(litecoin|ltc)\b",
    "TRX": r"\b(tron|trx)\b",
}

def detect_coin(text):
    text = text.lower()
    for coin, pattern in COIN_MAP.items():
        if re.search(pattern, text):
            return coin
    return None

df["coin"] = df["headline"].apply(detect_coin)
df = df.dropna(subset=["coin"])

print("Coin distribution:")
print(df["coin"].value_counts())

# SAVE FILTERED RAW DATA
df_raw = df[["date", "coin", "headline", "source_url"]]
df_raw.to_csv(RAW_PATH, index=False)
print("Filtered raw data saved")

# SENTIMENT ANALYSIS
sia = SentimentIntensityAnalyzer()

df["sentiment_score"] = df["headline"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)

def label_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

# FINAL PROCESSED DATA
processed_df = df[
    ["date", "coin", "headline", "sentiment_score", "sentiment_label", "source_url"]
].sort_values("date")

processed_df.to_csv(PROCESSED_PATH, index=False)

print("Processed sentiment saved")
print("SENTIMENT PIPELINE COMPLETED SUCCESSFULLY")
