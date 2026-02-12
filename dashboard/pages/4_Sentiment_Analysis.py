import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("🧠 Sentiment Analysis Dashboard")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()
# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    sentiment_df = pd.read_csv("../data/sentiment/processed/news_processed.csv")
    price_df = pd.read_csv("../data/processed/processed_data.csv")

    # Ensure date is datetime
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    price_df["date"] = pd.to_datetime(price_df["date"])

    return sentiment_df, price_df

sentiment_df, price_df = load_data()

# ==============================
# Sidebar Filters
# ==============================
st.sidebar.header("Filters")

coins = sorted(sentiment_df["coin"].unique())
selected_coin = st.sidebar.selectbox("Select Coin", coins)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [
        sentiment_df["date"].min().date(),
        sentiment_df["date"].max().date()
    ]
)

# Apply filters
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

sent_df = sentiment_df[
    (sentiment_df["coin"] == selected_coin) &
    (sentiment_df["date"] >= start_date) &
    (sentiment_df["date"] <= end_date)
]

price_df = price_df[
    (price_df["coin"] == selected_coin) &
    (price_df["date"] >= start_date) &
    (price_df["date"] <= end_date)
]

# ==============================
# KPIs
# ==============================
col1, col2, col3 = st.columns(3)

if not sent_df.empty:
    avg_sentiment = sent_df["sentiment_score"].mean()
    positive_pct = (sent_df["sentiment_label"] == "Positive").mean() * 100
else:
    avg_sentiment = 0
    positive_pct = 0

col1.metric("Avg Sentiment Score", f"{avg_sentiment:.3f}")
col2.metric("Positive Days (%)", f"{positive_pct:.1f}%")
col3.metric("Total News Articles", len(sent_df))

st.divider()

# ==============================
# Sentiment Over Time
# ==============================
st.subheader("📈 Sentiment Over Time")

if not sent_df.empty:
    sent_time = (
        sent_df.groupby("date")["sentiment_score"]
        .mean()
        .reset_index()
    )

    fig_sent_time = px.line(
        sent_time,
        x="date",
        y="sentiment_score",
        title=f"Average Sentiment Over Time ({selected_coin})"
    )
    st.plotly_chart(fig_sent_time, use_container_width=True)
else:
    st.warning("No sentiment data available for the selected coin and date range.")

# ==============================
# Sentiment vs Price Relationship
# ==============================
st.subheader("🔗 Sentiment vs Price Relationship")

MIN_POINTS = 3  # Minimum points to plot

merged = pd.DataFrame()

if not sent_df.empty and not price_df.empty:
    # Ensure datetime type for grouping
    sent_df["date"] = pd.to_datetime(sent_df["date"])
    price_df["date"] = pd.to_datetime(price_df["date"])

    # --- Weekly aggregation ---
    weekly_sent = sent_df.groupby(pd.Grouper(key="date", freq="W"))["sentiment_score"].mean().reset_index()
    weekly_price = price_df.groupby(pd.Grouper(key="date", freq="W"))["close"].mean().reset_index()

    merged = pd.merge(weekly_sent, weekly_price, on="date", how="inner")

    # --- Fallback to daily if weekly merge too small ---
    if len(merged) < MIN_POINTS:
        daily_sent = sent_df.groupby("date")["sentiment_score"].mean().reset_index()
        daily_price = price_df.groupby("date")["close"].mean().reset_index()
        merged = pd.merge(daily_sent, daily_price, on="date", how="inner")

if merged.empty or len(merged) < MIN_POINTS:
    st.warning("Not enough overlapping sentiment and price data to plot relationship. Try selecting a longer date range or a different coin.")
else:
    fig_sent_price = px.scatter(
        merged,
        x="sentiment_score",
        y="close",
        trendline="ols",
        title=f"Sentiment Score vs Closing Price ({selected_coin})",
        labels={
            "sentiment_score": "Average Sentiment Score",
            "close": "Closing Price"
        }
    )
    st.plotly_chart(fig_sent_price, use_container_width=True)

# ==============================
# Sentiment Label Distribution
# ==============================
st.subheader("📊 Sentiment Distribution")

if not sent_df.empty:
    label_counts = sent_df["sentiment_label"].value_counts().reset_index()
    label_counts.columns = ["Sentiment", "Count"]

    fig_labels = px.bar(
        label_counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        title=f"Sentiment Label Distribution ({selected_coin})"
    )
    st.plotly_chart(fig_labels, use_container_width=True)

    # ==============================
    # Insight Box
    # ==============================
    dominant = label_counts.iloc[0]["Sentiment"]
    st.info(
        f"📌 **Insight:** {selected_coin} sentiment is predominantly **{dominant.upper()}** "
        "during the selected period. Strong sentiment shifts often precede short-term "
        "price movements, indicating sentiment as a useful leading indicator."
    )
else:
    st.warning("No sentiment data available for the selected coin and date range.")
