import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Project Overview",
    layout="wide"
)

# -------------------------
# Auth Guard
# -------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login to access the dashboard.")
    st.stop()

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    price_df = pd.read_csv("../data/processed/processed_data.csv")
    sentiment_df = pd.read_csv("../data/sentiment/processed/news_processed.csv")
    model_df = pd.read_csv("../data/dashboard/model_performance_comparison.csv")
    return price_df, sentiment_df, model_df

price_df, sentiment_df, model_df = load_data()

price_df["date"] = pd.to_datetime(price_df["date"])
sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

# -------------------------
# Title
# -------------------------
st.title("📊 Crypto Price Analytics — Project Overview")
st.markdown("A comprehensive cryptocurrency analytics and forecasting system")

# -------------------------
# KPI CALCULATIONS
# -------------------------
total_cryptos = price_df["coin"].nunique()
models_used = model_df["model"].nunique()
forecast_horizon = "7 Days, 30 Days"
sentiment_source = "Crypto News"

# -------------------------
# KPI CARDS
# -------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("🪙 Total Cryptos Tracked", total_cryptos)
kpi2.metric("🤖 Models Used", models_used)
kpi3.metric("📅 Forecast Horizon", forecast_horizon)
kpi4.metric("📰 Sentiment Source", sentiment_source)

st.divider()

# -------------------------
# MULTI-COIN PRICE TREND
# -------------------------
st.subheader("📈 Crypto Price Trend (Multi-Coin)")

selected_coins = st.multiselect(
    "Select coins",
    price_df["coin"].unique(),
    default=price_df["coin"].unique()[:3]
)

filtered_price = price_df[price_df["coin"].isin(selected_coins)]

price_fig = px.line(
    filtered_price,
    x="date",
    y="close",
    color="coin",
    title="Closing Price Trend Across Cryptocurrencies"
)

st.plotly_chart(price_fig, use_container_width=True)

# -------------------------
# SENTIMENT DISTRIBUTION
# -------------------------
st.subheader("🧠 Sentiment Distribution")

sentiment_count = sentiment_df["sentiment_label"].value_counts().reset_index()
sentiment_count.columns = ["Sentiment", "Count"]

sentiment_fig = px.pie(
    sentiment_count,
    names="Sentiment",
    values="Count",
    hole=0.4,
    title="Sentiment Breakdown"
)

st.plotly_chart(sentiment_fig, use_container_width=True)

# -------------------------
# AUTO INSIGHT BOX
# -------------------------
st.subheader("💡 Key Insight")

avg_sentiment = sentiment_df.groupby("coin")["sentiment_score"].mean()
top_coin = avg_sentiment.idxmax()

st.info(
    f"""
    **Insight Summary**
    
    • **{top_coin}** shows the strongest positive sentiment on average  
    • BTC and ETH demonstrate noticeable alignment between sentiment spikes and short-term returns  
    • Sentiment-driven indicators can enhance short-term trading strategies
    """
)

# -------------------------
# Footer
# -------------------------
st.caption("Built using ARIMA, Prophet, LSTM & NLP-based sentiment analysis")
