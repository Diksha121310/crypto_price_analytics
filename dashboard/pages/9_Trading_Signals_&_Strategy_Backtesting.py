import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Trading Signals & Backtesting", layout="wide")

st.title("📈 Trading Signals & Strategy Backtesting")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()
# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("../data/dashboard/sentiment_price_merged.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# =========================
# Sidebar
# =========================
st.sidebar.header("Settings")

selected_coin = st.sidebar.selectbox(
    "Select Coin",
    sorted(df["coin"].unique())
)

threshold = st.sidebar.slider(
    "Sentiment Threshold",
    min_value=0.01,
    max_value=0.50,
    value=0.10,
    step=0.01
)

coin_df = df[df["coin"] == selected_coin].copy()
coin_df = coin_df.sort_values("date").reset_index(drop=True)

# =========================
# Returns Calculation
# =========================
coin_df["returns"] = coin_df["close"].pct_change().fillna(0)

# =========================
# Generate Signals (Crossover Logic)
# =========================
coin_df["signal"] = 0

# Buy when sentiment crosses above threshold
coin_df.loc[
    (coin_df["sentiment_7d"] > threshold) &
    (coin_df["sentiment_7d"].shift(1) <= threshold),
    "signal"
] = 1

# Sell when sentiment crosses below threshold
coin_df.loc[
    (coin_df["sentiment_7d"] < threshold) &
    (coin_df["sentiment_7d"].shift(1) >= threshold),
    "signal"
] = -1

# =========================
# Position Handling
# =========================
coin_df["position"] = 0
current_position = 0

for i in range(len(coin_df)):
    if coin_df.loc[i, "signal"] == 1:
        current_position = 1
    elif coin_df.loc[i, "signal"] == -1:
        current_position = 0

    coin_df.loc[i, "position"] = current_position

# =========================
# Strategy Returns
# =========================
coin_df["strategy_returns"] = (
    coin_df["returns"] * coin_df["position"].shift(1)
).fillna(0)

# Equity Curves
coin_df["buy_hold_equity"] = (1 + coin_df["returns"]).cumprod()
coin_df["strategy_equity"] = (1 + coin_df["strategy_returns"]).cumprod()

# =========================
# BUY / SELL Chart
# =========================
st.subheader("📊 Buy / Sell Signals")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=coin_df["date"],
    y=coin_df["close"],
    mode="lines",
    name="Price"
))

# Buy markers
fig.add_trace(go.Scatter(
    x=coin_df.loc[coin_df["signal"] == 1, "date"],
    y=coin_df.loc[coin_df["signal"] == 1, "close"],
    mode="markers",
    marker=dict(symbol="triangle-up", size=10),
    name="Buy"
))

# Sell markers
fig.add_trace(go.Scatter(
    x=coin_df.loc[coin_df["signal"] == -1, "date"],
    y=coin_df.loc[coin_df["signal"] == -1, "close"],
    mode="markers",
    marker=dict(symbol="triangle-down", size=10),
    name="Sell"
))

st.plotly_chart(fig, use_container_width=True)

# =========================
# Equity Curve
# =========================
st.subheader("📈 Strategy vs Buy & Hold")

equity_fig = go.Figure()

equity_fig.add_trace(go.Scatter(
    x=coin_df["date"],
    y=coin_df["buy_hold_equity"],
    mode="lines",
    name="Buy & Hold"
))

equity_fig.add_trace(go.Scatter(
    x=coin_df["date"],
    y=coin_df["strategy_equity"],
    mode="lines",
    name="Sentiment Strategy"
))

st.plotly_chart(equity_fig, use_container_width=True)

# =========================
# KPIs
# =========================
st.subheader("📊 Strategy KPIs")

total_return = coin_df["strategy_equity"].iloc[-1] - 1

sharpe_ratio = (
    coin_df["strategy_returns"].mean() /
    coin_df["strategy_returns"].std()
) * np.sqrt(252) if coin_df["strategy_returns"].std() != 0 else 0

rolling_max = coin_df["strategy_equity"].cummax()
drawdown = coin_df["strategy_equity"] / rolling_max - 1
max_drawdown = drawdown.min()

col1, col2, col3 = st.columns(3)

col1.metric("Total Return", f"{total_return:.2%}")
col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col3.metric("Max Drawdown", f"{max_drawdown:.2%}")

# =========================
# Insight Box
# =========================
st.info("""
📌 Insight:
The sentiment-based strategy reacts to market mood shifts.
It tends to outperform Buy & Hold during high volatility
but may underperform in strong bull trends.
""")
