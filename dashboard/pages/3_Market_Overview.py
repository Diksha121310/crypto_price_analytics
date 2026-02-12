import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Market Overview", layout="wide")

st.title("📊 Market Overview")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()
# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("../data/processed/processed_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# ==============================
# Sidebar Filters
# ==============================
st.sidebar.header("Filters")

coins = sorted(df["coin"].unique())
selected_coins = st.sidebar.multiselect(
    "Select Coins",
    coins,
    default=coins[:3]
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["date"].min(), df["date"].max()]
)

# Apply filters
filtered_df = df[
    (df["coin"].isin(selected_coins)) &
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1]))
]

# ==============================
# KPIs
# ==============================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Coins", filtered_df["coin"].nunique())
col2.metric("Highest Close", f"{filtered_df['close'].max():,.2f}")
col3.metric("Lowest Close", f"{filtered_df['close'].min():,.2f}")
col4.metric("Average Close", f"{filtered_df['close'].mean():,.2f}")

st.divider()

# ==============================
# Price Trend
# ==============================
st.subheader("📈 Price Trend")

fig_price = px.line(
    filtered_df,
    x="date",
    y="close",
    color="coin",
    title="Closing Price Trend"
)

st.plotly_chart(fig_price, use_container_width=True)

# ==============================
# Daily Returns
# ==============================
st.subheader("📉 Daily Returns")

filtered_df = filtered_df.sort_values(["coin", "date"])
filtered_df["daily_return"] = filtered_df.groupby("coin")["close"].pct_change()

fig_returns = px.line(
    filtered_df,
    x="date",
    y="daily_return",
    color="coin",
    title="Daily Returns"
)

st.plotly_chart(fig_returns, use_container_width=True)

# ==============================
# Rolling Volatility (14D)
# ==============================
st.subheader("⚠️ 14-Day Rolling Volatility")

filtered_df["volatility_14d"] = (
    filtered_df.groupby("coin")["daily_return"]
    .rolling(14)
    .std()
    .reset_index(level=0, drop=True)
)

fig_vol = px.line(
    filtered_df,
    x="date",
    y="volatility_14d",
    color="coin",
    title="14-Day Rolling Volatility"
)

st.plotly_chart(fig_vol, use_container_width=True)

# ==============================
# Insight Box
# ==============================
st.info(
    "📌 **Insight:** Periods of increased volatility often align with sharp price movements. "
    "BTC and ETH show relatively lower volatility compared to altcoins, indicating higher stability."
)
