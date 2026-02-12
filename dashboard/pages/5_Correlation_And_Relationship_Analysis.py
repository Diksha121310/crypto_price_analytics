import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Correlation & Relationship Analysis", layout="wide")
st.title("📊 Correlation & Relationship Analysis Dashboard")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()
# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    price_df = pd.read_csv("../data/processed/processed_data.csv")
    price_df["date"] = pd.to_datetime(price_df["date"])
    return price_df

price_df = load_data()

# ==============================
# Sidebar Filters
# ==============================
st.sidebar.header("Filters")

coins = sorted(price_df["coin"].unique())
selected_coins = st.sidebar.multiselect("Select Coins", coins, default=coins[:3])

date_range = st.sidebar.date_input(
    "Select Date Range",
    [price_df["date"].min().date(), price_df["date"].max().date()]
)

# Filter data
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

filtered_df = price_df[
    (price_df["coin"].isin(selected_coins)) &
    (price_df["date"] >= start_date) &
    (price_df["date"] <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No data available for selected coins and date range.")
    st.stop()

# ==============================
# Compute Metrics
# ==============================
filtered_df['daily_return'] = filtered_df.groupby('coin')['close'].pct_change()
filtered_df['month'] = filtered_df['date'].dt.to_period('M')

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Highest Close", f"{filtered_df['close'].max():.2f}")
col2.metric("Lowest Close", f"{filtered_df['close'].min():.2f}")
col3.metric("Average Close", f"{filtered_df['close'].mean():.2f}")
col4.metric("Average Volume", f"{filtered_df['volume'].mean():.2f}")

st.divider()

# ==============================
# Trend Alignment Across Coins
# ==============================
st.subheader("📈 Price Trend Comparison Across Coins")

fig_trend = px.line(
    filtered_df,
    x='date',
    y='close',
    color='coin',
    title='Price Trend Across Selected Coins'
)
st.plotly_chart(fig_trend, use_container_width=True)

# ==============================
# Price vs Daily Return
# ==============================
st.subheader("🔗 Price vs Daily Return")

fig_scatter = px.scatter(
    filtered_df,
    x='daily_return',
    y='close',
    color='coin',
    trendline="ols",
    title="Price vs Daily Return"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================
# Average Monthly Return by Coin
# ==============================
st.subheader("📊 Average Monthly Return by Coin")

monthly_return = (
    filtered_df.groupby(['coin', 'month'])['daily_return']
    .mean()
    .reset_index()
)
monthly_return['month'] = monthly_return['month'].dt.to_timestamp()

fig_monthly = px.bar(
    monthly_return,
    x='month',
    y='daily_return',
    color='coin',
    barmode='group',
    title="Average Daily Return by Month & Coin",
    labels={'daily_return': 'Average Daily Return', 'month': 'Month'}
)
st.plotly_chart(fig_monthly, use_container_width=True)

# ==============================
# Price Ranking Comparison
# ==============================
st.subheader("🏆 Price Ranking by Coin")

rank_df = (
    filtered_df.groupby('coin')['close']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
rank_df['rank'] = rank_df['close'].rank(ascending=False)

fig_rank = px.bar(
    rank_df,
    x='close',
    y='coin',
    orientation='h',
    text='rank',
    title="Average Closing Price Ranking by Coin",
    labels={'close': 'Average Close', 'coin': 'Coin'}
)
fig_rank.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_rank, use_container_width=True)
