import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="7D & 30D Forecast", layout="wide")

st.title("📈 7-Day & 30-Day Forecast Analysis")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()
# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_forecast():
    df = pd.read_csv("../data/dashboard/30_day_price_forecast.csv")
    df.columns = df.columns.str.lower().str.strip()
    return df

forecast_df = load_forecast()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

selected_coin = st.sidebar.selectbox(
    "Select Coin",
    sorted(forecast_df["coin"].unique())
)

selected_model = st.sidebar.selectbox(
    "Select Model",
    ["arima", "lstm", "prophet"]
)

coin_df = forecast_df[forecast_df["coin"] == selected_coin]

# Dynamically build column names
forecast_col = f"{selected_model}_forecast"
lower_col = f"{selected_model}_lower"
upper_col = f"{selected_model}_upper"

# Drop rows where forecast is missing (important for LSTM if empty)
coin_df = coin_df.dropna(subset=[forecast_col])

# -----------------------------
# 7-Day Forecast
# -----------------------------
st.subheader(f"📊 7-Day Forecast — {selected_coin} ({selected_model.upper()})")

forecast_7d = coin_df.head(7)

fig_7 = go.Figure()

fig_7.add_trace(go.Scatter(
    x=forecast_7d["day"],
    y=forecast_7d[forecast_col],
    mode="lines+markers",
    name="Forecast"
))

fig_7.add_trace(go.Scatter(
    x=forecast_7d["day"],
    y=forecast_7d[upper_col],
    mode="lines",
    line=dict(dash="dash"),
    name="Upper Band"
))

fig_7.add_trace(go.Scatter(
    x=forecast_7d["day"],
    y=forecast_7d[lower_col],
    mode="lines",
    line=dict(dash="dash"),
    fill="tonexty",
    name="Lower Band"
))

st.plotly_chart(fig_7, use_container_width=True)

# -----------------------------
# 30-Day Forecast
# -----------------------------
st.subheader("📈 30-Day Forecast — Stability vs Uncertainty")

fig_30 = go.Figure()

fig_30.add_trace(go.Scatter(
    x=coin_df["day"],
    y=coin_df[forecast_col],
    mode="lines",
    name="Forecast"
))

fig_30.add_trace(go.Scatter(
    x=coin_df["day"],
    y=coin_df[upper_col],
    mode="lines",
    line=dict(dash="dash"),
    name="Upper Band"
))

fig_30.add_trace(go.Scatter(
    x=coin_df["day"],
    y=coin_df[lower_col],
    mode="lines",
    line=dict(dash="dash"),
    fill="tonexty",
    name="Lower Band"
))

st.plotly_chart(fig_30, use_container_width=True)

# -----------------------------
# Forecast Table
# -----------------------------
st.subheader("📋 Forecast Table")

st.dataframe(
    coin_df[["day", forecast_col, lower_col, upper_col]],
    use_container_width=True
)
