import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(page_title="Forecast & Prediction Analysis", layout="wide")
st.title("📈 Forecast & Prediction Analysis")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()
# ==============================
# Load data
# ==============================
TEST_PATH = "../data/dashboard/test_set_predictions.csv"
FORECAST_PATH = "../data/dashboard/30_day_price_forecast.csv"

if not (os.path.exists(TEST_PATH) and os.path.exists(FORECAST_PATH)):
    st.error("❌ Required CSVs not found. Run forecasting notebook first.")
    st.stop()

test_df = pd.read_csv(TEST_PATH)
forecast_df = pd.read_csv(FORECAST_PATH)

# Parse historical dates
test_df["date"] = pd.to_datetime(test_df["date"])

# ==============================
# CREATE FORECAST DATES (CRITICAL FIX)
# ==============================
last_hist_date = test_df["date"].max()

forecast_df["date"] = last_hist_date + pd.to_timedelta(
    forecast_df["day"], unit="D"
)

# ==============================
# Sidebar filters
# ==============================
st.sidebar.header("Filters")

coins = sorted(test_df["coin"].unique())
selected_coin = st.sidebar.selectbox("Select Coin", coins)

models = ["ARIMA", "LSTM", "Prophet"]
selected_model = st.sidebar.selectbox("Select Model", models)

# ==============================
# Filter by coin
# ==============================
coin_test_df = test_df[test_df["coin"] == selected_coin].copy()
coin_forecast_df = forecast_df[forecast_df["coin"] == selected_coin].copy()

# ==============================
# Column maps
# ==============================
hist_model_map = {
    "ARIMA": "arima_prediction",
    "LSTM": "lstm_prediction",
    "Prophet": "prophet_prediction"
}

forecast_model_map = {
    "ARIMA": "arima_forecast",
    "LSTM": "lstm_forecast",
    "Prophet": "prophet_forecast"
}

lower_map = {
    "ARIMA": "arima_lower",
    "LSTM": "lstm_lower",
    "Prophet": "prophet_lower"
}

upper_map = {
    "ARIMA": "arima_upper",
    "LSTM": "lstm_upper",
    "Prophet": "prophet_upper"
}

# ==============================
# KPIs
# ==============================
last_actual = coin_test_df["actual_price"].iloc[-1]
last_forecast = coin_forecast_df[forecast_model_map[selected_model]].iloc[-1]

expected_growth = ((last_forecast - last_actual) / last_actual) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Last Actual Price", f"{last_actual:.4f}")
c2.metric("Forecasted Price", f"{last_forecast:.4f}")
c3.metric("Expected Growth %", f"{expected_growth:.2f}%")

st.divider()

# ==============================
# Historical: Actual vs Predicted
# ==============================
st.subheader("📊 Historical Actual vs Predicted")

fig_hist = go.Figure()

fig_hist.add_trace(go.Scatter(
    x=coin_test_df["date"],
    y=coin_test_df["actual_price"],
    mode="lines",
    name="Actual"
))

fig_hist.add_trace(go.Scatter(
    x=coin_test_df["date"],
    y=coin_test_df[hist_model_map[selected_model]],
    mode="lines",
    name=f"{selected_model} Prediction"
))

fig_hist.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified"
)

st.plotly_chart(fig_hist, use_container_width=True)

# ==============================
# Forecast + Confidence Interval
# ==============================
st.subheader("📈 30-Day Forecast with Confidence Interval")

fig_forecast = go.Figure()
st.write(
    coin_test_df[
        ["date", "actual_price", hist_model_map[selected_model]]
    ].head(10)
)

# Forecast line
fig_forecast.add_trace(go.Scatter(
    x=coin_forecast_df["date"],
    y=coin_forecast_df[forecast_model_map[selected_model]],
    mode="lines",
    name=f"{selected_model} Forecast"
))

# Confidence interval (NO SELF-JOIN BUG)
y_lower = coin_forecast_df[lower_map[selected_model]]
y_upper = coin_forecast_df[upper_map[selected_model]]

fig_forecast.add_trace(go.Scatter(
    x=list(coin_forecast_df["date"]) + list(coin_forecast_df["date"][::-1]),
    y=list(y_upper) + list(y_lower[::-1]),
    fill="toself",
    fillcolor="rgba(0,100,255,0.2)",
    line=dict(width=0),
    hoverinfo="skip",
    name="Confidence Interval"
))

fig_forecast.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified"
)

st.plotly_chart(fig_forecast, use_container_width=True)

# ==============================
# Prediction Error
# ==============================
st.subheader("📉 Prediction Error Over Time")

coin_test_df["error"] = (
    coin_test_df["actual_price"]
    - coin_test_df[hist_model_map[selected_model]]
)

fig_error = px.scatter(
    coin_test_df,
    x="date",
    y="error",
    title=f"{selected_model} Prediction Error"
)

st.plotly_chart(fig_error, use_container_width=True)
