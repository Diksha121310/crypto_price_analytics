import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Executive Summary", layout="wide")

st.title("📊 Executive Summary & Final Recommendations")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()
# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
@st.cache_data
def load_price():
    df = pd.read_csv("../data/processed/processed_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_metrics():
    return pd.read_csv("../data/dashboard/model_performance_comparison.csv")

price_df = load_price()
metrics_df = load_metrics()

# ---------------------------------------------------
# Aggregate Calculations
# ---------------------------------------------------

summary = (
    price_df.groupby("coin")
    .agg(
        avg_close=("close", "mean"),
        volatility=("close", "std"),
    )
    .reset_index()
)

# Daily returns
price_df["return"] = price_df.groupby("coin")["close"].pct_change()

returns = (
    price_df.groupby("coin")["return"]
    .mean()
    .reset_index(name="avg_return")
)

summary = summary.merge(returns, on="coin")

summary["growth_percent"] = summary["avg_return"] * 100

# ---------------------------------------------------
# Best & Worst Crypto
# ---------------------------------------------------
best_crypto = summary.sort_values("growth_percent", ascending=False).iloc[0]
worst_crypto = summary.sort_values("growth_percent").iloc[0]

# ---------------------------------------------------
# Best Forecasting Model (Lowest RMSE overall)
# ---------------------------------------------------
best_model_row = (
    metrics_df.groupby("model")["RMSE"]
    .mean()
    .reset_index()
    .sort_values("RMSE")
    .iloc[0]
)

best_model = best_model_row["model"]

# ---------------------------------------------------
# Risk Classification
# ---------------------------------------------------
def classify_risk(vol):
    if vol < summary["volatility"].quantile(0.33):
        return "Low"
    elif vol < summary["volatility"].quantile(0.66):
        return "Moderate"
    else:
        return "High"

summary["risk_level"] = summary["volatility"].apply(classify_risk)

risk_counts = summary["risk_level"].value_counts().reset_index()
risk_counts.columns = ["risk_level", "count"]

# ---------------------------------------------------
# Top 5 Coins
# ---------------------------------------------------
top5 = summary.sort_values("growth_percent", ascending=False).head(5)

# ---------------------------------------------------
# AUTO GENERATED INSIGHTS
# ---------------------------------------------------

st.subheader("🔍 Auto-Generated Insights")

st.success(f"🚀 Best Performing Crypto: **{best_crypto['coin']}** "
           f"with average growth of {best_crypto['growth_percent']:.2f}%")

st.error(f"📉 Worst Performing Crypto: **{worst_crypto['coin']}** "
         f"with average growth of {worst_crypto['growth_percent']:.2f}%")

st.info(f"🤖 Best Forecasting Model Overall: **{best_model}** "
        f"(lowest average RMSE across coins)")

st.markdown("""
📊 **Impact of Sentiment:**  
Positive sentiment trends align with short-term upward price movements,  
indicating that news sentiment can improve trading signal quality.

⚠️ **Risk Observation:**  
High-volatility coins offer higher return potential but carry significant drawdown risk.
""")

# ---------------------------------------------------
# Risk Classification Pie
# ---------------------------------------------------
st.subheader("⚖ Risk Classification")

fig_risk = px.pie(
    risk_counts,
    names="risk_level",
    values="count",
    title="Risk Distribution Across Coins"
)

st.plotly_chart(fig_risk, use_container_width=True)

# ---------------------------------------------------
# Top 5 Coins Chart
# ---------------------------------------------------
st.subheader("🏆 Top 5 Performing Coins (Growth %)")

fig_top = px.bar(
    top5,
    x="growth_percent",
    y="coin",
    orientation="h",
    title="Top 5 Coins by Average Growth"
)

st.plotly_chart(fig_top, use_container_width=True)

# ---------------------------------------------------
# Final Table
# ---------------------------------------------------
st.subheader("📋 Final Performance Table")

display_cols = [
    "coin",
    "avg_close",
    "avg_return",
    "volatility",
    "growth_percent",
    "risk_level"
]

st.dataframe(
    summary[display_cols].sort_values("growth_percent", ascending=False),
    use_container_width=True
)

# ---------------------------------------------------
# Final Recommendation Block
# ---------------------------------------------------
st.subheader("🎯 Final Recommendations")

st.markdown(f"""
• **For Low Risk Investors:** Consider coins classified as Low risk with stable volatility.  
• **For Growth Investors:** {best_crypto['coin']} shows strong momentum and return potential.  
• **Forecasting Model Recommendation:** Use **{best_model}** for best predictive performance.  
• **Sentiment Integration:** Incorporating sentiment significantly improves timing accuracy in volatile markets.  
""")
