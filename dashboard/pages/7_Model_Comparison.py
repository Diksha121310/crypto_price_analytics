import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Comparison", layout="wide")

st.title("📊 Model Comparison & Best Model Selection")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()
# =========================
# Load Data
# =========================
@st.cache_data
def load_metrics():
    df = pd.read_csv("../data/dashboard/model_performance_comparison.csv")
    return df

model_df = load_metrics()

# Ensure numeric columns
numeric_cols = ["MSE", "RMSE", "MAE", "R2", "MAPE"]
for col in numeric_cols:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("Filters")

selected_coin = st.sidebar.selectbox(
    "Select Coin",
    sorted(model_df["coin"].unique())
)

selected_metric = st.sidebar.selectbox(
    "Select Evaluation Metric",
    ["RMSE", "MAE", "MAPE", "R2"]
)

# Filter data for selected coin
coin_df = model_df[model_df["coin"] == selected_coin].copy()

# =========================
# Error Comparison Chart
# =========================
st.subheader(f"📊 {selected_metric} Comparison — {selected_coin}")

fig = px.bar(
    coin_df,
    x="model",
    y=selected_metric,
    color="model",
    text=selected_metric
)

fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig.update_layout(showlegend=False)

st.plotly_chart(fig, use_container_width=True)

# =========================
# Metrics Table
# =========================
st.subheader("📋 Model Performance Metrics")

display_cols = ["model", "RMSE", "MAE", "MAPE", "R2"]

if selected_metric == "R2":
    sorted_df = coin_df.sort_values(selected_metric, ascending=False)
else:
    sorted_df = coin_df.sort_values(selected_metric, ascending=True)

st.dataframe(
    sorted_df[display_cols],
    use_container_width=True
)

# =========================
# Model Ranking
# =========================
st.subheader("🏆 Model Ranking")

ranking = sorted_df.reset_index(drop=True)
ranking["Rank"] = ranking.index + 1

st.dataframe(
    ranking[["Rank", "model", "RMSE", "MAE", "MAPE", "R2"]],
    use_container_width=True
)

# =========================
# Best Model for Selected Coin
# =========================
if selected_metric == "R2":
    best_row = coin_df.loc[coin_df[selected_metric].idxmax()]
    st.success(
        f"🏆 Best Model for {selected_coin}: "
        f"{best_row['model']} (Highest R2 = {best_row[selected_metric]:,.4f})"
    )
else:
    best_row = coin_df.loc[coin_df[selected_metric].idxmin()]
    st.success(
        f"🏆 Best Model for {selected_coin}: "
        f"{best_row['model']} (Lowest {selected_metric} = {best_row[selected_metric]:,.4f})"
    )

# =========================
# Best Model per Coin (Overall)
# =========================
st.subheader("🥇 Best Model per Coin")

if selected_metric == "R2":
    best_per_coin = (
        model_df.sort_values(selected_metric, ascending=False)
        .groupby("coin")
        .first()
        .reset_index()
    )
else:
    best_per_coin = (
        model_df.sort_values(selected_metric, ascending=True)
        .groupby("coin")
        .first()
        .reset_index()
    )

st.dataframe(
    best_per_coin[["coin", "model", "RMSE", "MAE", "MAPE", "R2"]],
    use_container_width=True
)

# =========================
# Dynamic Insight
# =========================
st.subheader("📌 Insight")

model_counts = best_per_coin["model"].value_counts()

st.info(
    f"""
🔎 Based on **{selected_metric}**:

- {model_counts.to_dict()}
- The most frequently best-performing model across coins is:
  **{model_counts.idxmax()}**
"""
)
