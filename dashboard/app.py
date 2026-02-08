import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Crypto Price Dashboard", layout="wide")

st.title("📊 Cryptocurrency Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/processed_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# Sidebar options
st.sidebar.header("Select Options")

coin = st.sidebar.selectbox(
    "Select Cryptocurrency",
    sorted(df["coin"].unique())
)

metric = st.sidebar.selectbox(
    "Select Metric",
    ["close", "open", "high", "low"]
)

filtered_df = df[df["coin"] == coin]

# Line chart
st.subheader(f"{coin} - {metric.capitalize()} Price Trend")

fig, ax = plt.subplots()
ax.plot(filtered_df["date"], filtered_df[metric])
ax.set_xlabel("Date")
ax.set_ylabel(f"{metric.capitalize()} Price (USD)")
ax.grid(True)

st.pyplot(fig)

# Summary statistics
st.subheader("Summary Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Maximum", f"${filtered_df[metric].max():.2f}")
col2.metric("Minimum", f"${filtered_df[metric].min():.2f}")
col3.metric("Average", f"${filtered_df[metric].mean():.2f}")

# Volume chart
st.subheader("Trading Volume")

fig2, ax2 = plt.subplots()
ax2.bar(filtered_df["date"], filtered_df["volume"])
ax2.set_xlabel("Date")
ax2.set_ylabel("Volume")

st.pyplot(fig2)
