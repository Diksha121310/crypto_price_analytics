import streamlit as st

st.set_page_config(page_title="Crypto Analytics Dashboard", layout="wide")

# -----------------------------
# USER CREDENTIALS
# -----------------------------
USERS = {
    "Diksha": "Diksha12",
    "Deep": "Deep13",
    "Harsh": "Harsh14",
    "Aditi": "Aditi15"
}

# -----------------------------
# SESSION STATE
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -----------------------------
# LOGIN SCREEN
# -----------------------------
if not st.session_state.logged_in:
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if USERS.get(username) == password:
            st.session_state.logged_in = True
            st.success("Login Successful 🎉")
            st.switch_page("pages/2_Overview.py")
        else:
            st.error("Invalid username or password")

    st.stop()


selected_page = st.sidebar.radio("Go to", pages)

if selected_page == "2_Overview":
    import pages.Overview

elif selected_page == "3_Market_Overview":
    import pages.Market_Overview

elif selected_page == "4_Sentiment_Analysis":
    import pages.Sentiment_Analysis

elif selected_page == "5_Correlation_And_Relationship_Analysis":
    import pages.Correlation_Analysis

elif selected_page == "6_Forecast_And_Prediction_Analysis":
    import pages.Forecast_Analysis

elif selected_page == "7_Model_Comparison":
    import pages.Model_Comparison

elif selected_page == "8_7d_&_30d_Forecast":
    import pages._7d___30d_Forecast

elif selected_page == "9_Trading_Signals_&_Strategy_Backtesting":
    import pages.Trading_Signals___Strategy_Backtesting

elif selected_page == "10_Insights":
    import pages._10_Executive_Summary

