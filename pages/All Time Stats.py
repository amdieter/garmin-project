import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from data_utils import get_cached_workout_data, summarize_n_days, plot_weekly_training_time, plot_vo2max_over_time, plot_pr_only, get_training_stress
from style_utils import apply_custom_style
apply_custom_style()

st.set_page_config(page_title="Long Term Analytics", layout="wide")

# UI
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please log in on the Home page first to access analytics.")
    st.stop()

st.title("Long-Term Training Analytics")

# Check if we have data from the Home page
if "df_data" in st.session_state and st.session_state.df_data is not None:
    df = st.session_state.df_data

# Input for N Days
with st.sidebar:
    st.header("Settings")
    n_days = st.number_input("Lookback Period (Days)", min_value=1, max_value=5000, value=365)
    refresh = st.button("🔄 Refresh Data")

# Fetch Data
with st.spinner(f"Analyzing last {n_days} days..."):
    email = st.session_state.garmin_email
    password = st.session_state.garmin_password

    # refresh data
    if refresh:
        st.cache_data.clear()
        
    df_analytics = get_cached_workout_data(n_days, email, password)

if df_analytics is not None and not df_analytics.empty:
    col1, col2 = st.columns(2)

    st.subheader("🏆 Personal Best Progression")
    plot_pr_only(df_analytics)
    
    st.subheader("Aerobic Efficiency")
    plot_vo2max_over_time(df_analytics)

    st.divider()
    
    st.subheader("Weekly Volume")
    plot_weekly_training_time(df_analytics)

    st.divider()

    with st.expander("View Raw Data for this Range"):
        st.dataframe(df_analytics)
else:
    st.error("No running activities found in this date range.")