import streamlit as st
from data_utils import plot_activity_map, get_cached_workout_data 
from style_utils import apply_custom_style
apply_custom_style()

st.set_page_config(page_title="Activity Map", layout="wide")

# Security Check
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please log in on the Home page first.")
    st.stop()

st.title("Route Start Points")
st.write("This map shows the starting locations of your running activities.")

# Get data from session state
email = st.session_state.garmin_email
password = st.session_state.garmin_password
df = get_cached_workout_data(10000, email, password) # Should get all(or nearly all) data

if df is not None and not df.empty:

    plot_activity_map(df)
    
    st.info("💡 Click on the markers to see the activity name and distance.")
else:
    st.error("No activity data found. Please fetch data on the Home page.")