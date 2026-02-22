import streamlit as st
from data_utils import get_user_profile_data, plot_pr_only, get_pbs, format_seconds

st.set_page_config(page_title="User Profile", layout="wide")

#  Security Check
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please log in on the Home page first.")
    st.stop()

st.title("Athlete Profile")

# Fetch Profile & Workout Data
email = st.session_state.garmin_email
password = st.session_state.garmin_password

with st.spinner("Loading athlete profile..."):
    profile = get_user_profile_data(email, password)
    # We reuse the dataframe already stored in session state from the home page
    df = st.session_state.get("df_data")

if profile:
    # User info
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f"### {profile['Full Name']}")
        st.caption(f"@{profile['Username']}")
        st.write(f"📍 {profile['Location'] or 'Global'}")

    with col2:
        st.metric("Weight", f"{profile['Weight (lbs)']} lbs")
        st.metric("Height", f"{profile['Height (ft)']} ft")

    with col3:
        if df is not None:
            total_activities = len(df)
            st.metric("Recent Activities", total_activities)
            st.metric("Gender", profile['Gender'].capitalize())

    st.divider()

    # Personal Bests
    st.subheader("Current Personal Bests")
    
    if df is not None and not df.empty:
        pbs_df = get_pbs(df)
        
        if not pbs_df.empty:
            # Create 3 columns for 1 Mile, 5K, and 10K
            pb_cols = st.columns(3)
            
            # Map distance to columns
            dist_map = {"1 Mile": 0, "5K": 1, "10K": 2}
            
            for _, row in pbs_df.iterrows():
                idx = dist_map.get(row['Distance'])
                with pb_cols[idx]:
                    pretty_time = format_seconds(row['T'])
                    st.metric(
                        label=row['Distance'], 
                        value=pretty_time, 
                        help=f"Achieved on {row['Date'].strftime('%Y-%m-%d')}"
                    )
        else:
            st.info("No PBs recorded yet for the standard distances.")
    else:
        st.error("No activity data found. Please fetch data on the Home page.")

else:
    st.error("Could not retrieve Garmin profile. Please check your connection.")