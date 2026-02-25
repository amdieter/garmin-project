import streamlit as st
import pandas as pd
import garminconnect
from datetime import date, timedelta
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import os
from garth.exc import GarthException

def get_garmin_client(email, password):
    """Saves login info so that IP doesn't get soft-blocked"""
    # Define a directory to save the session tokens
    token_store = "~/.garth" 
    token_store = os.path.expanduser(token_store)
    
    client = garminconnect.Garmin(email, password)
    
    try:
        # Try to load an existing session from the folder
        client.login(token_store)
        print("Login successful using stored tokens.")
    except (GarthException, Exception):
        # If tokens are missing or expired, do a fresh login
        print("Token login failed. Attempting fresh email/password login...")
        try:
            client.login()
            # Save the new tokens so we don't have to do this next time
            client.garth.dump(token_store)
        except Exception as e:
            print(f"Total login failure: {e}")
            return None
            
    return client

# data fetching
def get_workout_dataframe_n_days(n_days, email, password):
    try:
        client = get_garmin_client(email, password)
        client.login()
        
        today = date.today()
        start = str(today - timedelta(days=n_days))
        activities = client.get_activities_by_date(startdate=start, enddate=str(today))

        if not activities:
            return None

        data_list = []
        for act in activities:
            if act["activityType"]["typeKey"] != "running":
                continue
            
            dist_mi = act.get('distance', 0) / 1609.34
            dur_min = act.get('duration', 0) / 60
            
            data_list.append({
                "Activity Name": act.get('activityName'),
                "Date": pd.to_datetime(act.get('startTimeLocal')),
                "Distance (mi)": round(dist_mi, 2),
                "Duration (min)": round(dur_min, 2),
                "Pace_Decimal": round(dur_min / dist_mi, 2) if dist_mi > 0 else 0,
                "Avg HR": act.get('averageHR'),
                "VO2 Max": act.get('vO2MaxValue'),
                "Elev Gain (ft)": round(act.get('elevationGain', 0) * 3.28084, 1),
                "Latitude": act.get('startLatitude'),
                "Longitude": act.get('startLongitude'),
                "Z1_Min": round(act.get('hrTimeInZone_1', 0) / 60, 2),
                "Z2_Min": round(act.get('hrTimeInZone_2', 0) / 60, 2),
                "Z3_Min": round(act.get('hrTimeInZone_3', 0) / 60, 2),
                "Z4_Min": round(act.get('hrTimeInZone_4', 0) / 60, 2),
                "Z5_Min": round(act.get('hrTimeInZone_5', 0) / 60, 2)
            })
        return pd.DataFrame(data_list)
    except Exception as e:
        st.error(f"Garmin Connection Error: {e}")
        return None

@st.cache_data(ttl=3600)
def get_cached_workout_data(days, email, password):
    return get_workout_dataframe_n_days(days, email, password)

# Calculations


def most_active_month(df):
    df['Month'] = df['Date'].dt.month
    monthly_miles = df.groupby("Month")["Distance (mi)"].sum()
    month = monthly_miles.idxmax()
    miles = monthly_miles.max()
    return month, miles

def summarize_n_days(df):
    if isinstance(df, tuple) or df is None or df.empty:
        return {"Total Distance Run (mi)": 0,
            "Total Elevation Gained (ft)": 0,
            "Current VO2 Max": "N/A",
            "Longest Run (mi)": 0}
    
    summary = {
        "Total Distance Run (mi)": df["Distance (mi)"].sum(),
        "Total Elevation Gained (ft)": df["Elev Gain (ft)"].sum(),
        "Current VO2 Max": df["VO2 Max"].dropna().iloc[0] if not df["VO2 Max"].dropna().empty else "N/A",
        "Longest Run (mi)": df["Distance (mi)"].max()
    }
    if len(df) > 1:
        clean_df = df.dropna()
        if len(clean_df) > 1:
            summary["VO2 Max Progress"] = clean_df["VO2 Max"].iloc[0] - clean_df["VO2 Max"].iloc[-1]
    if len(df) >= 31:
        summary["Most Active Month"] = most_active_month(df)
    return summary

# Plotting

def plot_vo2max_over_time(df, color="#10B981"):
    df_plot = df.copy().dropna(subset=["VO2 Max"]).sort_values("Date")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_plot["Date"], df_plot["VO2 Max"], color=color, linewidth=3, marker='o')
    ax.set_title("VO₂ Max Trend")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, width='stretch')

def plot_weekly_training_time(df, color="#6366F1"):
    df_plot = df.copy()
    df_plot["week"] = df_plot["Date"].dt.to_period("W").apply(lambda r: r.start_time)
    wk = df_plot.groupby("week")["Duration (min)"].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(wk["week"], wk["Duration (min)"], color=color, width=5)
    ax.set_title("Weekly Training Volume (Minutes)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig, width='stretch')

def get_pbs(df):
  df["Date"] = pd.to_datetime(df["Date"])

  df["distance_km"] = df["Distance (mi)"] * 1.60934

  df["pace_sec_per_km"] = df["Pace_Decimal"] * 60 / 1.0
  df["pace_sec_per_km"] /= 1.60934

  dcm = {
      "1 Mile": (1.60934, "#3B82F6"),
      "5K": (5.0, "#F59E0B"),
      "10K": (10.0, "#6366F1"),
  }

  rows = []
  for dist_label, (dist_km, color) in dcm.items():
      sub = df[df["distance_km"] >= dist_km].copy()
      if len(sub) == 0:
          continue

      sub["T"] = sub["pace_sec_per_km"] * dist_km
      rows.append(sub[["Date", "T"]].assign(Distance=dist_label, Color=color))

  mlt = pd.concat(rows).sort_values("Date")

  mlt["PR"] = mlt.groupby("Distance")["T"].cummin()
  mlt["isPR"] = mlt["T"] == mlt["PR"]
  final_pbs = mlt[mlt["isPR"]].groupby("Distance").tail(1)
  return final_pbs[['Date', 'T', 'Distance']]

def format_seconds(seconds):
    """Helper to turn raw seconds into MM:SS or HH:MM:SS."""
    if pd.isna(seconds): return "N/A"
    td = timedelta(seconds=int(seconds))
    # Returns 00:00:00, we slice to 00:00 if under an hour
    ts = str(td)
    return ts if seconds >= 3600 else ts[2:]

def plot_pr_only(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    
    # 1. Conversion Logic
    df["distance_km"] = df["Distance (mi)"] * 1.60934
    df["pace_sec_per_km"] = df["Pace_Decimal"] * 60 / 1.60934

    dcm = {
        "1 Mile": (1.609, "#3B82F6"),
        "5K": (5.0, "#F59E0B"),
        "10K": (10.0, "#6366F1"),
    }

    rows = []
    for dist_label, (dist_km, color) in dcm.items():
        # Look for runs that are at least the target distance
        sub = df[df["distance_km"] >= (dist_km - 0.1)].copy() 
        if sub.empty:
            continue

        # Calculate total time for that distance based on pace
        sub["T"] = sub["pace_sec_per_km"] * dist_km
        rows.append(sub[["Date", "T"]].assign(Distance=dist_label, Color=color))

    if not rows:
        st.warning("Not enough data to calculate PRs for these distances.")
        return

    mlt = pd.concat(rows).sort_values("Date")
    
    # 2. PR Calculation: Only keep times that are the "Cumulative Minimum" (the new PR)
    mlt["PR"] = mlt.groupby("Distance")["T"].cummin()
    prs = mlt[mlt["T"] == mlt["PR"]]

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(14, 7))

    for dist_label, (_, color) in dcm.items():
        grp = prs[prs["Distance"] == dist_label]
        if grp.empty:
            continue

        ax.scatter(
            grp["Date"],
            grp["T"],
            s=180, # Made slightly bigger for the stars
            color=color,
            marker="*",
            edgecolor="black",
            linewidth=0.8,
            label=f"{dist_label} PR ★"
        )

    ax.invert_yaxis() # Faster times (lower seconds) are at the top
    ax.set_title("Personal Bests Progression (PRs Only)", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Estimated Time (seconds)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig, width='stretch')

def get_user_profile_data(email, password):
    try:
        client = get_garmin_client(email, password)
        client.login()
        
        user_profile = client.get_user_profile()
        user_data = user_profile.get('userData', {})
        
        return {
            'Full Name': client.full_name,
            'Username': client.username,
            'Gender': user_data.get('gender'),
            'Weight (lbs)': round(user_data.get('weight', 0) / 453.592, 1),
            'Height (ft)': round(user_data.get('height', 0) / 30.48, 1),
            'Location': user_profile.get('location')
        }
    except Exception as e:
        return None

def get_training_stress(df):
    """Calculates the ratio of last 7 days volume vs last 30 days."""
    if df is None or df.empty:
        return None
    
    today = pd.Timestamp.now()
    
    # Acute Load 
    acute_mask = (df['Date'] >= (today - pd.Timedelta(days=7)))
    acute_load = df.loc[acute_mask, 'Distance (mi)'].sum()
    
    # Chronic Load 
    chronic_mask = (df['Date'] >= (today - pd.Timedelta(days=30)))
    chronic_load = df.loc[chronic_mask, 'Distance (mi)'].sum() / 4.28 # Avg weeks in a month
    
    if chronic_load == 0:
        return 0.0
        
    ratio = acute_load / chronic_load
    return round(ratio, 2)

def plot_activity_map(df):
    # Filter for rows that have GPS data
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        st.warning("GPS data not found in the current dataset.")
        return None

    coords = df[["Activity Name", "Latitude", "Longitude", "Distance (mi)", "Date"]].dropna()
    
    if coords.empty:
        st.info("No coordinate data available to map.")
        return None

    # Center map on the most recent activity or mean location
    m = folium.Map(
        location=[coords["Latitude"].mean(), coords["Longitude"].mean()],
        zoom_start=12,
        tiles="CartoDB positron" # Clean, professional look
    )

    # Add markers for activity start points
    marker_group = folium.FeatureGroup(name="Individual Runs").add_to(m)
    for _, row in coords.iterrows():

        popup_info = f"""
            <strong>{row['Activity Name']}</strong><br>
            Distance: {row['Distance (mi)']} mi<br>
            Date: {row['Date'].strftime('%Y-%m-%d')}
        """
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=4,
            color="#3B82F6",
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_info, max_width=250)
        ).add_to(marker_group)
    
    heat_data = coords[["Latitude", "Longitude"]].values.tolist()
    HeatMap(heat_data, name="Heatmap (Density)", min_opacity=0.4, radius=16).add_to(m)

    folium.LayerControl().add_to(m)
    
    st_folium(m, width='stretch', height=600, returned_objects=[])
