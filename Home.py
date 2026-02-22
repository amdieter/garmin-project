import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import StructuredTool
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from garminconnect import Garmin
import garminconnect
from datetime import datetime, date, timedelta
from langchain_core.messages import SystemMessage
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import matplotlib.pyplot as plt
from data_utils import get_cached_workout_data, summarize_n_days, get_training_stress, get_workout_dataframe_n_days
GARMIN_CACHE = None
load_dotenv("cred.env")

# persists specifically for Streamlit sessions
msgs = StreamlitChatMessageHistory(key="chat_messages")
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        chat_memory=msgs, 
        return_messages=True
    )

# Functions
# def summarize_n_days(n_days=1):
#     df = get_workout_dataframe_n_days(n_days)
#     if isinstance(df, tuple) or df is None or df.empty:
#         return {k: 0 for k in ["Longest Run (mi)", "Total Distance Run (mi)", "Total Elevation Gained (ft)"]} | {"Current VO2 Max": "N/A"}

#     summary = {}
#     summary["Longest Run (mi)"] = df["Distance (mi)"].max()
#     summary["Longest Run (min)"] = df["Duration (min)"].max()
#     summary["Average Run Length (mi)"] = df["Distance (mi)"].mean()
#     summary["Total Distance Run (mi)"] = df["Distance (mi)"].sum()
#     summary["Total Duration Run (min)"] = df["Duration (min)"].sum()
#     summary["Total Elevation Gained (ft)"] = df["Elev Gain (ft)"].sum()
#     summary["Total Elevation Lost (ft)"] = df["Elev Loss (ft)"].sum()
#     summary["Current VO2 Max"] = df["VO2 Max"][0]
#     if n_days > 1:
#         clean_df = df.dropna()
#         if len(clean_df) > 1:
#             summary["VO2 Max Progress"] = clean_df["VO2 Max"].iloc[0] - clean_df["VO2 Max"].iloc[-1]
#     if n_days >= 31:
#         summary["Most Active Month"] = most_active_month(df)
#     return summary

def most_active_month(df):
    df['Month'] = df['Date'].dt.month
    monthly_miles = df.groupby("Month")["Distance (mi)"].sum()
    month = monthly_miles.idxmax()
    miles = monthly_miles.max()
    return month, miles

# def get_workout_dataframe_n_days(n_days):
#     try:
#         client = garminconnect.Garmin(os.getenv("GARMIN_EMAIL"), os.getenv("GARMIN_PASSWORD"))
#         client.login()
#         today = date.today()
#         days_to_subtract = timedelta(days=n_days)
#         start = str(today - days_to_subtract)
#         activities = client.get_activities_by_date(startdate = start, enddate = str(today))

#         # if there's no activities in date range, return first (most recent) activity
#         if not activities:
#             activities = client.get_activities(0,1)

#         data_list = []
#         for act in activities:
#             if act["activityType"]["typeKey"] != "running":
#                 # print("non running")
#                 continue
#             dist_mi = act.get('distance', 0) / 1609.34
#             dur_min = act.get('duration', 0) / 60

#             # Pace Calculations
#             pace_decimal = dur_min / dist_mi if dist_mi > 0 else 0

#             # HR Zones (Converted from Seconds to Minutes)
#             z1 = act.get('hrTimeInZone_1', 0) / 60
#             z2 = act.get('hrTimeInZone_2', 0) / 60
#             z3 = act.get('hrTimeInZone_3', 0) / 60
#             z4 = act.get('hrTimeInZone_4', 0) / 60
#             z5 = act.get('hrTimeInZone_5', 0) / 60

#             record = {
#                 "Activity Name": act.get('activityName'),
#                 "Date": pd.to_datetime(act.get('startTimeLocal')),
#                 "Distance (mi)": round(dist_mi, 2),
#                 "Duration (min)": round(dur_min, 2),
#                 "Pace_Decimal": round(pace_decimal, 2),
#                 "Avg HR": act.get('averageHR'),
#                 "Max HR": act.get('maxHR'),
#                 "Elev Gain (ft)": round(act.get('elevationGain', 0) * 3.28084, 1),
#                 "Elev Loss (ft)": round(act.get('elevationLoss', 0) * 3.28084, 1),
#                 "VO2 Max": act.get('vO2MaxValue'),
#                 # HR Zone columns
#                 "Z1_Min": round(z1, 2),
#                 "Z2_Min": round(z2, 2),
#                 "Z3_Min": round(z3, 2),
#                 "Z4_Min": round(z4, 2),
#                 "Z5_Min": round(z5, 2)
#             }
#             data_list.append(record)

#         df = pd.DataFrame(data_list)

#         return df

#     except Exception as e:
#         return None, f"Error: {e}"

# Tool Setup

def coach_retrieval(q, retriever):
    docs = retriever.invoke(q)
    return "\n\n".join([d.page_content for d in docs])

def get_ai_hot_take(stats):
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        llm_joke = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=api_key)
        # Use the already initialized coach_agent or llm
        prompt = f"Write a 1-sentence funny, running coach comment about these stats: {stats}. Keep it fun and use comparisons to real-world things, focus on ridiculous comparisons with total distance and total elevation gain."
        response = llm_joke.invoke(prompt) # Or use your llm.invoke()
        return response.content
    except Exception as e:
        return f"Keep up the pace, human! {e}"

def get_agent():
    load_dotenv("cred.env")
    api_key = os.getenv("GOOGLE_API_KEY")
    # Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Embeddings
    embeddings = GPT4AllEmbeddings()

    # Vector Stores
    if os.path.exists("docs/coaching-store"):
        coaching_store = FAISS.load_local("docs/coaching-store", embeddings, allow_dangerous_deserialization=True)
        coaching_retriever = coaching_store.as_retriever()
    else:
        all_docs = []
        for file in os.listdir("./docs/"):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(f"./docs/{file}")
                all_docs.extend(loader.load())
        
        coaching_chunks = splitter.split_documents(all_docs)
        coaching_store = FAISS.from_documents(coaching_chunks, embeddings)
        coaching_retriever = coaching_store.as_retriever()
        coaching_store.save_local("docs/coaching-store")

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key = api_key, temperature=0)

    def workout_data_query(query: str):
        try:
            current_df = st.session_state.get("df_data")
            if current_df is None:
                return "Dataframe not found. Please refresh the data."
            
            df_agent = create_pandas_dataframe_agent(
                llm, 
                current_df, 
                verbose=False, 
                allow_dangerous_code=True,
                handle_parsing_errors=True 
            )
            response = df_agent.invoke({"input": query})
            return response["output"]
        except Exception as e:
            return f"Error analyzing data: {str(e)}"

    search_tool = TavilySearchResults(tavily_api_key = os.getenv("TAVILY_API_KEY"))

    tools = [
        Tool(
            name="coaching_expert",
            func=lambda q: coach_retrieval(q, coaching_retriever),
            description="Search this for analytical running principles and workout definitions."
        ), 
        Tool(
        name="Workout_Data_Analyzer",
        func=workout_data_query,
        description="Query this to get stats on the user's recent running activities, pace, and heart rate."
        )
    ]

    # Prompt Definition
    Custom_Coach_Prompt = """
        You are an analytical running coach, being presented with a Garmin User's recent activity data. Your job is to analyze their statistics, and respond to their queries using the tools available. Provide recommendations about training, warnings about heightened intensity or load, or injury risk.
        For context: 5 heart rate zones are calculated as percentages of your maximum heart rate (roughly): Zone 1 (50–60%): Very light, warm-up; Zone 2 (60–70%): Light, aerobic base; Zone 3 (70–80%): Moderate, tempo; Zone 4 (80–90%): Hard, threshold; Zone 5 (90–100%): Maximum effort.
    """
    instructions = SystemMessage(content=Custom_Coach_Prompt)
    
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an analytical running coach. Analyze Garmin data and provide advice."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
        
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=st.session_state.memory, 
        verbose=True
    )

if __name__ == "__main__":

    st.set_page_config(page_icon = "Home", page_title="Garmin Guru", layout="wide")

    # Initialize login state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        st.title("🔐 Login to Garmin Connect")
        
        with st.form("login_form"):
            email = st.text_input("Garmin Email")
            password = st.text_input("Garmin Password", type="password")
            submit = st.form_submit_button("Login")
    
            if submit:
                if email and password:
                    # Store in session state for other pages to use
                    st.session_state.garmin_email = email
                    st.session_state.garmin_password = password
                    
                    st.session_state.logged_in = True
                    st.success("Credentials saved locally for this session!")
                    st.rerun()
                else:
                    st.error("Please enter both email and password.")
        st.stop() # Stops the rest of the app from loading until logged in

    if st.session_state.logged_in:
        
        email = st.session_state.garmin_email
        pwd = st.session_state.garmin_password
        current_range = st.session_state.get("range_days", 7) # Pull last week
        
        df = get_cached_workout_data(current_range, email, pwd)
        stats = summarize_n_days(df)

        if "df_all_time" not in st.session_state:
            with st.spinner("Fetching all-time history for PRs..."):
                # Fetch 3650 days (10 years)
                st.session_state.df_all_time = get_cached_workout_data(3650, email, pwd)
            
        # Store df in session state
        st.session_state.df_data = df
        st.session_state.coach_agent = get_agent()
        # Get all-time for pr calc
        all_time_stats = summarize_n_days(st.session_state.df_all_time)
    
    st.title("Garmin Guru")

    with st.sidebar:
        st.header("📊 Training Summary")
        c1, c2, c3 = st.columns(3)
        
        # Use st.session_state to track the active selection
        if c1.button("Week"):
            msgs.clear()
            st.session_state.memory.clear()
            st.session_state.range_days = 7
            st.rerun()
        if c2.button("Month"):
            msgs.clear()
            st.session_state.memory.clear()
            st.session_state.range_days = 30
            st.rerun()
        if c3.button("Year"):
            msgs.clear()
            st.session_state.memory.clear()
            st.session_state.range_days = 365
            st.rerun()
            
    
        # Logic: If the button was just clicked, the script reruns and picks up this value
        current_range = st.session_state.get("range_days", 7)
    
    # Aliases
    coach_agent = st.session_state.coach_agent
    df = st.session_state.get("df_data")
    
    # Activity Selector Dropdown
    if df is not None and not df.empty:
        st.divider()
        
        # Create a label for the dropdown (Date + Name)
        df['display_name'] = df['Date'].dt.strftime('%Y-%m-%d') + " - " + df['Activity Name']
        
        # Selection Box
        selected_run_name = st.selectbox("Select a past activity to analyze:", df['display_name'])
        
        # Get the specific row for that activity
        run_data = df[df['display_name'] == selected_run_name].iloc[0]
    
        # Key Metrics for the specific run
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Distance", f"{run_data['Distance (mi)']} mi")
        col2.metric("Pace", f"{run_data['Pace_Decimal']:.2f} min/mi")
        col3.metric("Avg HR", f"{run_data['Avg HR']} bpm")
        col4.metric("Elev Gain", f"{run_data['Elev Gain (ft)']} ft")
    
    if st.button("Coach: Critique this Run"):
        with st.spinner("Analyzing effort..."):
            # We pass the specific run data to the agent
            critique_query = f"Analyze this specific run: {run_data.to_json()}. Based on the HR zones and pace, was this a good workout? Check the coaching PDFs for context."
            response = coach_agent.invoke({"input": critique_query})
            st.chat_message("assistant").write(response["output"])

    if df is not None and not df.empty:
        # Get the Stress Score
        stress_score = get_training_stress(df)
        
        # Create a high-level status banner
        st.subheader("Training Readiness")
        
        # Define status colors/labels
        if stress_score < 0.8:
            status_label, status_color, status_icon = "Recovery / Detraining", "normal", "🧊"
        elif 0.8 <= stress_score <= 1.3:
            status_label, status_color, status_icon = "Optimal Load", "normal", "✅"
        else:
            status_label, status_color, status_icon = "High Load / Overreaching", "inverse", "⚠️"

        # Display the readiness metric at the top of the dashboard
        r_col1, r_col2 = st.columns([1, 3])
        with r_col1:
            st.metric("Stress Ratio", f"{stress_score}", delta=status_label, delta_color=status_color)
        with r_col2:
            st.markdown(f"**Status: {status_icon}**")
            if status_icon == "⚠️":
                st.warning("Your volume is spiking! Consider an easy day to prevent injury.")
            elif status_icon == "✅":
                st.success("You're in the training 'Sweet Spot.' Keep it up!")
            else:
                st.info("You are currently in a recovery phase or decreasing volume.")

    st.divider()

    # Heart Rate Zone Visualization
    st.subheader("Heart Rate Zone Distribution")
    
    # Prepare data for a Bar Chart
    hr_zones = {
        "Zone 1 (Recovery)": run_data['Z1_Min'],
        "Zone 2 (Aerobic)": run_data['Z2_Min'],
        "Zone 3 (Tempo)": run_data['Z3_Min'],
        "Zone 4 (Threshold)": run_data['Z4_Min'],
        "Zone 5 (Anaerobic)": run_data['Z5_Min']
    }
    
    # Convert to a mini dataframe for plotting
    hr_df = pd.DataFrame(list(hr_zones.items()), columns=['Zone', 'Minutes'])
    
    # Display as a Bar Chart
    st.bar_chart(hr_df.set_index('Zone'))
    
    # Sidebar Statistics
    with st.sidebar:
        if df is not None and not df.empty:
            # Pass the already loaded df to summary to avoid double-fetching
            stats = summarize_n_days(df) 
            
            st.markdown(f"### Last {current_range} Days")
            st.metric("Total Distance", f"{stats.get('Total Distance Run (mi)', 0):.1f} mi")
            st.metric("Elevation Gain", f"{stats.get('Total Elevation Gained (ft)', 0):,.0f} ft")
            st.metric("Current VO2 Max", f"{stats.get('Current VO2 Max', 'N/A')}")
    
            st.divider()
            
            # Use a spinner because calling the AI for a joke can take a second
            with st.spinner("Generating commentary..."):
                # Ensure the function name matches what you defined (get_ai_hot_take)
                joke = get_ai_hot_take(stats)
                st.info(joke)
    
    with st.expander("📊 View Recent Activity Data"):
        if df is not None:
            st.dataframe(df, use_container_width=True)
        else:
            st.error("Could not load Garmin data. Check your credentials.")
    
    # Display existing messages from history
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
    
    # Handle new user input
    if user_query := st.chat_input("Ask Coach about your training..."):
        # Immediately show user message
        st.chat_message("human").write(user_query)
        
        with st.chat_message("assistant"):
            # Create a container for the response
            response_container = st.container()
            
            with st.spinner("Analyzing data and coaching files..."):
                try:
                    # IMPORTANT: Double check the agent exists in state
                    if "coach_agent" in st.session_state:
                        # Clear old charts
                        if os.path.exists("current_chart.png"):
                            os.remove("current_chart.png")
                        
                        # Run the agent
                        response = st.session_state.coach_agent.invoke({"input": user_query})
                        
                        # Write result to the container
                        response_container.write(response["output"])
                        
                        # Display chart if generated
                        if os.path.exists("current_chart.png"):
                            st.image("current_chart.png", use_container_width=True)
                    else:
                        st.error("Coach agent is not initialized. Try refreshing the page.")
                
                except Exception as e:
                    st.error(f"Agent Error: {e}")







