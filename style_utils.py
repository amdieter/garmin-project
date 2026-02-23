import streamlit as st

def apply_custom_style():
    st.markdown("""
    <style>
    /* Global App Background */
    .stApp {
        background-color: #0e1117;
    }

    /* Universal Text & Header Visibility */
    html, body, [class*="css"], .stMarkdown, [data-testid="stMarkdownContainer"], 
    h1, h2, h3, h4, h5, h6, label, p, summary, [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebarNav"] span, div[data-baseweb="select"] span {
        color: #ffffff !important;
        letter-spacing: -0.2px;
    }

    /* Specific Header Weight & Metric Labels */
    h1, h2, h3 { font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #8899a6 !important; }
    [data-testid="stMetricValue"] { font-size: 2rem; color: #0078ff; }

    /* Glassmorphism Containers: Chat Bubbles, Expanders, and Sidebar */
    [data-testid="stChatMessage"], 
    .st-expanderHeader, 
    [data-testid="stSidebar"] > div:first-child {
        background-color: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar Specific Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Coach Message Accent */
    [data-testid="stChatMessageContent"]:has(div[aria-label="assistant"]) {
        border-left: 3px solid #0078ff !important;
    }

    /* Specific styling for Buttons (Login, etc.) */
    .stButton>button {
        background-color: #1a1c23 !important; /* Dark charcoal */
        color: #ffffff !important;           /* Pure white text */
        border: 1px solid #0078ff !important; /* Garmin Blue border */
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }

    /* Hover state to make it feel interactive */
    .stButton>button:hover {
        background-color: #0078ff !important; /* Blue background on hover */
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(0, 120, 255, 0.4);
    }

    /* Active/Click state */
    .stButton>button:active {
        transform: scale(0.98);
    }

    /* Chat Input & Bottom UI */
    [data-testid="stChatInput"] {
        background-color: #1a1c23 !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px;
    }
    
    [data-testid="stChatInputTextArea"] {
        color: #ffffff !important;
    }

    [data-testid="stChatInputButton"] {
        color: #0078ff !important;
        background-color: transparent !important;
    }

    /* Hidden Header Elements */
    [data-testid="stHeader"] {
        visibility: hidden;
        height: 0%;
    }
    /* Dark Chat Input Styling */
    [data-testid="stChatInput"] {
        background-color: #1a1c23 !important; /* Rich black/dark charcoal */
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px;
    }

    /* Target the text area inside to ensure no white background bleeds through */
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #ffffff !important;
    }

    /* Add a slight glow when the user clicks into the box */
    [data-testid="stChatInput"]:focus-within {
        border: 1px solid #0078ff !important;
        box-shadow: 0 0 10px rgba(0, 120, 255, 0.2);
    }

    [class="st-emotion-cache-128upt6 e1td4qo63"] {
        background-color: transparent;
    }
    
    </style>
    """, unsafe_allow_html=True)
