# Garmin Guru
A Streamlit-powered dashboard that connects to your Garmin Connect account to provide deep-dive analytics, interactive maps, and an AI Running Coach that knows your personal training history.

---

## Features
Live Garmin Integration: Syncs recent activities, heart rate zones, and VO2 Max data.

AI Running Coach: A LangChain agent using Gemini to analyze your performance and answer questions about your training load.

Interactive Heatmaps: Visualize your running routes with click-to-view activity details.

Training Stress Analysis: Automated calculation of recovery vs. overreaching status based on recent volume.

RAG (Retrieval-Augmented Generation): The coach is powered by professional coaching PDFs to provide advice grounded in sports science.

---
## Setup & Installation
1. Clone the Repo
```bash
git clone [https://github.com/yourusername/garmin-analytics-ai.git](https://github.com/amdieter/garmin-guru.git)
cd garmin-guru
```
2. Activate a virtual environment
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Environment Variables
Create a cred.env file in the root directory and add your API keys:

```
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_search_api_key
# Garmin credentials can be entered via the UI or added here
GARMIN_EMAIL=your_email@example.com
GARMIN_PASSWORD=your_password
```
---

## Usage
1. Login: Enter your Garmin credentials on the landing page.

2. Dashboard: Toggle between Week, Month, and Year views to see mileage and elevation trends.

3. Critique: Select a specific run from the dropdown and click "Coach: Critique this Run" for an AI breakdown of your HR zones and performance.

4. Chat: Use the chat bar at the bottom to ask things like, "Am I running my easy runs too fast?" or "How does my elevation gain this week compare to last week?"
