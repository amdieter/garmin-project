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


## Usage
1. Login: Enter your Garmin credentials on the landing page.

2. Dashboard: Toggle between Week, Month, and Year views to see mileage and elevation trends.

3. Critique: Select a specific run from the dropdown and click "Coach: Critique this Run" for an AI breakdown of your HR zones and performance.

4. Chat: Use the chat bar at the bottom to ask things like, "Am I running my easy runs too fast?" or "How does my elevation gain this week compare to last week?"
