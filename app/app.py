"""
app.py — Lead Scoring UI (Streamlit)
--------------------------------------
A simple, non-technical-friendly interface for scoring leads.

Run with:
    streamlit run app/app.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from src.predict import LeadScorer

st.set_page_config(page_title="Lead Scoring System", page_icon="📊", layout="centered")

MODEL_PATH = "models/lead_scoring_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"


@st.cache_resource
def load_scorer():
    return LeadScorer(MODEL_PATH, SCALER_PATH, FEATURES_PATH)


st.title("📊 Lead Scoring System")
st.write("Enter a lead's details below to predict their conversion probability.")

scorer = load_scorer()

with st.form("lead_form"):
    col1, col2 = st.columns(2)

    with col1:
        total_visits = st.number_input("Total Visits", min_value=0, value=5)
        time_spent = st.number_input("Total Time Spent on Website (seconds)", min_value=0, value=800)
        page_views = st.number_input("Page Views Per Visit", min_value=0.0, value=2.5)
        lead_origin = st.selectbox("Lead Origin", [
            "Landing Page Submission", "API", "Lead Add Form", "Lead Import", "Quick Add Form"
        ])
        lead_source = st.selectbox("Lead Source", [
            "Google", "Direct Traffic", "Olark Chat", "Organic Search",
            "Reference", "Welingak Website", "Facebook", "Other"
        ])
        do_not_email = st.selectbox("Do Not Email", ["No", "Yes"])

    with col2:
        occupation = st.selectbox("Current Occupation", [
            "Unemployed", "Working Professional", "Student", "Businessman", "Unknown"
        ])
        specialization = st.text_input("Specialization", value="Unknown")
        city = st.text_input("City", value="Unknown")
        country = st.text_input("Country", value="India")
        last_activity = st.selectbox("Last Activity", [
            "Email Opened", "SMS Sent", "Page Visited on Website",
            "Olark Chat Conversation", "Converted to Lead", "Other"
        ])
        do_not_call = st.selectbox("Do Not Call", ["No", "Yes"])

    submitted = st.form_submit_button("Score This Lead")

if submitted:
    raw_lead = pd.DataFrame([{
        "TotalVisits": total_visits,
        "Total Time Spent on Website": time_spent,
        "Page Views Per Visit": page_views,
        "Lead Origin": lead_origin,
        "Lead Source": lead_source,
        "Do Not Email": do_not_email,
        "Do Not Call": do_not_call,
        "What is your current occupation": occupation,
        "Specialization": specialization,
        "City": city,
        "Country": country,
        "Last Activity": last_activity,
    }])

    report = scorer.score_raw_leads(raw_lead)
    result = report.iloc[0]

    st.divider()
    prob = result["Conversion_Probability"]
    priority = result["Priority"]

    if priority == "Hot":
        st.success(f"🔥 **Hot Lead** — {prob}% conversion probability")
    elif priority == "Warm":
        st.warning(f"🌤️ **Warm Lead** — {prob}% conversion probability")
    else:
        st.error(f"❄️ **Cold Lead** — {prob}% conversion probability")

    st.metric("Prediction", result["Prediction"])

st.divider()
st.subheader("📁 Score a Batch of Leads (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV with raw lead data", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    batch_report = scorer.score_raw_leads(batch_df)
    st.write(f"Scored {len(batch_report)} leads:")
    st.dataframe(batch_report)

    csv = batch_report.to_csv(index=True).encode("utf-8")
    st.download_button("Download Scored Report (CSV)", csv, "lead_scoring_report.csv", "text/csv")
