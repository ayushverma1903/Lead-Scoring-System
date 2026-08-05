import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About the Lead Scoring System")

st.markdown("""
### Overview
The Lead Scoring System is an end-to-end Machine Learning pipeline that predicts the likelihood of a lead converting into a customer. By assigning a score between 0 and 100 to each lead, the sales team can prioritize 'Hot' leads and improve their conversion rate.

### Architecture
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Machine Learning:** Scikit-Learn (Logistic Regression)
- **Containerization:** Docker

### Phases Completed
- ✅ Data Cleaning & EDA
- ✅ Feature Engineering & Selection
- ✅ Model Training & Evaluation
- ✅ API Development
- ✅ Dashboard Creation
- ✅ MLOps (Retraining & Testing)

Developed by the Data Science Team.
""")
