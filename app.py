import streamlit as st

st.set_page_config(
    page_title="Lead Scoring System",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Lead Scoring System")
st.markdown("""
Welcome to the **Lead Scoring System**! 

This application predicts the conversion probability of leads for the LMS platform, helping the sales team prioritize their efforts on hot leads.

### 👈 Navigation
Use the sidebar to navigate through the application:
- **Dataset Overview:** View raw and processed data.
- **EDA Dashboard:** Interactive visualizations of the data.
- **Predict Lead:** Make single or batch predictions using the ML model.
- **Model Performance:** View evaluation metrics of the currently trained model.
- **SHAP Explainability:** Understand why the model makes certain predictions.
- **Business Insights:** High-level insights and KPI cards.
- **About:** Information about the project.

### Quick KPI
""")

col1, col2, col3 = st.columns(3)
col1.metric(label="Total Leads Processed", value="9,240")
col2.metric(label="Overall Conversion Rate", value="38.5%")
col3.metric(label="Model Accuracy", value="82.4%")

st.info("Navigate to the **Predict Lead** page to start scoring new leads!")
