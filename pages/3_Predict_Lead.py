import streamlit as st
import pandas as pd
import requests
import json
import os

st.set_page_config(page_title="Predict Lead", page_icon="🔮", layout="wide")

st.title("🔮 Predict Lead Conversion")

st.markdown("Use this page to score leads. You can either score a single lead manually or upload a CSV file for batch prediction.")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

API_URL = os.getenv("API_URL", "http://localhost:8000")

with tab1:
    st.subheader("Single Lead Prediction")
    
    with st.form("single_lead_form"):
        col1, col2 = st.columns(2)
        with col1:
            total_visits = st.number_input("Total Visits", min_value=0.0, value=0.0)
            time_spent = st.number_input("Total Time Spent on Website", min_value=0.0, value=0.0)
            page_views = st.number_input("Page Views Per Visit", min_value=0.0, value=0.0)
        with col2:
            lead_origin = st.selectbox("Lead Origin", ["Landing Page Submission", "API", "Lead Add Form", "Lead Import"])
            lead_source = st.selectbox("Lead Source", ["Google", "Direct Traffic", "Organic Search", "Reference", "Other"])
            do_not_email = st.selectbox("Do Not Email", ["No", "Yes"])
        
        submit = st.form_submit_button("Predict")
        
        if submit:
            data = {
                "TotalVisits": total_visits,
                "Total Time Spent on Website": time_spent,
                "Page Views Per Visit": page_views,
                "Lead Origin": lead_origin,
                "Lead Source": lead_source,
                "Do Not Email": do_not_email
            }
            try:
                response = requests.post(f"{API_URL}/predict", json={"data": data})
                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction Successful!")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Conversion Probability", f"{result['Conversion_Probability']}%")
                    c2.metric("Prediction", result['Prediction'])
                    c3.metric("Priority", result['Priority'])
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")

with tab2:
    st.subheader("Batch Lead Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file with leads data", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        if st.button("Score Leads"):
            with st.spinner("Scoring leads..."):
                try:
                    import math
                    
                    batch_size = 500
                    num_batches = math.ceil(len(df) / batch_size)
                    results_list = []
                    
                    progress_text = "Scoring leads..."
                    progress_bar = st.progress(0, text=progress_text)
                    
                    success = True
                    for i in range(num_batches):
                        chunk = df.iloc[i * batch_size : (i + 1) * batch_size]
                        # Use pandas to_json which natively converts NaN → null
                        leads_list = json.loads(chunk.to_json(orient="records"))
                        
                        try:
                            response = requests.post(f"{API_URL}/batch_predict", json={"leads": leads_list}, timeout=30)
                            if response.status_code == 200:
                                results_list.extend(response.json())
                            else:
                                st.error(f"Error in batch {i + 1}: {response.text}")
                                success = False
                                break
                        except requests.exceptions.RequestException as e:
                            st.error(f"Failed to connect to API on batch {i + 1}: {str(e)}")
                            success = False
                            break
                            
                        # Update progress bar
                        progress_bar.progress((i + 1) / num_batches, text=f"Scored {min((i+1)*batch_size, len(df))} of {len(df)} leads...")
                        
                    if success and len(results_list) > 0:
                        results_df = pd.DataFrame(results_list)
                        st.success("Batch Prediction Successful!")
                        st.dataframe(results_df.head(50)) # preview first 50
                        
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download All Results as CSV",
                            data=csv,
                            file_name='scored_leads.csv',
                            mime='text/csv',
                        )
