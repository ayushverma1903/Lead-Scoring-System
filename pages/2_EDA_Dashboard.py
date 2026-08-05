import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="EDA Dashboard", page_icon="📈", layout="wide")

st.title("📈 Exploratory Data Analysis")

data_path = "data/raw/Lead Scoring.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    st.markdown("### Conversion Rate by Lead Source")
    # Group by Lead Source
    lead_source_conv = df.groupby('Lead Source')['Converted'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(lead_source_conv)
    
    st.markdown("### Total Visits vs Conversion")
    # Filter out outliers for better visualization
    visits_df = df[df['TotalVisits'] < 50].dropna(subset=['TotalVisits', 'Converted'])
    # Convert 'Converted' to string for categorical coloring in some libraries, but here we can just use a scatter or bar
    # Let's group by converted status
    avg_visits = visits_df.groupby('Converted')['TotalVisits'].mean()
    st.bar_chart(avg_visits)
    
    st.markdown("### Missing Values Heatmap (Top 15 Columns)")
    missing = df.isnull().sum().sort_values(ascending=False).head(15)
    st.bar_chart(missing)
    
else:
    st.info("Raw dataset not found at 'data/raw/Lead Scoring.csv'. Please upload the data or run the notebooks.")
