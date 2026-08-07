# 🎯 Lead Scoring System

An end-to-end Machine Learning pipeline that predicts the likelihood of a lead converting into a customer for an online education platform (LMS). By assigning a score to each lead, the sales team can prioritize "Hot" leads, thereby improving conversion rates and operational efficiency.

---

## 📖 Business Problem
The company generates a large number of leads (professionals looking for courses) through various channels. However, the current lead conversion rate is low (around 30%). To improve this, the sales team needs to focus their efforts on leads most likely to convert ("Hot Leads"). This system provides a Lead severy lead to prioritize calling efforts and achieve a target lead conversion rate of ~80%.

## 🏗️ Architecture
- **Data Engineering:** Pandas, Scikit-Learn (Preprocessing Pipelines)
- **Machine Learning:** Logistic Regression, SHAP (Explainability)
- **Backend API:** FastAPI (RESTful API, Batch Processing, Retraining trigger)
- **Frontend Dashboard:** Streamlit (Multi-page Interactive Dashboard)
- **Deployment:** Docker, Docker Compose
- **CI/CD:** GitHub Actions (Automated Retraining via Cron)

## 🔄 Workflow
1. **Data Ingestion:** Raw leads are ingested via the API or CSV upload.
2. **Preprocessing:** Data is cleaned, imputed, encoded, and aligned to the model's expected features.
3. **Scoring:** The trained model predicts the conversion probability and assigns a priority tier.
4. **Insights:** SHAP values explain *why* a lead received a particular score.
5. **Retraining:** A background process regularly checks for new data to retrain the model. If the new model outperforms the old one, it is automatically swapped.

## 📸 Screenshots
*(Add screenshots of the Streamlit dashboard here)*
- **Predict Lead:** Shows the probability score and priority.
- **SHAP Explainability:** Displays feature importance.

## 📡 API Documentation
Once running, you can access the interactive Swagger API docs at `http://localhost:8000/docs`.

### Key Endpoints:
- `GET /health` - Check API and model status.
- `POST /predict` - Score a single lead.
- `POST /batch_predict` - Score multiple leads via JSON array.
- `POST /retrain` - Trigger background model retraining.
- `GET /metrics` - Get the current model's performance metrics.

## 🚀 Installation & Usage (Docker)

The easiest way to run the entire system is using Docker Compose.

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Lead-Scoring-System
   ```

2. **Build and Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access the Applications:**
   - **Streamlit Dashboard:** `http://localhost:8501`
   - **FastAPI Swagger Docs:** `http://localhost:8000/docs`

## 🌍 Deployment

### 1. Streamlit Community Cloud (UI)
- Connect your GitHub repository to [Streamlit Community Cloud](https://streamlit.io/cloud).
- Set the main file path to `app.py`.
- Set the `API_URL` environment variable to your deployed FastAPI backend URL.

### 2. Render (FastAPI Backend)
- Create a new Web Service on [Render](https://render.com/).
- Connect your repository and select Docker as the runtime.
- Override the start command if necessary: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### 3. Alternative Providers (Railway / AWS / Azure)
- Use the provided `Dockerfile` to deploy both services using a managed container platform like AWS ECS, Azure App Service, or Railway.

---

*Developed by the Data Science Team.*
