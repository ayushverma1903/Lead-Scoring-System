FROM python:3.9-slim

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Expose port and run Streamlit
EXPOSE 8501
CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
