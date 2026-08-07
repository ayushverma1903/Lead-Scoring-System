FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports for both services (though usually overridden in compose)
EXPOSE 8000
EXPOSE 8501

# The default command can be anything, docker-compose will override it
CMD sh -c "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"
