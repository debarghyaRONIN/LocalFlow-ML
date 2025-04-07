FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional system dependencies for SHAP
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY services/model-api/app/ /app/app/

# Copy data directory from project root
COPY data/ /app/data/

# Set up directories that might not exist in the copied data/
RUN mkdir -p /app/data/feature_store/transformers \
    /app/data/feature_store/schemas \
    /app/data/feature_store/features \
    /app/data/validation/expectations \
    /app/data/validation/validations

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=minio123

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 