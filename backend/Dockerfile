# Phase 3: Model Serving Dockerfile for Banking Fraud Detection API
# Base image with Python 3.13 runtime
FROM python:3.13-slim

# Set environment variables
# - Prevent Python from writing .pyc files to disc
# - Force stdin, stdout and stderr to be unbuffered (real-time logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set application working directory
WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy configuration, model artifacts, and application source code
COPY configs/ ./configs/
COPY models/ ./models/
COPY src/ ./src/

# Create a non-root user for security best practices
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Run FastAPI using Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

