FROM python:3.11-slim

ARG BUILD_DATE
ARG GIT_SHA
LABEL build_date="${BUILD_DATE}" git_sha="${GIT_SHA}"

WORKDIR /app

# Install system deps needed by psycopg2-binary and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/app ./app
COPY config.yaml ./config.yaml

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
