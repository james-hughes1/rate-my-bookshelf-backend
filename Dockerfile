# syntax=docker/dockerfile:1

####################################
# Stage 1: Builder
####################################
FROM python:3.12.2-slim AS builder

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only the pyproject.toml and poetry.lock first (for caching)
COPY pyproject.toml poetry.lock* ./

# Install dependencies in isolated environment, no dev packages
RUN poetry install --no-root --no-interaction --no-ansi

# Copy the rest of the project
COPY . .

####################################
# Stage 2: Runtime
####################################
FROM python:3.12.2-slim

# Runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry in runtime
RUN pip install --no-cache-dir poetry

# Copy installed Python packages and app code from builder
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Expose port
EXPOSE 8080

# Run FastAPI via Poetry
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
