# syntax=docker/dockerfile:1
FROM python:3.12.2-slim

# -----------------------
# Environment variables
# -----------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.2.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    APP_HOME=/app

# -----------------------
# System dependencies
# -----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Upgrade pip and install Poetry
# -----------------------
RUN python -m pip install --upgrade pip
RUN pip install "poetry==$POETRY_VERSION"

# -----------------------
# Set workdir
# -----------------------
WORKDIR $APP_HOME

# -----------------------
# Copy only pyproject.toml & poetry.lock to leverage Docker cache
# -----------------------
COPY pyproject.toml poetry.lock* ./

# -----------------------
# Install dependencies (no root)
# -----------------------
RUN poetry install --no-root --no-interaction --no-ansi

# -----------------------
# Copy project source
# -----------------------
COPY . .

# -----------------------
# Preload RapidOCR model (cached in Docker layer)
# -----------------------
RUN python -c "from rapidocr_onnxruntime import RapidOCR; engine = RapidOCR()"

# -----------------------
# Expose FastAPI port
# -----------------------
EXPOSE 8080

# -----------------------
# Start server
# -----------------------
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
