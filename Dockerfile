# syntax=docker/dockerfile:1

FROM python:3.12.2-slim

# -----------------------
# Environment variables
# -----------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.2.1 \
    POETRY_VIRTUALENVS_CREATE=false

# -----------------------
# Install system dependencies
# -----------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Upgrade pip and install Poetry
# -----------------------
RUN python -m pip install --upgrade pip
RUN pip install "poetry==$POETRY_VERSION"

# -----------------------
# Set workdir
# -----------------------
WORKDIR /app

# -----------------------
# Copy project files and install dependencies
# -----------------------
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --no-interaction --no-ansi
COPY . .

# -----------------------
# Expose port
# -----------------------
EXPOSE 8080

# -----------------------
# Pre-download EasyOCR weights (during build)
# -----------------------
RUN python -c "from app.services.ocr import init_easyocr; init_easyocr()"

# -----------------------
# Run FastAPI via Poetry
# -----------------------
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
