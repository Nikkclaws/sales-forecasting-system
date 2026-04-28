FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for prophet, statsmodels, lightgbm-style builds, matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libgomp1 \
        libopenblas-dev \
        liblapack-dev \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY config.yaml ./
COPY dashboard ./dashboard
COPY tests ./tests
COPY data ./data

RUN pip install --upgrade pip && pip install -e .

EXPOSE 8000 8501

CMD ["uvicorn", "sales_forecast.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
