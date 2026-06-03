FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --default-timeout=300 --retries=10 --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY . .

ENTRYPOINT ["python"]
CMD ["scripts/run_full_pipeline.py", "--help"]
