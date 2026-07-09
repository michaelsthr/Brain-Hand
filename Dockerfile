FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .


RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .


RUN mkdir -p /app/data /app/trained_models

ENV PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=10000

EXPOSE 10000

CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 0 app:app
