FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Defaults: safe-by-default + localhost only (override on NAS)
ENV C4_WRITE_GUARDRAILS=true \
    C4_WRITES_ENABLED=false \
    C4_BIND_HOST=127.0.0.1 \
    C4_PORT=3333

EXPOSE 3333

CMD ["python", "app.py"]
