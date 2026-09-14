FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ledgerml_monitoring ./ledgerml_monitoring

ENTRYPOINT ["python", "-m", "ledgerml_monitoring.detect_drift"]
