FROM python:3.12-slim

WORKDIR /app
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY artifacts/ledgerml_artifacts ./ledgerml_artifacts
COPY serving/ledgerml_serving ./ledgerml_serving

EXPOSE 8080
ENTRYPOINT ["python", "-m", "ledgerml_serving.server"]
