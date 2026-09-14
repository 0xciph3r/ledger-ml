FROM python:3.12.6-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY artifacts/ledgerml_artifacts /app/ledgerml_artifacts
COPY training/ledgerml_training /app/ledgerml_training
COPY training/train.py /app/train.py

ENTRYPOINT ["python", "/app/train.py"]
