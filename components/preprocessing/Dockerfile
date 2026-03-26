FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r requirements.txt

COPY component.py .

ENTRYPOINT ["python", "component.py"]
