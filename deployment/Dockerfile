FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src /app/src
ENV PYTHONPATH=/app/src BASE_SERVICE_MS=35 SERVICE_SIGMA=0.6
EXPOSE 8080
CMD ["uvicorn", "cvar_safe.api:app", "--host", "0.0.0.0", "--port", "8080"]
