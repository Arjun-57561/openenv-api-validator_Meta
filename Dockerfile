FROM python:3.11-slim

WORKDIR /app

ARG API_BASE_URL=""
ARG MODEL_NAME=""
ARG HF_TOKEN=""
ENV API_BASE_URL=$API_BASE_URL
ENV MODEL_NAME=$MODEL_NAME
ENV HF_TOKEN=$HF_TOKEN

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY openenv.yaml .
COPY inference.py .

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
