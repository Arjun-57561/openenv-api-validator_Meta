FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cache bust: v4
ENV API_BASE_URL="https://api-inference.huggingface.co/v1" \
    MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct" \
    HF_TOKEN="dummy-token" \
    PORT=7860

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
