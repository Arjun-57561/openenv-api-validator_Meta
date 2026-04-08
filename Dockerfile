FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV API_BASE_URL="https://api-inference.huggingface.co/v1" \
    MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" \
    HF_TOKEN="" \
    PORT=7860

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]