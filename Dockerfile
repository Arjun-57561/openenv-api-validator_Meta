FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV API_BASE_URL="https://api.groq.com/openai/v1" \
    MODEL_NAME="llama-3.1-8b-instant" \
    GROQ_API_KEY="" \
    HF_TOKEN="" \
    PORT=7860

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
