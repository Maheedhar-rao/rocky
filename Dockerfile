FROM python:3.11-slim

WORKDIR /app

# System deps: poppler for pdf2image, tesseract for OCR fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl poppler-utils tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (~700MB instead of ~2.2GB with CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY predict.py app.py download_model.py upload_model.py \
     credit_debit_model.py spending_demo.py \
     align_labels.py retrain.py train.py ./

# Model + data directories (weights downloaded at startup, data persisted via volume)
RUN mkdir -p /app/models/statement_parser /app/models/credit_debit /app/data

COPY start.sh .
RUN chmod +x start.sh

ENV PORT=8080 PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["/app/start.sh"]
