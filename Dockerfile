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

# Copy all Python source files
COPY *.py ./

# Copy model weights (baked into image — no Supabase download needed)
COPY models/statement_parser/ /app/models/statement_parser/
COPY models/credit_debit/ /app/models/credit_debit/

# Data directory (persisted via volume mount)
RUN mkdir -p /app/data

COPY start.sh .
RUN chmod +x start.sh

ENV PORT=8080 PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["/app/start.sh"]
