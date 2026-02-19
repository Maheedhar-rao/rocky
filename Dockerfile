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

# Download model weights from GitHub Release (445MB compressed)
ARG MODEL_URL=https://github.com/Maheedhar-rao/rocky/releases/download/v1.0.0/rocky-models.tar.gz
RUN mkdir -p /app/models/statement_parser /app/models/credit_debit \
    && curl -L -o /tmp/models.tar.gz "$MODEL_URL" \
    && tar xzf /tmp/models.tar.gz -C /app \
    && rm /tmp/models.tar.gz

# Data directory (persisted via volume mount)
RUN mkdir -p /app/data

COPY start.sh .
RUN chmod +x start.sh

ENV PORT=8080 PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["/app/start.sh"]
