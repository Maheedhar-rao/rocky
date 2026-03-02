FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps: Python 3.11, poppler for pdf2image, tesseract for OCR fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl build-essential \
    poppler-utils tesseract-ocr tesseract-ocr-eng \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

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

# Data directory
RUN mkdir -p /app/data

COPY start.sh .
RUN chmod +x start.sh

# HuggingFace Spaces uses port 7860
ENV PORT=7860 PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["/app/start.sh"]
