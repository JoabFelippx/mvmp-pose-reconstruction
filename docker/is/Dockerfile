FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libfreetype6 \
        libpng16-16 \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Código-fonte
COPY src/ ./

# Calibrações embutidas na imagem
COPY calibrations/ ./calibrations/

# Apenas config.json é montado em runtime (fácil de trocar sem rebuild)
VOLUME ["/app/etc"]

CMD ["python", "skeleton_tracker_main.py"]