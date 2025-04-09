# CPU-only base image (using correct PyTorch tag)
# FROM python:3.8-slim
FROM python:3.10.4-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install heavy, slow-changing requirements
COPY requirements_heavy.txt .
RUN pip install --no-cache-dir -r requirements_heavy.txt

# Install Detectron2 separately to avoid re-downloading on each build
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Clone and install LightRAG
RUN git clone https://github.com/HKUDS/LightRAG.git /tmp/LightRAG \
    && mv /tmp/LightRAG /app/LightRAG \
    && rm -rf /tmp/LightRAG/.git  
WORKDIR /app/LightRAG
RUN pip install -e .
WORKDIR /app


# Copy and install base (fast-changing) requirements
COPY requirements_base.txt .
RUN pip install --no-cache-dir -r requirements_base.txt

RUN pip install --no-cache-dir pillow==9.5.0

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application with hot reloading enabled
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 