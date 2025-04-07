FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download CutLER model and configs
RUN mkdir -p model_zoo/configs/COCO-Semisupervised model_zoo/checkpoints
RUN wget -O model_zoo/configs/COCO-Semisupervised/cascade_mask_rcnn_R_50_FPN_100perc.yaml https://raw.githubusercontent.com/facebookresearch/CutLER/main/model_zoo/configs/COCO-Semisupervised/cascade_mask_rcnn_R_50_FPN_100perc.yaml
RUN wget -O model_zoo/checkpoints/cutler_fully_100perc.pth http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_fully_100perc.pth

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 