# syntax=docker/dockerfile:1

# Use PyTorch image that matches repo pins (torch 2.3.1, CUDA 12.1, cuDNN8) and Python 3.10
# Why: avoid wheel resolution issues across Python versions; keep environment consistent.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install only deps not already provided by the base PyTorch image.
# Base includes torch==2.3.1, torchvision==0.18.1, torchaudio==2.3.1.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      --find-links https://data.pyg.org/whl/torch-2.3.1+cu121.html \
      torch-geometric==2.5.3 \
      torch-scatter==2.1.2 \
      torch-sparse==0.6.18 \
      torch-cluster==1.6.3 \
      torch-spline-conv==1.2.2 && \
    pip install --no-cache-dir \
      numpy==1.26.4 \
      pandas==2.2.2 \
      scipy==1.13.1 \
      scikit-learn==1.4.2 \
      tqdm==4.54.1 \
      PyYAML \
      matplotlib \
      seaborn \
      geopy==2.4.1 \
      geopandas==1.0.1 \
      shapely==2.0.6 \
      osmnx==2.0.1 \
      pyproj==3.7.0 \
      tensorboard==2.18.0 \
      transformers==4.41.2 \
      huggingface-hub==0.23.3 \
      accelerate==1.1.1 \
      requests==2.31.0 \
      joblib==1.4.0 \
      regex \
      protobuf==3.20.3

# Default to an interactive shell; run project commands via docker compose
CMD ["bash"]
