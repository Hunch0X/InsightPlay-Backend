FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# The default PyPI torch wheel bundles CUDA and runs on GPU hosts started with the NVIDIA container runtime; it also runs on CPU.
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
