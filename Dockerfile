FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by OpenCV/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY src ./src
COPY models/resnet18_webcam_caps_finetuned_v2.pth ./models/resnet18_webcam_caps_finetuned_v2.pth

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]