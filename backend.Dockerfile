FROM python:3.10-slim

WORKDIR /app

# Install system deps for numpy, scipy, scikit-learn, tensorflow
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY main.py .

# Expose backend port
EXPOSE 5000

CMD ["python", "main.py"]