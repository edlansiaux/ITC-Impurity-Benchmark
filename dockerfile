FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p results/raw_results results/tables results/figures data/datasets

# Set Python path
ENV PYTHONPATH=/app

# Command to run when container starts
CMD ["python", "run_all_benchmarks.py"]
