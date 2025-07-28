FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model during build time (when internet is available)
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    import torch; \
    model_id = 'microsoft/DialoGPT-small'; \
    print('Downloading DialoGPT-small model...'); \
    tokenizer = AutoTokenizer.from_pretrained(model_id); \
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32); \
    print('Model downloaded and cached successfully!')"

# Copy Challenge 1a components
COPY challenge_1a/ ./challenge_1a/

# Copy application code
COPY process_documents.py .

# Set environment variables for better compatibility
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_DISABLE_MKL_WARNINGS=1
ENV OMP_NUM_THREADS=1

# Ensure the script is executable
RUN chmod +x process_documents.py

# Run the main script
CMD ["python", "process_documents.py"]
