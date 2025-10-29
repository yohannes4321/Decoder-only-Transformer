# Use PyTorch image with CUDA + cuDNN for GPU acceleration
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy requirements first (for efficient caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the project files
COPY . /app

# Create directories for model/data (if not present)
RUN mkdir -p /app/Data /app/assets

# Expose the port uvicorn will use
EXPOSE 8000

CMD ["sh", "-c", "uvicorn decoder_only_transformer.app:app --host 0.0.0.0 --port $PORT"]
