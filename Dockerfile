# Stage 1: Build stage
FROM python:3.9 AS builder

# Set working directory
WORKDIR /app

# Copy dependencies file
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application files
COPY app.py .
COPY resnet18_mnist.pth .

# Ensure the virtual environment is used
ENV PATH="/opt/venv/bin:$PATH"

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
