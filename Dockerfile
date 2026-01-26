# Dockerfile for Hugging Face Spaces (Docker SDK)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (using JSON array format for paths with spaces)
COPY ["Source Code/requirements.txt", "./requirements.txt"]

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire Source Code directory (using JSON array format and explicit directory destination)
COPY ["Source Code/", "./"]

# Expose the port that Hugging Face Spaces expects (7860)
EXPOSE 7860

# Define the command to run the application
CMD ["streamlit", "run", "Stock-RL.py", "--server.port=7860", "--server.address=0.0.0.0"]
