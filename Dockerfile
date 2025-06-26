# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_dashboard.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_dashboard.txt

# Copy application code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p data

# Expose the port Streamlit runs on
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
