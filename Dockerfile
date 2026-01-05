FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt .
COPY api/main.py api/
COPY src/ src/
COPY models/ models/
COPY mlruns/ mlruns/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]