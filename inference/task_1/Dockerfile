# Start with a Python base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy over requirements
COPY inference/task_1/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


COPY inference/task_1/entrypoint.py .

# Copy over the app files
COPY checkpoints/*.joblib checkpoints/

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "entrypoint:app", "--host", "0.0.0.0", "--port", "8000"]
