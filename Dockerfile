# Use Python 3.9 as the base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Hugging Face Spaces expects the app to run on port 7860
EXPOSE 7860

# Command to run the app using Gunicorn with Eventlet worker for SocketIO
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:7860", "app:app"]