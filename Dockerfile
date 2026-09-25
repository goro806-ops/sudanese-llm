# Use official Python lightweight image
FROM python:3.12-slim

WORKDIR /app

# Copy setup_project.py and install dependencies
COPY setup_project.py /app/
RUN python setup_project.py
RUN pip install --no-cache-dir fastapi uvicorn pydantic httpx pytest

# Copy all repository files into container
COPY . /app/

# Expose port for FastAPI
EXPOSE 7860

# Command to run the FastAPI app (Hugging Face Spaces default port is 7860)
CMD ["uvicorn", "web_server.app:app", "--host", "0.0.0.0", "--port", "7860"]
