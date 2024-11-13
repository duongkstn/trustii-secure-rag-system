import os

SERVICE_HOST = os.getenv("SERVICE_HOST", "localhost")
SERVICE_PORT = os.getenv("SERVICE_PORT", "8000")
WORKERS = os.getenv("WORKERS", 1)
OLLAMA_URL = os.getenv("OLLAMA_URL", "localhost:11434")