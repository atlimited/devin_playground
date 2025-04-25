
if [ -f .env ]; then
    source .env
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

echo "Starting LiteLLM proxy server on $HOST:$PORT..."
python proxy_server.py
