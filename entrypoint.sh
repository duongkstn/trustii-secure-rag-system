#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "🔴 Retrieve Ollama models..."
ollama pull all-minilm:l6-v2
ollama pull qwen2.5:latest
ollama list
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid