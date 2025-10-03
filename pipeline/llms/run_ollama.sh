#!/bin/bash

# Check if "ollama serve" is available
if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama not found."
    if [[ "$(uname)" == "Linux" ]]; then
        echo "Installing Ollama ..."
        OLLAMA_VERSION=0.6.5 curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "Error: Visit https://ollama.com/download to install Ollama." >&2
        exit 1
    fi
fi

trap 'echo "Running cleanup..."; pkill -f "ollama serve"; exit' SIGINT

echo "Running Ollama server ..."
# Temporarily helps fix the issue with Ollama memory leak: https://github.com/ollama/ollama/issues/10132#issuecomment-2785711746
while true; do
    # Kill any existing process
    pkill -f "ollama serve"

    # Start fresh
    OLLAMA_HOST="0.0.0.0:11434" OLLAMA_FLASH_ATTENTION=1 nohup ollama serve > ollama.log 2>&1 &

    # Wait 5 minutes
    sleep 300
done
