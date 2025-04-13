#!/bin/bash

# Temporarily helps fix the issue with Ollama memory leak: https://github.com/ollama/ollama/issues/10132#issuecomment-2785711746

while true; do
    # Kill any existing process
    pkill -f "ollama serve"

    # Start fresh
    OLLAMA_HOST="0.0.0.0:11434" OLLAMA_FLASH_ATTENTION=1 nohup ollama serve > ollama.log 2>&1 &

    # Wait 15 minutes
    sleep 900
done
