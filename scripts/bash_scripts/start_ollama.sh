#!/bin/bash

# Get full path to the ollama binary
source ~/.bashrc
OLLAMA_BIN=$(which ollama)

if [[ ! -x "$OLLAMA_BIN" ]]; then
  echo "❌ Error: 'ollama' not found in PATH."
  exit 1
fi

# Define GPU-port-model mapping
declare -A MODEL_PORTS
MODEL_PORTS=(
  ["llama3.2:3b-instruct-fp16_0"]="0 11438"
  ["llama3.2:3b-instruct-fp16_1"]="1 11439"
  ["llama3.2:3b-instruct-fp16_2"]="2 11440"
  ["llama3.2:3b-instruct-fp16_3"]="3 11441"
)

# Kill previous tmux sessions (optional cleanup)
for model in "${!MODEL_PORTS[@]}"; do
  session_name="ollama_${model}"
  tmux kill-session -t "$session_name" 2>/dev/null
done

# Start new tmux sessions for each model/GPU/port
for model in "${!MODEL_PORTS[@]}"; do
  IFS=' ' read -r gpu port <<< "${MODEL_PORTS[$model]}"
  session_name="ollama_${model}"
  
  echo "🚀 Launching $model on GPU $gpu (port $port) in tmux session: $session_name"
  
  tmux new-session -d -s "$session_name" \
    "CUDA_VISIBLE_DEVICES=$gpu OLLAMA_NUM_PARALLEL=16 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_HOST=127.0.0.1:$port $OLLAMA_BIN serve"
done


# CUDA_VISIBLE_DEVICES=0,1,2,3 OLLAMA_NUM_PARALLEL=8 OLLAMA_HOST=127.0.0.1:11438 ollama serve

echo -e "\n✅ All tmux Ollama servers launched."

# Print usage instructions
for model in "${!MODEL_PORTS[@]}"; do
  session_name="ollama_${model}"
  echo "🔎 To monitor: tmux attach -t $session_name"
done
