# Agent Jail Breaking

This repository contains the code for the Agent Jail Breaking project using Ollama on Linux.

## Prerequisites

- Linux operating system
- Python 3.x installed
- Git installed

## Installation

### Step 1: Install Ollama

To install Ollama, run the following command:

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Clone the repository

To clone the repository, run the following command:

```sh
git clone https://github.com/trilokpadhi/jailbreaking-agents.git
```

### Step 3: Install the dependencies
pip install autogen

### Step 4: Start Ollama server

To start the Ollama server, run the following command:

```sh 
ollama/bin init
ollama pull llama3.1
```

### Step 5: Run the code
python agent-jailbreak.py

#### Commands 
To start the ollama server, run the following command:
```bash
../ollamatry/bin/ollama serve
```
How to run ollama server in a different port and different gpu:
```bash
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=127.0.0.1:11435 ../ollamatry/bin/ollama serve > output_ollama2.log 2>&1 &
```

### Troubleshooting

#### Checking GPT Batch Job Progress

To check the progress of a GPT batch job, you can use the following `curl` command:

```bash
curl https://api.openai.com/v1/batches/batch_68008730508481909bc5e84265dd2512 \
    -H "Authorization: Bearer $OPENAI_API_KEY_TP"
```

Make sure to replace `$OPENAI_API_KEY_TP` with your actual OpenAI API key.

```bash
curl https://api.openai.com/v1/batches/batch_68008730508481909bc5e84265dd2512 \
  -H "Authorization: Bearer $OPENAI_API_KEY_TP" \
  | jq '{status, request_counts}'

```