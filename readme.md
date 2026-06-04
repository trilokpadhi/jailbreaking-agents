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

### Step 6: Run the code
```bash
python agent-jailbreak_parallel.py --input_csv datasets/pinxian-pipeline/original/original_april16/instagram/type6_v13u_output.csv --output_dir generations_/jail_breaking/ > output_JB_insta_type6.log 2>&1 &
```
```bash
python agent-jailbreak-BM_parallel.py --input_csv datasets/pinxian-pipeline/original/original_april16/instagram/type6_v13u_output.csv --output_dir generations_/jail_breaking/ > output_JB_BM_insta_type6.log 2>&1 &
```

How to run ollama server in a different port and different gpu:
```bash
#### GPU 0 - Port 11434
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 ../ollamatry/bin/ollama serve > ollama_gpu0.log 2>&1 &

#### GPU 1 - Port 11435  
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 ../ollamatry/bin/ollama serve > ollama_gpu1.log 2>&1 &

#### GPU 2 - Port 11436
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=127.0.0.1:11436 ../ollamatry/bin/ollama serve > ollama_gpu2.log 2>&1 &

#### GPU 3 - Port 11437
CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=127.0.0.1:11437 ../ollamatry/bin/ollama serve > ollama_gpu3.log 2>&1 &
```

#### Pull model on each instance
```bash
OLLAMA_HOST=127.0.0.1:11434 ../ollamatry/bin/ollama pull llama3.1
OLLAMA_HOST=127.0.0.1:11435 ../ollamatry/bin/ollama pull llama3.1
OLLAMA_HOST=127.0.0.1:11436 ../ollamatry/bin/ollama pull llama3.1
OLLAMA_HOST=127.0.0.1:11437 ../ollamatry/bin/ollama pull llama3.1
```

## Update as of 18th August 2025 - Run the simulations in parallel
```bash 
bash run_simulations.sh
```

## Evaluating Conversations with LLM Judge

### Single Unified Script: `llm-judge.py`

All functionality is now in one script with different modes:

**Two Classification Modes:**
- **`--mode realtime`**: Fast parallel API calls (~10 seconds for 35 tasks)
  - ✅ Immediate results
  - ✅ Auto-retry with exponential backoff
  - ❌ ~2x more expensive
  
- **`--mode batch`**: Cheaper batch API (~10-30 minutes for 35 tasks)
  - ✅ 50% cheaper
  - ✅ Auto-retry for all API operations
  - ❌ Slower (minutes to hours)

**Evaluation:**
- **`--evaluate`**: Computes all metrics and statistics

### Quick Start Examples

```bash
# Set your API key
export OPENAI_API_KEY_TP="your-api-key-here"

# 1. Classify with real-time API (FAST)
python llm-judge.py --input_csv data.csv --mode realtime

# 2. Classify with batch API (CHEAPER)
python llm-judge.py --input_csv data.csv --mode batch

# 3. Evaluate already-classified results
python llm-judge.py --input_csv data_output_with_analysis.csv --evaluate

# 4. Classify AND evaluate in one command
python llm-judge.py --input_csv data.csv --mode realtime --evaluate
```

### Testing with Sample Data

Test with the sample file first (5 conversations):

```bash
# FAST: Real-time API (completes in ~10 seconds)
python llm-judge.py --input_csv datasets/finetune/convo_for_finetuning_sample.csv --mode realtime

# Then evaluate
python llm-judge.py --input_csv convo_for_finetuning_sample_output_with_analysis.csv --evaluate

# OR do both in one command
python llm-judge.py --input_csv datasets/finetune/convo_for_finetuning_sample.csv --mode realtime --evaluate
```

### Running on Full Dataset

```bash
# Option A: Real-time API (Faster, 2-3 hours for full dataset)
python llm-judge.py --input_csv datasets/finetune/convo_for_finetuning.csv --mode realtime --evaluate

# Option B: Batch API (Cheaper, several hours)
python llm-judge.py --input_csv datasets/finetune/convo_for_finetuning.csv --mode batch --evaluate
```

### Advanced Options

```bash
# Customize parallel workers for real-time mode
python llm-judge.py --input_csv data.csv --mode realtime --max-workers 20

# Use different model
python llm-judge.py --input_csv data.csv --mode realtime --model gpt-4o

# Analyze more turns per conversation
python llm-judge.py --input_csv classified_data.csv --evaluate --max-turns 10
```

### Output Files

**Real-time/Batch Classification:**
- `{filename}_output_with_analysis.csv` - Main results with classifications
- `{filename}_batch_tasks.jsonl` - Batch API tasks (batch mode only)
- `{filename}_batch_results.jsonl` - Batch API results (batch mode only)  
- `{filename}_batch_metadata.json` - Job metadata (batch mode only)
- `{filename}_batch_tracking.log` - Processing log (batch mode only)

**Evaluation:**
- `{filename}_evaluated.csv` - Enriched with metrics (refusal_count, harasser_count, attack_success, tts, tts_rate)
- Console output with full statistical report

### Evaluation Metrics

The evaluation provides:
- **Total Conversations**: Number of conversations analyzed
- **Refusal Rate**: Percentage of turns where the model refused
- **Attack Success Rate**: Percentage of conversations with at least one harassment instance
- **Avg TTS** (Turn to Success): Average number of turns before successful attack
- **Avg TTS Rate**: Average of 1/TTS across successful attacks
- **Per-metric Analysis**: For each cyberbullying category:
  - Any-Turn True: % of conversations with at least one instance
  - All-Turns True: % of conversations where all turns show this behavior
  - Per-Turn Rate: % of turns showing this behavior (breakdown by turn 1-5)

