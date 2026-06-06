# Jailbreaking Agents

Research codebase for studying cyberbullying jailbreak attacks on LLM-based agents. We evaluate how fine-tuned and base LLMs (Qwen3, LLaMA, DeepSeek) respond under various agent prompting strategies: **none**, **CoT**, **ReACT**, and **memory injection**.

## Repository Structure

```
agent-jailbreak_parallel_v2.py          # Main simulation driver (parallel vLLM)
llm-judge.py / llm-judge-realtime.py    # LLM judge for classifying responses
llm-judge-batch.py / llm-judge-fast.py  # Batch & fast judge variants
run_simulations*.sh                     # Run scripts per model family
run_llm_judge*.sh                       # Judge run scripts per model

datasets/                               # Input conversation datasets
agent-responses/                        # Labeled response CSVs + result summaries
evaluation/                             # Batch eval results, qual analysis, metadata
llm_judge_results/                      # LLM judge output CSVs
synthetic-conversation-generation-eval/ # Synthetic conv generation & quality eval
tables/                                 # LaTeX tables for paper
logs/                                   # Experiment run logs & judge logs
playground/                             # Training scripts, analysis notebooks, plots
```

## Setup

```bash
pip install autogen openai pandas
export OPENAI_API_KEY_TP="your-key-here"
```

Simulations use vLLM. Start servers before running:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model <model-path> --port 8000 &
```

## Running Simulations

```bash
# Qwen3 (base + FT, all strategies)
bash run_simulations.sh

# DeepSeek or LLaMA variants
bash run_simulations_deepseek.sh
bash run_simulations_llama.sh
```

Each script runs `agent-jailbreak_parallel_v2.py` across all strategy configs and saves outputs to `agent-responses/`.

## LLM Judge Evaluation

```bash
# Fast: real-time API (~10s for 35 tasks)
python llm-judge.py --input_csv agent-responses/data.csv --mode realtime --evaluate

# Cheap: batch API (50% cheaper, minutes to hours)
python llm-judge.py --input_csv agent-responses/data.csv --mode batch --evaluate
```

Key metrics: **Attack Success Rate**, **Refusal Rate**, **TTS** (Turns to Success), **per-category per-turn rates**.

```bash
# Run all judges in sequence
bash run_eval.sh
```

## Synthetic Conversation Generation

Scripts in `synthetic-conversation-generation-eval/`:

```bash
# Generate synthetic conversations
python synthetic-conversation-generation-eval/conversation_generation.py

# Quality & relatedness filtering
python synthetic-conversation-generation-eval/quality_judge.py
python synthetic-conversation-generation-eval/relatedness_judge.py
```

## Large Files

Raw agent generation outputs (`generations_deepseek/`, `generations_llama/`, `generations_qwen/`, ~1.2 GB total) are gitignored. To share or download:

> **Recommended**: Upload to [Hugging Face Datasets](https://huggingface.co/datasets)

```bash
pip install huggingface_hub
huggingface-cli upload <your-org>/<dataset-name> generations_qwen/ --repo-type dataset
```


