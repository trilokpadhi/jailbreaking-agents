# Llama Guard Multi-GPU Analysis

This script processes conversations from a CSV file and analyzes them for safety using Meta's Llama Guard model across multiple GPUs.

## Installation

First, install the required dependencies:

```bash
pip install -r llama_guard_requirements.txt
```

## Usage

### 1. Basic Multi-GPU Usage

```bash
# Using 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc-per-node=2 \
  llama-guard.py \
    --input_csv /path/to/your/conversations.csv \
    --output_dir /path/to/output/directory \
    --conversation_column "agent3_output_converted"
```

### 2. Single GPU Usage

```bash
python llama-guard.py \
  --input_csv /path/to/your/conversations.csv \
  --output_dir /path/to/output/directory \
  --conversation_column "agent3_output_converted"
```

### 3. Full Example with All Options

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc-per-node=2 \
  llama-guard.py \
    --input_csv /home/tsutar3/HEART/data/convo_for_memory.csv \
    --output_dir /home/tsutar3/HEART/llama_guard_results \
    --conversation_column "agent3_output_converted" \
    --model_path "meta-llama/LlamaGuard-7b" \
    --limit_rows 100 \
    --checkpoint_interval 25 \
    --max_length 2048
```

## Parameters

- `--input_csv`: Path to your CSV file containing conversations
- `--output_dir`: Directory where results will be saved
- `--conversation_column`: Name of the column containing conversation data (default: "agent3_output_converted")
- `--model_path`: Hugging Face model path for Llama Guard (default: "meta-llama/LlamaGuard-7b")
- `--limit_rows`: Limit number of rows to process (0 = all rows)
- `--checkpoint_interval`: Save progress every N rows (default: 50)
- `--max_length`: Maximum sequence length for model input (default: 2048)

## Output

The script will create:
- One CSV file per GPU: `llama_guard_analysis_rank_X_of_Y.csv`
- Each output file contains the original data plus two new columns:
  - `safety_label`: "safe", "unsafe", "uncertain", or "error"
  - `safety_explanation`: Detailed explanation from Llama Guard

## Merging Results

After processing completes, merge the results from all GPUs:

```bash
python merge_llama_guard_results.py \
  --output_dir /path/to/output/directory \
  --output_filename "final_safety_analysis.csv"
```

## Safety Categories

Llama Guard analyzes conversations for these unsafe content categories:

1. **Violence and Hate**: Violent content, hate speech, discrimination
2. **Sexual Content**: Graphic sexual discussions or content
3. **Criminal Planning**: Assistance with illegal activities
4. **Guns and Illegal Weapons**: Firearm-related crimes
5. **Regulated or Controlled Substances**: Drug trafficking, illegal substances
6. **Self-Harm**: Encouraging self-harm or suicide

## Performance Tips

- Use multiple GPUs for faster processing on large datasets
- Adjust `--max_length` based on your conversation lengths
- Use `--limit_rows` for testing on smaller subsets
- Monitor GPU memory usage and adjust batch processing if needed

## Expected Conversation Format

The script expects conversations in these formats:

1. **JSON string**: `'[{"message": "Hello", "speaker": "User1"}, {"message": "Hi", "speaker": "User2"}]'`
2. **Python list string**: `"[{'message': 'Hello'}, {'message': 'Hi'}]"`
3. **Dictionary with message/content keys**

The script will automatically detect and parse different formats.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--max_length` or use fewer GPUs
2. **Model not found**: Ensure you have access to the Llama Guard model on Hugging Face
3. **Parsing errors**: Check your conversation data format
4. **Slow processing**: Use more GPUs or reduce conversation lengths

### Performance Monitoring

The script logs memory usage and processing speed. Check the logs for:
- GPU memory allocation
- Processing rate (rows/second)
- Error counts and types 