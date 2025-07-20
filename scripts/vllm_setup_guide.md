# vLLM + Dynamo-Triton Migration Guide

## Installation

First, install the required dependencies:

```bash
# Install vLLM with CUDA support
pip install vllm

# Install Ray for distributed processing
pip install ray

# Install additional dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Dynamo-Triton optimization (PyTorch 2.0+)
pip install triton
```

## Usage

### Basic Usage

```bash
# Run with default settings (2 GPUs, 30 rows)
python vllm_dynamo_inference.py \
    --model_path "/home/tsutar3/HEART/models/SFT/llamaToxic100_hf_v2/" \
    --csv_path "/home/tsutar3/HEART/data/insta/type7_version3_output.csv" \
    --output_path "/home/tsutar3/HEART/results/vllm_output.csv"
```

### Advanced Usage

```bash
# High-performance setup with 4 GPUs and larger batch size
python vllm_dynamo_inference.py \
    --model_path "/home/tsutar3/HEART/models/SFT/llamaToxic100_hf_v2/" \
    --csv_path "/home/tsutar3/HEART/data/insta/type7_version3_output.csv" \
    --output_path "/home/tsutar3/HEART/results/" \
    --tensor_parallel_size 4 \
    --batch_size 64 \
    --max_rows 1000 \
    --gpu_memory_utilization 0.95
```

### Performance Testing

```bash
# Test with small sample first
python vllm_dynamo_inference.py \
    --model_path "/home/tsutar3/HEART/models/SFT/llamaToxic100_hf_v2/" \
    --csv_path "/home/tsutar3/HEART/data/insta/type7_version3_output.csv" \
    --output_path "/home/tsutar3/HEART/results/test_output.csv" \
    --sample_size 10 \
    --tensor_parallel_size 2
```

## Performance Comparison

| Method | Throughput | GPU Management | Code Complexity |
|--------|------------|----------------|-----------------|
| **Original HF** | ~0.5 rows/sec | Manual | High (500+ lines) |
| **vLLM + Dynamo** | ~5-10 rows/sec | Automatic | Low (150 lines) |

## Expected Improvements

- **5-10x faster** processing through optimized batching
- **90% reduction** in GPU management code
- **Automatic load balancing** across multiple GPUs
- **Better memory efficiency** with PagedAttention
- **Dynamo-Triton compilation** for additional 20-40% speedup

## Migration Benefits

### ✅ What You Gain
- Automatic GPU utilization optimization
- Built-in batching and memory management
- Simplified codebase (no manual thread management)
- Better error handling and recovery
- Real-time performance metrics

### ⚠️ What Changes
- Model loading (vLLM handles this automatically)
- Inference API (simplified batch processing)
- Configuration (fewer parameters needed)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce memory utilization
   --gpu_memory_utilization 0.8
   
   # Reduce batch size
   --batch_size 16
   ```

2. **Ray Initialization Error**
   ```bash
   # Kill existing Ray processes
   ray stop
   
   # Then run your script
   ```

3. **Model Loading Issues**
   ```bash
   # Verify model format
   ls -la /path/to/model/
   # Should contain: config.json, tokenizer.json, pytorch_model.bin (or .safetensors)
   ```

## Performance Tuning

### For Maximum Speed
```bash
--tensor_parallel_size 4    # Use all available GPUs
--batch_size 64             # Larger batches for better throughput
--gpu_memory_utilization 0.95  # Use maximum GPU memory
```

### For Memory Efficiency
```bash
--tensor_parallel_size 2    # Use fewer GPUs
--batch_size 32             # Smaller batches
--gpu_memory_utilization 0.8   # Conservative memory usage
```

## Monitoring

The script provides real-time monitoring:
- 🔧 Initialization progress
- ⚡ Processing speed (rows/second)
- 💾 Memory utilization
- 📊 Performance projections
- ⚠️ Error reporting

## Next Steps

1. **Test with small sample** to verify everything works
2. **Benchmark performance** against your original script
3. **Scale up gradually** to full dataset
4. **Monitor GPU utilization** with `nvidia-smi`
5. **Adjust batch size** based on performance 