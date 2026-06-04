#!/bin/bash

# GPU 0 - Port 11434
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 ../ollamatry/bin/ollama serve > ollama_gpu0.log 2>&1 &
PID1=$!
# GPU 1 - Port 11435  
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 ../ollamatry/bin/ollama serve > ollama_gpu1.log 2>&1 &
PID2=$!
# GPU 2 - Port 11436
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=127.0.0.1:11436 ../ollamatry/bin/ollama serve > ollama_gpu2.log 2>&1 &
PID3=$!
# GPU 3 - Port 11437
CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=127.0.0.1:11437 ../ollamatry/bin/ollama serve > ollama_gpu3.log 2>&1 &
PID4=$!
echo "Started 4 Ollama instances on ports 11434-11437"
sleep 10  # Wait for servers to start

# Pull model on each instance
OLLAMA_HOST=127.0.0.1:11434 ../ollamatry/bin/ollama pull llama3.1
OLLAMA_HOST=127.0.0.1:11435 ../ollamatry/bin/ollama pull llama3.1
OLLAMA_HOST=127.0.0.1:11436 ../ollamatry/bin/ollama pull llama3.1
OLLAMA_HOST=127.0.0.1:11437 ../ollamatry/bin/ollama pull llama3.1


# Run the parallel processing script for no memory version
python agent-jailbreak_parallel_v2.py --input_csv datasets/type3_Aug21_combined_balanced_output.csv --output_dir generations_/ --planning_method none > output_no_memory.log 2>&1 
echo "Completed no memory version"
# Run the parallel processing script for memory version
python agent-jailbreak_parallel_v2.py --input_csv datasets/type3_Aug21_combined_balanced_output.csv --output_dir generations_/ --planning_method none --with_memory > output_memory.log 2>&1 
echo "Completed memory version"
# Run the parallel processing script for ReACT version
python agent-jailbreak_parallel_v2.py --input_csv datasets/type3_Aug21_combined_balanced_output.csv --output_dir generations_/ --planning_method react > output_react.log 2>&1
echo "Completed ReACT version"
# Run the parallel processing script for cot version
python agent-jailbreak_parallel_v2.py --input_csv datasets/type3_Aug21_combined_balanced_output.csv --output_dir generations_/ --planning_method cot > output_cot.log 2>&1
echo "Completed cot version"


# Lets also kill the Ollama servers
echo "Killing Ollama servers..."
kill $PID1 $PID2 $PID3 $PID4
echo "All Ollama servers killed"
echo "All simulations completed successfully"
echo "You can find the output in the generations_ directory"
