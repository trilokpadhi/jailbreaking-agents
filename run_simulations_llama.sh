#!/bin/bash
# Run all simulation variants for both base and fine-tuned DeepSeek models.
# Uses vLLM to serve the model (OpenAI-compatible endpoint).
#
# Prerequisites:
#   pip install vllm
#   # For the fine-tuned model, merge LoRA once first:
#   cd playground && python merge_lora.py && cd ..
#
# GPU memory: DeepSeek MoE 16B in bf16 ~32 GB — fits on a single L40S (46 GB).
# Running 3 independent vLLM instances (TP=1 each, one per GPU) for maximum throughput:
#   GPU 0 → port 8003, GPU 1 → port 8004, GPU 2 → port 8005
# 3 instances × 8 workers = 24 concurrent conversations; no inter-GPU sync overhead.

set -e

HF_CACHE="${HF_HOME:-/staging/users/tpadhi1/transformers_cache}"
BASE_MODEL="models/Llama-3.1-8B-Instruct"
FT_MODEL="models/llamaToxic100"          # created by merge_lora.py
VLLM_PORTS=(8003 8004)                           # one per GPU
GPUS=(0 1)
VLLM_ENDPOINTS=$(printf "http://127.0.0.1:%s/v1," "${VLLM_PORTS[@]}")
VLLM_ENDPOINTS=${VLLM_ENDPOINTS%,}                    # strip trailing comma
VLLM_PIDS=()
# INPUT_CSV="datasets/type3_Aug21_combined_balanced_output.csv"
INPUT_CSV="/staging/users/tpadhi1/agent-jail-breaking/datasets/finetune/convo_for_memory_cleaned_0412.csv"
OUTPUT_DIR="generations_llama/"

export HF_HOME="${HF_CACHE}"

start_vllm() {
    local model_path="$1" 
    local model_tag="$2"

    # Kill any stale vLLM servers still occupying the ports (from a previous run).
    echo "Clearing any stale servers on ports ${VLLM_PORTS[*]}..."
    for port in "${VLLM_PORTS[@]}"; do
        fuser -k "${port}/tcp" 2>/dev/null || true
    done
    sleep 2

    echo "=========================================="
    echo "Starting 2 vLLM servers for: ${model_tag}"
    echo "=========================================="
    VLLM_PIDS=()
    for i in 0 1; do
        CUDA_VISIBLE_DEVICES=${GPUS[$i]} python -m vllm.entrypoints.openai.api_server \
            --model "${model_path}" \
            --tensor-parallel-size 1 \
            --port ${VLLM_PORTS[$i]} \
            --served-model-name "${model_tag}" \
            --max-num-seqs 32 \
            > "logs/vllm_server_gpu${GPUS[$i]}.log" 2>&1 &
        VLLM_PIDS+=($!)
        echo "  GPU ${GPUS[$i]} → port ${VLLM_PORTS[$i]} (PID ${VLLM_PIDS[$i]})"
    done

    echo "Waiting for all 2 servers to become ready (model loading may take a few minutes)..."
    for i in 0 1; do
        local ready=false
        for attempt in $(seq 1 120); do
            if ! kill -0 "${VLLM_PIDS[$i]}" 2>/dev/null; then
                echo "ERROR: vLLM GPU ${GPUS[$i]} process died. Check logs/vllm_server_gpu${GPUS[$i]}.log"
                tail -20 "logs/vllm_server_gpu${GPUS[$i]}.log"
                stop_vllm; exit 1
            fi
            if curl -sf "http://127.0.0.1:${VLLM_PORTS[$i]}/health" > /dev/null 2>&1; then
                echo "  GPU ${GPUS[$i]} ready (port ${VLLM_PORTS[$i]})"
                ready=true
                break
            fi
            sleep 5
        done
        if [[ "$ready" != "true" ]]; then
            echo "ERROR: GPU ${GPUS[$i]} did not become ready in 10 min"
            stop_vllm; exit 1
        fi
    done
    echo "All 2 vLLM servers ready!"
}

stop_vllm() {
    echo "Stopping all vLLM servers..."
    for pid in "${VLLM_PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${VLLM_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    echo "All vLLM servers stopped."
}

run_all_variants() {
    local model_tag="$1"
    local out_subdir="${OUTPUT_DIR}${model_tag}/"
    set +e  # Don't exit on non-zero return from python scripts

    echo "--- Running: no memory  [${model_tag}] ---"
    python agent-jailbreak_parallel_v2.py \
        --input_csv "${INPUT_CSV}" \
        --output_dir "${out_subdir}" \
        --planning_method none \
        --model_name "${model_tag}" \
        --endpoints "${VLLM_ENDPOINTS}" \
        --n_workers_per_gpu 8 \
        > "logs/output_${model_tag}_no_memory.log" 2>&1
    echo "Completed no-memory variant"

    echo "--- Running: with memory  [${model_tag}] ---"
    python agent-jailbreak_parallel_v2.py \
        --input_csv "${INPUT_CSV}" \
        --output_dir "${out_subdir}" \
        --planning_method none \
        --with_memory \
        --model_name "${model_tag}" \
        --endpoints "${VLLM_ENDPOINTS}" \
        --n_workers_per_gpu 8 \
        > "logs/output_${model_tag}_memory.log" 2>&1
    echo "Completed with-memory variant"

    echo "--- Running: ReACT  [${model_tag}] ---"
    python agent-jailbreak_parallel_v2.py \
        --input_csv "${INPUT_CSV}" \
        --output_dir "${out_subdir}" \
        --planning_method react \
        --model_name "${model_tag}" \
        --endpoints "${VLLM_ENDPOINTS}" \
        --n_workers_per_gpu 8 \
        > "logs/output_${model_tag}_react.log" 2>&1
    echo "Completed ReACT variant"

    echo "--- Running: CoT  [${model_tag}] ---"
    python agent-jailbreak_parallel_v2.py \
        --input_csv "${INPUT_CSV}" \
        --output_dir "${out_subdir}" \
        --planning_method cot \
        --model_name "${model_tag}" \
        --endpoints "${VLLM_ENDPOINTS}" \
        --n_workers_per_gpu 8 \
        > "logs/output_${model_tag}_cot.log" 2>&1
    echo "Completed CoT variant"
}

# ── 1. Base Llama (no fine-tuning) ──────────────────────────────────────────
MODEL_TAG="llama-base"
start_vllm "${BASE_MODEL}" "${MODEL_TAG}"
run_all_variants "${MODEL_TAG}"
stop_vllm

# ── 2. Fine-tuned Llama ──────────────────────────────────────────────────────
MODEL_TAG="llama-ft"
start_vllm "${FT_MODEL}" "${MODEL_TAG}"
run_all_variants "${MODEL_TAG}"
stop_vllm

echo "=========================================="
echo "All simulations completed!"
echo "Results in: ${OUTPUT_DIR}"
echo "=========================================="
