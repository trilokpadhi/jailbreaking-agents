CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /home/shared_models/Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000 --tensor-parallel-size 4 --max-model-len 32768 --served-model-name Qwen3-30B-Instruct > vllm_q30inst.log 2>&1 &
# PID_VLLM=$!

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --model-path /home/shared_models/Qwen/Qwen3-Next-80B-A3B-Instruct --served-model-name Qwen/Qwen3-Next-80B-A3B-Instruct --port 30000 --tp-size 4 --context-length 32768 --mem-fraction-static 0.8 --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 > sglang_qnext.log 2>&1 &

vllm serve /home/shared_models/Qwen/Qwen3-Next-80B-A3B-Instruct --port 8001 --tensor-parallel-size 4 --max-model-len 32768 --served-model-name Qwen3-Next-80B-A3B-Instruct > vllm_qnext.log 2>&1 &

sleep 10
echo "Starting VLLM Inference Script"

python /home/tsutar3/jailbreaking-agents/scripts/inference/vllm_convo_gen.py 