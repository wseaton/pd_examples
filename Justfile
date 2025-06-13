# Setting this allows creating a symlink to Justfile from another dir
set working-directory := "/home/weaton/pd_examples/"

# Needed for the proxy server
vllm-directory := "/home/weaton/vllm/" 

# MODEL := "meta-llama/Llama-1.1-8B-Instruct"
TP_SIZE := "1"
DP_SIZE := "1"

export UCX_TLS := "^dc"
# export CUDA_DEVICE_ORDER := "PCI_BUS_ID"
# export UCX_TLS := "tcp"
# PREFILL_GPUS := "1"
# DECODE_GPUS := "4"
#
export VLLM_SERVER_DEV_MODE := "1"

# MODEL := "deepseek-ai/DeepSeek-V2-Lite"
MODEL := "Qwen/Qwen3-0.6B"
# TP_SIZE := "2"
PREFILL_GPUS := "2"
DECODE_GPUS := "3"

MEMORY_UTIL := "0.3"

port PORT: 
  @python port_allocator.py {{PORT}}

# For comparing against baseline vLLM
vanilla_serve:
    CUDA_VISIBLE_DEVICES={{PREFILL_GPUS}} \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve {{MODEL}} \
      --port $(just port 8192) \
      --disable-log-requests \
      --tensor-parallel-size {{TP_SIZE}} \
      --data-parallel-size {{DP_SIZE}} \
      --gpu-memory-utilization {{MEMORY_UTIL}} \
      --trust-remote-code

prefill:
    VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5777) \
    UCX_LOG_LEVEL=warn \
    CUDA_VISIBLE_DEVICES={{PREFILL_GPUS}} \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve {{MODEL}} \
      --port $(just port 8100) \
      --tensor-parallel-size {{TP_SIZE}} \
      --data-parallel-size {{DP_SIZE}} \
      --gpu-memory-utilization {{MEMORY_UTIL}} \
      --trust-remote-code \
      --max-model-len 2048 \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

decode:
    VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5778) \
    CUDA_VISIBLE_DEVICES={{DECODE_GPUS}} \
    UCX_LOG_LEVEL=debug \
    NCCL_DEBUG=INFO \
    NIXL_LOG_LEVEL=DEBUG \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve {{MODEL}} \
      --port $(just port 8200) \
      --tensor-parallel-size {{TP_SIZE}} \
      --data-parallel-size {{DP_SIZE}} \
      --gpu-memory-utilization {{MEMORY_UTIL}} \
      --trust-remote-code \
      --max-model-len 2048 \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

proxy:
    VLLM_SERVER_DEV_MODE=1 \
    python "{{vllm-directory}}tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
      --port $(just port 8192) \
      --prefiller-port $(just port 8100) \
      --decoder-port $(just port 8200)


send_request:
  curl -X POST http://localhost:$(just port 8192)/v1/completions \
    -H "Content-Type: application/json" \
    -d '{ \
      "model": "{{MODEL}}", \
      "prompt": "Red Hat is the best open source company by far across Linux, K8s, and AI, and vLLM has the greatest community in open source AI software infrastructure. Prefill-decode disaggregation will enable vLLM to ", \
      "max_tokens": 150, \
      "temperature": 0.7 \
    }'

send_request_direct:
  curl -X POST http://localhost:$(just port 8200)/v1/completions \
    -H "Content-Type: application/json" \
    -d '{ \
      "model": "{{MODEL}}", \
      "prompt": "Red Hat is the best open source company by far across Linux, K8s, and AI, and vLLM has the greatest community in open source AI software infrastructure. Prefill-decode disaggregation will enable vLLM to ", \
      "max_tokens": 150, \
      "temperature": 0.7 \
    }'

benchmark:
  python {{vllm-directory}}/benchmarks/benchmark_serving.py --port $(just port 8192) --model {{MODEL}} --dataset-name random --random-input-len 1000 --random-output-len 100

benchmark_one_concurrent:
  python {{vllm-directory}}/benchmarks/benchmark_one_concurrent_req.py \
    --model {{MODEL}} \
    --input-len 1000 \
    --output-len 100 \
    --num-requests 10 \
    --seed 12147 \
    --port $(just port 8192)

reset_prefix_cache:
  curl -X POST http://localhost:$(just port 8200)/reset_prefix_cache && \
      curl -X POST http://localhost:$(just port 8100)/reset_prefix_cache
  

clear_ports: 
  lsof -i:$(just port 8200) | awk 'NR > 1 {print $2}' | xargs kill -9 || true
  lsof -i:$(just port 8100) | awk 'NR > 1 {print $2}' | xargs kill -9 || true 
  lsof -i:$(just port 8192) | awk 'NR > 1 {print $2}' | xargs kill -9 || true


eval:
  lm_eval --model local-completions --tasks gsm8k \
    --model_args model={{MODEL}},base_url=http://127.0.0.1:$(just port 8192)/v1/completions,num_concurrent=5,max_retries=3,tokenized_requests=False \
    --limit 100

clean_vllm: 
  #!/bin/bash
  # Clean VLLM processes and ports for justfile integration

  echo "Cleaning vllm processes and ports..."

  # Kill all vllm processes (only for current user)
  pkill -u $(whoami) -f "vllm serve" 2>/dev/null || true
  sleep 2

  # Force kill any remaining (only for current user)
  pkill -9 -u $(whoami) -f "vllm serve" 2>/dev/null || true
  sleep 1

  # Clean up ports
  lsof -i:$(just port 8200) | awk 'NR > 1 {print $2}' | xargs kill -9 2>/dev/null || true
  lsof -i:$(just port 8100) | awk 'NR > 1 {print $2}' | xargs kill -9 2>/dev/null || true 
  lsof -i:$(just port 8192) | awk 'NR > 1 {print $2}' | xargs kill -9 2>/dev/null || true

  # Check if any vllm processes remain (only for current user)
  remaining=$(ps -u $(whoami) -o pid,cmd | grep -v grep | grep "vllm serve" | wc -l)
  if [ "$remaining" -gt 0 ]; then
      echo "Warning: $remaining vllm processes still running"
      ps -u $(whoami) -o pid,cmd | grep -v grep | grep "vllm serve"
      exit 1
  fi

  echo "Clean complete"


clear_gpu:
   nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader | rg $(whoami) | awk '{print $1}' | sed 's/,*$//g' | xargs kill -9
