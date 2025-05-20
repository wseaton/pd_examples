# Setting this allows creating a symlink to Justfile from another dir
set working-directory := "/home/weaton/pd_examples/"

# Needed for the proxy server
vllm-directory := "/home/weaton/vllm/" 

# MODEL := "meta-llama/Llama-3.1-8B-Instruct"
TP_SIZE := "1"
PREFILL_GPUS := "0"
DECODE_GPUS := "5,6"

MODEL := "deepseek-ai/DeepSeek-V2-Lite"
# TP_SIZE := "2"
# PREFILL_GPUS := "3,4"
# DECODE_GPUS := "5,6"

MEMORY_UTIL := "0.5"

port PORT: 
  @python port_allocator.py {{PORT}}

# For comparing against baseline vLLM
vanilla_serve:
    VLLM_SERVER_DEV_MODE=1 \
    CUDA_VISIBLE_DEVICES={{PREFILL_GPUS}} \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve {{MODEL}} \
      --port $(just port 8192) \
      --disable-log-requests \
      --tensor-parallel-size {{TP_SIZE}} \
      --gpu-memory-utilization {{MEMORY_UTIL}} \
      --trust-remote-code

prefill:
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5557) \
    UCX_LOG_LEVEL=debug \
    CUDA_VISIBLE_DEVICES={{PREFILL_GPUS}} \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve {{MODEL}} \
      --port $(just port 8100) \
      --disable-log-requests \
      --tensor-parallel-size {{TP_SIZE}} \
      --gpu-memory-utilization {{MEMORY_UTIL}} \
      --trust-remote-code \
      --max-model-len 2048 \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

decode:
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5558) \
    UCX_LOG_LEVEL=info \
    CUDA_VISIBLE_DEVICES={{DECODE_GPUS}} \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve {{MODEL}} \
      --port $(just port 8200) \
      --disable-log-requests \
      --tensor-parallel-size {{TP_SIZE}} \
      --trust-remote-code \
      --max-model-len 2048 \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

proxy:
    python "{{vllm-directory}}tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
      --port $(just port 8192) \
      --prefiller-port $(just port 8100) \
      --decoder-port $(just port 8200)


send_request:
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
  curl -X POST http://localhost:$(just port 8100)/reset_prefix_cache && \
  curl -X POST http://localhost:$(just port 8200)/reset_prefix_cache
  

eval:
  lm_eval --model local-completions --tasks gsm8k \
    --model_args model={{MODEL}},base_url=http://127.0.0.1:$(just port 8192)/v1/completions,num_concurrent=5,max_retries=3,tokenized_requests=False \
    --limit 100

dp:
  VLLM_SERVER_DEV_MODE=1 \
  VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5558) \
  UCX_LOG_LEVEL=info \
  VLLM_LOGGING_LEVEL="DEBUG" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python dp.py \
          --model="{{MODEL}}" \
          --master-port $(just port 8300) \
          --dp-size=2 \
          --tp-size=1 \
          --trust-remote-code \
          --enforce-eager \
          --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'


