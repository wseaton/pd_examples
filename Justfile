# Setting this allows creating a symlink to Justfile from another dir
set working-directory := "/home/weaton/pd_examples/"

# Needed for the proxy server
vllm-directory := "/home/weaton/vllm/"

# MODEL := "meta-llama/Llama-1.1-8B-Instruct"
TP_SIZE := "1"
DP_SIZE := "1"

# export UCX_TLS := "^dc"
# export CUDA_DEVICE_ORDER := "PCI_BUS_ID"
# export UCX_TLS := "tcp"
# PREFILL_GPUS := "1"
# DECODE_GPUS := "4"
#
export VLLM_SERVER_DEV_MODE := "1"
export VLLM_ATTENTION_BACKEND := "FLASH_ATTN"
export VLLM_NIXL_HANDSHAKE_METHOD := "http"

# MODEL := "deepseek-ai/DeepSeek-V2-Lite"
MODEL := "Qwen/Qwen3-0.6B"
# MODEL := "RedHatAI/Llama-3.1-8B-Instruct"
# TP_SIZE := "2"
PREFILL_GPUS := "2"
DECODE_GPUS := "3,4"

MEMORY_UTIL := "0.3"

# SSL certificate paths
SSL_CERT_DIR := "/home/weaton/pd_examples/certs"
SSL_CA_KEY := SSL_CERT_DIR + "/ca.key"
SSL_CA_CERT := SSL_CERT_DIR + "/ca.crt"
SSL_KEY_FILE := SSL_CERT_DIR + "/server.key"
SSL_CERT_FILE := SSL_CERT_DIR + "/server.crt"

port PORT: 
  @python3 port_allocator.py {{PORT}}

# Generate SSL certificates with CA for mutual trust
generate_ssl_certs:
  @mkdir -p {{SSL_CERT_DIR}}
  @if [ ! -f {{SSL_CA_CERT}} ] || [ ! -f {{SSL_CA_KEY}} ]; then \
    echo "Generating CA certificate..."; \
    openssl req -x509 -new -nodes -keyout {{SSL_CA_KEY}} -sha256 -days 1024 \
      -out {{SSL_CA_CERT}} -subj "/C=US/ST=State/L=City/O=vLLM-CA/CN=vLLM-CA"; \
    chmod 600 {{SSL_CA_KEY}}; \
    chmod 644 {{SSL_CA_CERT}}; \
  fi
  @if [ ! -f {{SSL_KEY_FILE}} ] || [ ! -f {{SSL_CERT_FILE}} ]; then \
    echo "Generating server certificate signed by CA..."; \
    openssl genrsa -out {{SSL_KEY_FILE}} 4096; \
    openssl req -new -key {{SSL_KEY_FILE}} -out {{SSL_CERT_DIR}}/server.csr \
      -subj "/C=US/ST=State/L=City/O=vLLM/CN=localhost"; \
    printf "[v3_req]\nsubjectAltName=DNS:localhost,IP:127.0.0.1\n" > {{SSL_CERT_DIR}}/v3.ext; \
    openssl x509 -req -in {{SSL_CERT_DIR}}/server.csr -CA {{SSL_CA_CERT}} \
      -CAkey {{SSL_CA_KEY}} -CAcreateserial -out {{SSL_CERT_FILE}} \
      -days 365 -sha256 -extensions v3_req -extfile {{SSL_CERT_DIR}}/v3.ext; \
    rm {{SSL_CERT_DIR}}/server.csr {{SSL_CERT_DIR}}/v3.ext; \
    chmod 600 {{SSL_KEY_FILE}}; \
    chmod 644 {{SSL_CERT_FILE}}; \
    echo "SSL certificates generated in {{SSL_CERT_DIR}}"; \
  else \
    echo "SSL certificates already exist in {{SSL_CERT_DIR}}"; \
  fi

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
    @just generate_ssl_certs
    @truncate -s 0 {{vllm-directory}}/prefill.log
    VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5777) \
    UCX_LOG_LEVEL=warn \
    CUDA_VISIBLE_DEVICES={{PREFILL_GPUS}} \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve {{MODEL}} \
      --port $(just port 8100) \
      --tensor-parallel-size 1 \
      --data-parallel-size 1 \
      --gpu-memory-utilization {{MEMORY_UTIL}} \
      --enforce-eager \
      --trust-remote-code \
      --max-model-len 2048 \
      --ssl-keyfile {{SSL_KEY_FILE}} \
      --ssl-certfile {{SSL_CERT_FILE}} \
      --ssl-ca-certs {{SSL_CA_CERT}} \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

decode:
    @just generate_ssl_certs
    @truncate -s 0 {{vllm-directory}}/decode.log
    VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5778) \
    CUDA_VISIBLE_DEVICES={{DECODE_GPUS}} \
    UCX_LOG_LEVEL=debug \
    NCCL_DEBUG=INFO \
    NIXL_LOG_LEVEL=DEBUG \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_LOGGING_CONFIG_PATH=./vllm_decode_debug.log \
    vllm serve {{MODEL}} \
      --port $(just port 8200) \
      --enforce-eager \
      --tensor-parallel-size {{TP_SIZE}} \
      --data-parallel-size {{DP_SIZE}} \
      --gpu-memory-utilization {{MEMORY_UTIL}} \
      --trust-remote-code \
      --max-model-len 2048 \
      --ssl-keyfile {{SSL_KEY_FILE}} \
      --ssl-certfile {{SSL_CERT_FILE}} \
      --ssl-ca-certs {{SSL_CA_CERT}} \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

prefill_podman:
    podman run --rm -it \
      --security-opt=label=disable \
      --cap-add=ALL \
      --user root \
      --network host \
      --device nvidia.com/gpu={{PREFILL_GPUS}} \
      -e HF_TOKEN \
      -e VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5777) \
      -e UCX_LOG_LEVEL=debug \
      -e VLLM_LOGGING_LEVEL="DEBUG" \
      -e HF_HUB_OFFLINE="" \
      -e VLLM_NO_USAGE_STATS=0 \
      -e VLLM_USAGE_STATS_SERVER="http://localhost:8080" \
      -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
      -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      -v /dev/infiniband:/dev/infiniband \
      quay.io/wseaton/vllm:llmd-multistage-6 \
        --model={{MODEL}} \
        --port $(just port 8100) \
        --tensor-parallel-size 1 \
        --data-parallel-size 1 \
        --enforce-eager \
        --gpu-memory-utilization {{MEMORY_UTIL}} \
        --trust-remote-code \
        --max-model-len 2048 \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

smoke:
    podman run --rm -it \
      --cap-add=ALL --network host \
      --user root \
      --device nvidia.com/gpu=3 \
      -e HF_TOKEN=$HF_TOKEN \
      -e HF_HUB_OFFLINE="" \
      -e UCX_LOG_LEVEL=debug \
      -e VLLM_LOGGING_LEVEL="DEBUG" \
      -e VLLM_NO_USAGE_STATS=0 \
      -e VLLM_USAGE_STATS_SERVER="https://vllm-usage-stats-handler-rhaiis-telemetry--runtime-ext.apps.ext.spoke.preprod.us-west-2.aws.paas.redhat.com" \
      -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
      -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      -v /proving-grounds/:/proving-grounds/ \
      registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.0 \
        --model='RedHatAI/Qwen3-8B-FP8-dynamic' \
        --port $(just port 8100) \
        --tensor-parallel-size 1 \
        --data-parallel-size 1 \
        --enforce-eager \
        --gpu-memory-utilization 0.3 \
        --trust-remote-code \
        --max-model-len 2048

decode_podman:
    podman run --rm -it \
      --network=host \
      --security-opt=label=disable \
      --cap-add=ALL \
      --user root \
      --device nvidia.com/gpu=4 \
      -e HF_TOKEN \
      -e VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5778) \
      -e UCX_LOG_LEVEL=debug \
      -e NCCL_DEBUG=INFO \
      -e HF_HUB_OFFLINE="" \
      -e NIXL_LOG_LEVEL=DEBUG \
      -e VLLM_LOGGING_LEVEL="DEBUG" \
      -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
      -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      -v /dev/infiniband:/dev/infiniband \
      quay.io/wseaton/vllm:llmd-multistage-6 \
        --model={{MODEL}} \
        --port $(just port 8200) \
        --tensor-parallel-size 1 \
        --data-parallel-size 1 \
        --gpu-memory-utilization {{MEMORY_UTIL}} \
        --enforce-eager \
        --trust-remote-code \
        --max-model-len 2048 \
        --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}"

proxy:
    VLLM_SERVER_DEV_MODE=1 \
    python3 "{{vllm-directory}}tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
      --port $(just port 8192) \
      --prefiller-port $(just port 8100) \
      --decoder-port $(just port 8200) \
      --enable-ssl \
      --ssl-ca-certs {{SSL_CA_CERT}}


send_request:
  curl -X POST http://localhost:$(just port 8192)/v1/completions \
    -H "Content-Type: application/json" \
    -k \
    -d '{ \
      "model": "{{MODEL}}", \
      "prompt": "Red Hat is the best open source company by far across Linux, K8s, and AI, and vLLM has the greatest community in open source AI software infrastructure. Prefill-decode disaggregation will enable vLLM to ", \
      "max_tokens": 150, \
      "temperature": 0.7 \
    }'

send_request_direct:
  curl -X POST http://localhost:$(just port 8100)/v1/completions \
    -H "Content-Type: application/json" \
    -d '{ \
      "model": "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic", \
      "prompt": "Red Hat is the best open source company by far across Linux, K8s, and AI, and vLLM has the greatest community in open source AI software infrastructure. Prefill-decode disaggregation will enable vLLM to ", \
      "max_tokens": 150, \
      "temperature": 0.7 \
    }'

benchmark:
  python {{vllm-directory}}/benchmarks/benchmark_serving.py --port $(just port 8192) --model {{MODEL}} --dataset-name random --random-input-len 1000 --random-output-len 100


guide_smoke: 
  guidellm benchmark --target http://localhost:$(just port 8100) --model RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 --output-path output_llama3.json --data prompt_tokens=512,prompt_tokens_stdev=128,prompt_tokens_min=1,prompt_tokens_max=1024,output_tokens=256,output_tokens_stdev=64,output_tokens_min=1,output_tokens_max=1024 --rate-type sweep --max-seconds 400 --warmup-percent 0.2

lm_eval_smoke:
  HF_TOKEN=$HF_TOKEN llm-eval-test run \
    --endpoint http://localhost:$(just port 8100)/v1/completions \
    --model meta-llama/Meta-Llama-3-8B-Instruct  \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --datasets ./datasets \
    --tasks gsm8k_cot \
    --output gsm8k_results.json

lm_eval_smoke2:
  python -m lm_eval --model local-completions --model_args model=meta-llama/Meta-Llama-3-8B-Instruct,base_url=http://localhost:$(just port 8100)/v1/completions,max_length=4096 --tasks gsm8k_cot --num_fewshot 8 
lm_eval_smoke3:
  python -m lm_eval --model local-completions --model_args model=meta-llama/Meta-Llama-3-8B-Instruct,base_url=http://localhost:$(just port 8100)/v1/completions,max_length=4096 --tasks winogrande --num_fewshot 5 --batch_size auto 


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

run_dev_servers:
  ./run_dev_servers.sh
