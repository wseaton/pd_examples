# Setting this allows creating a symlink to Justfile from another dir
set working-directory := "/home/wseaton/pd_examples/"

# Needed for the proxy server
vllm-directory := "/home/wseaton/vllm/"

# Default container image for podman recipes
CONTAINER_IMAGE := "localhost/test:latest"

# MODEL := "meta-llama/Llama-1.1-8B-Instruct"
TP_SIZE := "1"
DP_SIZE := "1"

# export UCX_TLS := "^dc"
# export CUDA_DEVICE_ORDER := "PCI_BUS_ID"
# export UCX_TLS := "tcp"
# DECODE_GPUS := "4"
#
export VLLM_SERVER_DEV_MODE := "1"
export VLLM_ATTENTION_BACKEND := "FLASH_ATTN"
export VLLM_NIXL_HANDSHAKE_METHOD := "zmq"
export VLLM_USE_V1 := "1"
export UCX_FAULT_DEBUG := "1"


# MODEL := "deepseek-ai/DeepSeek-V2-Lite"
# MODEL := "RedHatAI/Mistral-Large-Instruct-2407-FP8"
MODEL := "RedHatAI/Llama-3.1-8B-Instruct"
# TP_SIZE := "2"
PREFILL_GPUS := "2"
DECODE_GPUS := "3"

# MEMORY_UTIL := "0.3"

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
      --trust-remote-code

prefill:
    HF_HOME=/home/wseaton/.cache/hf \
    HF_HUB_CACHE=/home/wseaton/.cache/hf \
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
      --enforce-eager \
      --trust-remote-code \
      --max-model-len 2048 \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_retry_policy":"abort"}'

decode:
    @rm -f /home/wseaton/pd_examples/decode.log
    HF_HOME=/home/wseaton/.cache/hf \
    HF_HUB_CACHE=/home/wseaton/.cache/hf \
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
      --enforce-eager \
      --tensor-parallel-size {{TP_SIZE}} \
      --data-parallel-size {{DP_SIZE}} \
      --trust-remote-code \
      --max-model-len 2048 \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_retry_policy":"abort"}' \
      2>&1 | tee /home/wseaton/pd_examples/decode.log

decode_with_faults:
    @rm -f /home/wseaton/pd_examples/decode.log
    HF_HOME=/home/wseaton/.cache/hf \
    HF_HUB_CACHE=/home/wseaton/.cache/hf \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5778) \
    CUDA_VISIBLE_DEVICES={{DECODE_GPUS}} \
    UCX_LOG_LEVEL=debug \
    UCX_FAULT_DEBUG=1 \
    NCCL_DEBUG=INFO \
    NIXL_LOG_LEVEL=DEBUG \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    UCX_FAULT_DEBUG=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    RUST_LOG=debug \
    LD_PRELOAD=/home/wseaton/ucx-fault-injector/target/release/libucx_fault_injector.so \
    vllm serve {{MODEL}} \
      --port $(just port 8200) \
      --enforce-eager \
      --tensor-parallel-size {{TP_SIZE}} \
      --data-parallel-size {{DP_SIZE}} \
      --trust-remote-code \
      --max-model-len 2048 \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_retry_policy":"abort"}' \
      2>&1 | tee /home/wseaton/pd_examples/decode.log

prefill_podman image=CONTAINER_IMAGE:
    @rm -f /home/wseaton/pd_examples/memray_output/profile.bin
    podman run --rm -it \
      --privileged \
      --security-opt=label=disable \
      --cap-add=ALL \
      --user root \
      --network host \
      --device nvidia.com/gpu=all \
      --shm-size=8g \
      -v /dev/infiniband:/dev/infiniband \
      -v /home/wseaton/pd_examples/memray_output:/memray_output \
      -v /home/wseaton/.cache/hf:/root/.cache/hf:Z \
      -e HF_TOKEN \
      -e "HF_HOME=/root/.cache/hf" \
      -e CUDA_VISIBLE_DEVICES={{PREFILL_GPUS}} \
      -e VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5777) \
      -e UCX_LOG_LEVEL=warn \
      -e VLLM_LOGGING_LEVEL="DEBUG" \
      -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
      -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      -e HF_HUB_OFFLINE="" \
      --entrypoint="" \
      {{image}} \
      python -m vllm.entrypoints.openai.api_server \
        --model={{MODEL}} \
        --port $(just port 8100) \
        --tensor-parallel-size 1 \
        --data-parallel-size 1 \
        --enforce-eager \
        --trust-remote-code \
        --max-model-len 2048 \
        --disable-log-requests \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_retry_policy":"abort"}'

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

decode_podman image=CONTAINER_IMAGE:
    @rm -f /home/wseaton/pd_examples/memray_output/decode_profile.bin
    podman run --rm -it \
      --network=host \
      --security-opt=label=disable \
      --cap-add=ALL \
      --user root \
      --device nvidia.com/gpu=all \
      -v /dev/infiniband:/dev/infiniband \
      --shm-size=8g \
      -v /home/wseaton/pd_examples/memray_output:/memray_output \
      -v /home/wseaton/.cache/hf:/root/.cache/hf:Z \
      -e HF_TOKEN \
      -e "HF_HOME=/root/.cache/hf" \
      -e CUDA_VISIBLE_DEVICES={{DECODE_GPUS}} \
      -e VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5778) \
      -e UCX_LOG_LEVEL=debug \
      -e NCCL_DEBUG=INFO \
      -e HF_HUB_OFFLINE="" \
      -e NIXL_LOG_LEVEL=DEBUG \
      -e VLLM_LOGGING_LEVEL="DEBUG" \
      -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
      -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      --entrypoint="" \
      {{image}} \
      python -m vllm.entrypoints.openai.api_server \
        --model={{MODEL}} \
        --port $(just port 8200) \
        --enforce-eager \
        --tensor-parallel-size {{TP_SIZE}} \
        --data-parallel-size {{DP_SIZE}} \
        --trust-remote-code \
        --max-model-len 2048 \
        --disable-log-requests \
        --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_load_retry_policy\":\"abort\"}"

proxy:
    VLLM_SERVER_DEV_MODE=1 \
    python3 "{{vllm-directory}}tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
      --port $(just port 8192) \
      --prefiller-port $(just port 8100) \
      --decoder-port $(just port 8200)


send_request:
  curl -i -X POST http://localhost:$(just port 8192)/v1/completions \
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
  HF_HOME=/home/wseaton/.cache/hf HF_HUB_CACHE="" vllm bench serve --port $(just port 8192) --model {{MODEL}} --dataset-name random --random-input-len 10240 --random-output-len 1280 --max-concurrency 8


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
  #!/bin/bash
  # Try lsof first, fallback to ss if not available
  if command -v lsof &> /dev/null; then
    lsof -i:$(just port 8200) | awk 'NR > 1 {print $2}' | xargs kill -9 2>/dev/null || true
    lsof -i:$(just port 8100) | awk 'NR > 1 {print $2}' | xargs kill -9 2>/dev/null || true
    lsof -i:$(just port 8192) | awk 'NR > 1 {print $2}' | xargs kill -9 2>/dev/null || true
  else
    ss -lptn 2>/dev/null | grep -E ":$(just port 8200)\s" | grep -oP 'pid=\K[0-9]+' | xargs kill -9 2>/dev/null || true
    ss -lptn 2>/dev/null | grep -E ":$(just port 8100)\s" | grep -oP 'pid=\K[0-9]+' | xargs kill -9 2>/dev/null || true
    ss -lptn 2>/dev/null | grep -E ":$(just port 8192)\s" | grep -oP 'pid=\K[0-9]+' | xargs kill -9 2>/dev/null || true
  fi


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
   #!/bin/bash
   nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do \
     if ps -o user= -p "$pid" 2>/dev/null | grep -q "^$(whoami)$"; then \
       echo "Killing GPU process $pid"; \
       kill -9 "$pid"; \
     fi; \
   done

# Build ucx-fault-injector client if needed
build_fault_injector:
    #!/bin/bash
    if [ ! -f /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client ]; then
        echo "Building ucx-fault-injector..."
        cd /home/wseaton/ucx-fault-injector
        cargo build --release
    else
        echo "ucx-fault-client already built"
    fi

# Show fault injection help and current status
fault_help:
    #!/bin/bash
    echo "UCX Fault Injection Commands (using ZMQ-based control):"
    echo ""
    echo "Basic control:"
    echo "  just toggle_faults          - Toggle fault injection on/off"
    echo "  just fault_status           - Show current fault injection status"
    echo "  just reset_faults           - Reset to defaults (disabled)"
    echo ""
    echo "Quick presets:"
    echo "  just set_10_percent_faults  - Enable 10% NETWORK_ERROR faults"
    echo "  just set_100_percent_faults - Enable 100% TIMEOUT faults"
    echo ""
    echo "Manual control:"
    echo "  just set_scenario_0         - Set to NETWORK_ERROR faults"
    echo "  just set_scenario_1         - Set to TIMEOUT faults"
    echo "  just set_scenario_2         - Set to MEMORY_ERROR faults"
    echo "  just enable_faults 1 50     - Enable scenario 1 at 50% rate"
    echo ""
    echo "Current status:"
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client status

# Set fault injection to 100% rate using ZMQ client
set_100_percent_faults:
    #!/bin/bash
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client toggle
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client scenario 1
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client probability 100
    echo "Fault injection set to 100% rate, scenario 1 (TIMEOUT)"

# Set fault injection to 10% rate using ZMQ client
set_10_percent_faults:
    #!/bin/bash
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client toggle
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client scenario 0
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client probability 10
    echo "Fault injection set to 10% rate, scenario 0 (NETWORK_ERROR)"

# Toggle fault injection on/off
toggle_faults:
    #!/bin/bash
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client toggle
    echo "Fault injection toggled"

# Reset fault injection settings
reset_faults:
    #!/bin/bash
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client reset
    echo "Fault injection RESET (disabled)"

# Set specific fault scenarios
set_scenario_0:
    #!/bin/bash
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client scenario 0
    echo "Set fault scenario to 0 (NETWORK_ERROR)"

set_scenario_1:
    #!/bin/bash
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client scenario 1
    echo "Set fault scenario to 1 (TIMEOUT)"

set_scenario_2:
    #!/bin/bash
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client scenario 2
    echo "Set fault scenario to 2 (MEMORY_ERROR)"

# Show current fault injection status
fault_status:
    #!/bin/bash
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client status
    echo "Note: Status is broadcast to all fault injector instances"

# Enable faults with specific scenario and probability
enable_faults scenario="0" probability="10":
    #!/bin/bash
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client toggle
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client scenario {{scenario}}
    /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client probability {{probability}}
    echo "Enabled fault injection: scenario {{scenario}}, probability {{probability}}%"


dump_record:
    echo '{"command": "dump_recording", "export_format": "records", "value": 50}' | /home/wseaton/ucx-fault-injector/target/release/ucx-fault-client 

generate_flamegraph:
    podman run --rm -it \
      -v /home/wseaton/pd_examples/memray_output:/memray_output \
      --user root \
      --entrypoint=/bin/bash \
      localhost/test:latest \
      -c "memray flamegraph /memray_output/profile.bin -o /memray_output/flamegraph.html"

run_dev_servers:
  ./run_dev_servers.sh

nixl_selftest image=CONTAINER_IMAGE:
    #!/bin/bash
    echo "Starting NIXL self test between two {{image}} containers..."
    
    # Get available port for the test
    TEST_PORT=$(just port 5555)
    
    # Start target container in background
    echo "Starting target container on port $TEST_PORT..."
    podman run --rm -d \
      --name nixl-target \
      --network=host \
      --security-opt=label=disable \
      --cap-add=ALL \
      --user root \
      --device nvidia.com/gpu=all \
      -v /dev/infiniband:/dev/infiniband \
      -v {{justfile_directory()}}/nixl_test.py:/nixl_test.py:ro \
      -e UCX_LOG_LEVEL=debug \
      -e NIXL_LOG_LEVEL=DEBUG \
      --entrypoint="" \
      {{image}} \
      python /nixl_test.py --ip 127.0.0.1 --port $TEST_PORT --mode target --use_cuda true
    
    # Wait a moment for target to start
    sleep 3
    
    # Run initiator container and wait for completion
    echo "Starting initiator container..."
    podman run --rm \
      --name nixl-initiator \
      --network=host \
      --security-opt=label=disable \
      --cap-add=ALL \
      --user root \
      --device nvidia.com/gpu=all \
      -v /dev/infiniband:/dev/infiniband \
      -v {{justfile_directory()}}/nixl_test.py:/nixl_test.py:ro \
      -e UCX_LOG_LEVEL=debug \
      -e NIXL_LOG_LEVEL=DEBUG \
      --entrypoint="" \
      {{image}} \
      python /nixl_test.py --ip 127.0.0.1 --port $TEST_PORT --mode initiator --use_cuda true
    
    # Clean up target container
    echo "Cleaning up target container..."
    podman stop nixl-target || true
    
    echo "NIXL self test completed."

deepgemm_selftest image=CONTAINER_IMAGE:
    #!/bin/bash
    echo "Starting DeepGEMM self test with {{image}} container..."
    
    # Run DeepGEMM tests in container
    echo "Running DeepGEMM tests..."
    podman run --rm \
      --name deepgemm-test \
      --network=host \
      --security-opt=label=disable \
      --cap-add=ALL \
      --user root \
      --device nvidia.com/gpu=all \
      -v {{justfile_directory()}}/deepgemm_test.py:/deepgemm_test.py:ro \
      -v {{justfile_directory()}}/DeepGEMM:/opt/deepgemm:ro \
      --entrypoint="" \
      {{image}} \
      python /deepgemm_test.py --test all --gpu 0
    
    echo "DeepGEMM self test completed."

deepgemm_simple_selftest image=CONTAINER_IMAGE:
    #!/bin/bash
    echo "Starting DeepGEMM simple self test with {{image}} container..."
    
    # Run simplified DeepGEMM tests in container
    echo "Running DeepGEMM simple tests..."
    podman run --rm \
      --name deepgemm-simple-test \
      --network=host \
      --security-opt=label=disable \
      --cap-add=ALL \
      --user root \
      --device nvidia.com/gpu=all \
      -v {{justfile_directory()}}/deepgemm_simple_test.py:/deepgemm_simple_test.py:ro \
      --entrypoint="" \
      {{image}} \
      python /deepgemm_simple_test.py --gpu 0
    
    echo "DeepGEMM simple self test completed."

deepgemm_minimal_selftest image=CONTAINER_IMAGE:
    #!/bin/bash
    echo "Starting DeepGEMM minimal self test with {{image}} container..."
    
    # Run minimal DeepGEMM availability tests in container
    echo "Running DeepGEMM minimal tests..."
    podman run --rm \
      --name deepgemm-minimal-test \
      --network=host \
      --security-opt=label=disable \
      --cap-add=ALL \
      --user root \
      --device nvidia.com/gpu=all \
      -v {{justfile_directory()}}/deepgemm_minimal_test.py:/deepgemm_minimal_test.py:ro \
      --entrypoint="" \
      {{image}} \
      python /deepgemm_minimal_test.py --gpu 0
    
    echo "DeepGEMM minimal self test completed."

pplx_kernels_selftest image=CONTAINER_IMAGE:
    #!/bin/bash
    echo "Starting pplx-kernels all-to-all benchmark with {{image}} container..."
    
    # Run pplx-kernels all-to-all benchmark in container
    echo "Running pplx-kernels all-to-all benchmark..."
    podman run --rm \
      --name pplx-kernels-test \
      --network=host \
      --security-opt=label=disable \
      --cap-add=ALL \
      --user root \
      --device nvidia.com/gpu=all \
      -v /dev/infiniband:/dev/infiniband \
      --shm-size=8g \
      -v {{justfile_directory()}}/pplx-kernels:/opt/pplx-kernels \
      --entrypoint="" \
      {{image}} \
      bash -c "cd /opt/pplx-kernels && pip install pytest && python -m tests.bench_all_to_all --dp-size 1"
    
    echo "pplx-kernels all-to-all benchmark completed."

deepep_intranode_test image=CONTAINER_IMAGE:
    #!/bin/bash
    echo "Starting DeepEP intranode (single node) test with {{image}} container..."
    
    # Run DeepEP intranode test in container
    echo "Running DeepEP intranode test..."
    podman run --rm \
      --name deepep-intranode-test \
      --network=host \
      --security-opt=label=disable \
      --cap-add=ALL \
      --user root \
      --device nvidia.com/gpu=all \
      -v /dev/infiniband:/dev/infiniband \
      --shm-size=8g \
      -v {{justfile_directory()}}/DeepEP:/opt/DeepEP \
      -e MASTER_ADDR=127.0.0.1 \
      -e MASTER_PORT=29500 \
      -e WORLD_SIZE=1 \
      -e RANK=0 \
      --entrypoint="" \
      {{image}} \
      bash -c "cd /opt/DeepEP && python tests/test_intranode.py --num-processes 4 --num-tokens 1024 --hidden 2048 --num-topk 4 --num-experts 64"
    
    echo "DeepEP intranode test completed."

deepep_low_latency_test image=CONTAINER_IMAGE:
    #!/bin/bash
    echo "Starting DeepEP low latency test with {{image}} container..."
    
    # Run DeepEP low latency test in container
    echo "Running DeepEP low latency test..."
    podman run --rm \
      --name deepep-low-latency-test \
      --network=host \
      --security-opt=label=disable \
      --cap-add=ALL \
      --user root \
      --device nvidia.com/gpu=all \
      -v /dev/infiniband:/dev/infiniband \
      --shm-size=8g \
      -v {{justfile_directory()}}/DeepEP:/opt/DeepEP \
      -e MASTER_ADDR=127.0.0.1 \
      -e MASTER_PORT=29501 \
      -e WORLD_SIZE=1 \
      -e RANK=0 \
      --entrypoint="" \
      {{image}} \
      bash -c "cd /opt/DeepEP && python tests/test_low_latency.py --num-processes 2 --num-tokens 512 --hidden 1024 --num-topk 2 --num-experts 32"
    
    echo "DeepEP low latency test completed."

shell:
    podman run --rm -it \
      --network=host \
      --security-opt=label=disable \
      --cap-add=ALL \
      --user root \
      --device nvidia.com/gpu=4 \
      --entrypoint=/bin/bash \
      -e HF_TOKEN \
      -e VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5778) \
      -e UCX_LOG_LEVEL=debug \
      -e NCCL_DEBUG=INFO \
      -e HF_HUB_OFFLINE="" \
      -e NIXL_LOG_LEVEL=DEBUG \
      -e VLLM_LOGGING_LEVEL="DEBUG" \
      -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
      -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      localhost/test:latest

zmq_hostname_test:
    python3 zmq_test_script.py


