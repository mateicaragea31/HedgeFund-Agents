#!/bin/bash
# Phase 0.5: start the `dev` model (Llama-3.1-8B) in the lab's vLLM image on 1 GPU,
# then run the four smoke tests in tests/gpu/ against it. Everything stops at the end.
#
# Submit from FEP:
#   cd $AIHF_WS/AI_HedgeFund && mkdir -p logs && sbatch slurm/smoke_dev.sbatch
#
#SBATCH --job-name=aihf-smoke-dev
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00              # always set it: the default is 4 h
#SBATCH --output=logs/%x-%j.out      # logs/ must already exist
set -euo pipefail
# Do NOT set CUDA_VISIBLE_DEVICES: SLURM assigns the GPU requested with --gres.

: "${AIHF_WS:?set AIHF_WS in ~/.bashrc}"
: "${AIHF_LAB:?set AIHF_LAB in ~/.bashrc}"
: "${AIHF_STORE:?set AIHF_STORE in ~/.bashrc}"
cd "$AIHF_WS/AI_HedgeFund"

SERVE_SIF=$AIHF_LAB/containers/vllm_v4.sif           # lab image, read-only use
AGENTS_SIF=$AIHF_STORE/containers/python312.sif
AGENTS_ENV=$AIHF_STORE/envs/agents
MODEL_CONFIG=configs/models/llama31_8b_dev.yaml
MODEL_DIR=$AIHF_LAB/.cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
SERVED_NAME=llama-3.1-8b                              # must match `name` in $MODEL_CONFIG
export HF_HOME=$AIHF_STORE/hf                         # never the lab space
export HF_HUB_OFFLINE=1                               # load only the pinned snapshot, never download
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
VLLM_LOG=logs/vllm-${SLURM_JOB_ID}.log

echo "== job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), $(date -Is)"
nvidia-smi -L

echo "== sha256 of the serving image (15 GB, takes a minute or two)"
sha256sum "$SERVE_SIF" | tee "logs/image-${SLURM_JOB_ID}.sha256"

echo "== starting vLLM on port $PORT (log: $VLLM_LOG)"
apptainer exec --nv "$SERVE_SIF" vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser llama3_json \
  --chat-template serving/templates/tool_chat_template_llama3.1_json.jinja \
  --seed 0 --host 127.0.0.1 --port "$PORT" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true' EXIT

echo "== waiting for vLLM to be ready (up to 10 min)"
ready=0
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "vLLM ready after about $(( i * 10 )) s"; ready=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM stopped while starting. Last lines of its log:"; tail -50 "$VLLM_LOG"; exit 1
  fi
  sleep 10
done
if [ "$ready" -ne 1 ]; then
  echo "vLLM not ready after 10 min. Last lines of its log:"; tail -50 "$VLLM_LOG"; exit 1
fi

echo "== running the smoke tests"
export AIHF_VLLM_URL="http://127.0.0.1:$PORT/v1" AIHF_MODEL_CONFIG=$MODEL_CONFIG
status=0
apptainer exec "$AGENTS_SIF" "$AGENTS_ENV/bin/pytest" -m gpu -v -s tests/gpu || status=$?

echo "== last lines of the vLLM log"
tail -20 "$VLLM_LOG"
echo "== done, exit code $status"
exit "$status"
