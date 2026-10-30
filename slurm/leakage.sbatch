#!/bin/bash
# §6.2 leakage probe: serve one model, ask it the probe questions (no tools), score them.
# Output: results/leakage/<served name>/{questions,answers,llm_log}.jsonl, summary.csv, meta.json
#
# Submit from FEP (MODEL=dev is the default; the 70B needs 3 GPUs, asked for on the command line):
#   cd $AIHF_WS/AI_HedgeFund && mkdir -p logs
#   sbatch slurm/leakage.sbatch                                        # Llama-3.1-8B, 1 GPU
#   sbatch --gres=gpu:3 --export=ALL,MODEL=70b slurm/leakage.sbatch    # Llama-3.3-70B, 3 GPUs
# (at most 3 GPUs per job for the student account: the 70B runs pipeline parallel, see its yaml)
# Add --partition=dgxh100 to use the H100s, --exclude=<node> to avoid a broken node.
#
#SBATCH --job-name=aihf-leakage
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00              # always set it: the default is 4 h
#SBATCH --output=logs/%x-%j.out      # logs/ must already exist
set -euo pipefail
# Do NOT set CUDA_VISIBLE_DEVICES: SLURM assigns the GPUs requested with --gres.

: "${AIHF_WS:?set AIHF_WS in ~/.bashrc}"
: "${AIHF_LAB:?set AIHF_LAB in ~/.bashrc}"
: "${AIHF_STORE:?set AIHF_STORE in ~/.bashrc}"
: "${AIHF_DATA:?set AIHF_DATA in ~/.bashrc}"
cd "$AIHF_WS/AI_HedgeFund"

MODEL=${MODEL:-dev}
case "$MODEL" in
  dev)
    MODEL_CONFIG=configs/models/llama31_8b_dev.yaml
    MODEL_DIR=$AIHF_LAB/.cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
    SERVED_NAME=llama-3.1-8b; TP=1; PP=1; WAIT_MIN=10 ;;
  70b)
    MODEL_CONFIG=configs/models/llama33_70b_main.yaml
    MODEL_DIR=$AIHF_LAB/.cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b
    SERVED_NAME=llama-3.3-70b; TP=1; PP=3; WAIT_MIN=40 ;;   # 140 GB of weights from network storage
  *) echo "MODEL must be dev or 70b"; exit 1 ;;
esac

GPUS=$(nvidia-smi -L | wc -l)
if [ "$GPUS" -ne $(( TP * PP )) ]; then
  echo "STOP: $MODEL needs $(( TP * PP )) GPU(s) but this job has $GPUS. For 70b submit with --gres=gpu:3."
  exit 1
fi

SERVE_SIF=$AIHF_LAB/containers/vllm_v4.sif           # lab image, read-only use
AGENTS_SIF=$AIHF_STORE/containers/python312.sif
AGENTS_ENV=$AIHF_STORE/envs/agents
export HF_HOME=$AIHF_STORE/hf                         # never the lab space
export HF_HUB_OFFLINE=1                               # load only the pinned snapshot, never download
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
VLLM_LOG=logs/vllm-${SLURM_JOB_ID}.log

export AIHF_GIT_COMMIT=$(git rev-parse HEAD)
echo "== job $SLURM_JOB_ID on $(hostname) ($SLURM_JOB_PARTITION), commit ${AIHF_GIT_COMMIT:0:7}, $(date -Is)"
echo "== model $MODEL: $SERVED_NAME, tensor parallel $TP, pipeline parallel $PP"
nvidia-smi -L

echo "== sha256 of the serving image (15 GB, takes under a minute)"
sha256sum "$SERVE_SIF" | tee "logs/image-${SLURM_JOB_ID}.sha256"
ACTUAL=$(cut -d' ' -f1 "logs/image-${SLURM_JOB_ID}.sha256")
EXPECTED=$(awk '/^serving:/{s=1} s && /sha256:/{print $2; exit}' containers/images.yaml)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "STOP: the serving image changed (expected $EXPECTED). Check with Stefan before trusting any result."
  exit 1
fi

echo "== the questions (dry run)"
apptainer exec "$AGENTS_SIF" "$AGENTS_ENV/bin/python" -m aihf.scripts.leakage_probe --model "$MODEL_CONFIG" --dry-run

echo "== starting vLLM on port $PORT (log: $VLLM_LOG)"
START=$(date +%s)
# --structured-outputs-config: no whitespace between JSON tokens in forced forms. It is a server
# setting (vLLM's engines ignore the per-request field; job 289033), and needs a named backend.
apptainer exec --nv "$SERVE_SIF" vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --dtype bfloat16 \
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  --enable-auto-tool-choice --tool-call-parser llama3_json \
  --chat-template configs/templates/tool_chat_template_llama3.1_json.jinja \
  --seed 0 --host 127.0.0.1 --port "$PORT" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true' EXIT

echo "== waiting for vLLM to be ready (up to $WAIT_MIN min)"
ready=0
for i in $(seq 1 $(( WAIT_MIN * 6 ))); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
    READY_AT=$(date +%s); echo "vLLM ready after $(( READY_AT - START )) s"; ready=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM stopped while starting. Last lines of its log:"; tail -50 "$VLLM_LOG"; exit 1
  fi
  sleep 10
done
if [ "$ready" -ne 1 ]; then
  echo "vLLM not ready after $WAIT_MIN min. Last lines of its log:"; tail -50 "$VLLM_LOG"; exit 1
fi

echo "== asking the questions"
status=0
apptainer exec "$AGENTS_SIF" "$AGENTS_ENV/bin/python" -m aihf.scripts.leakage_probe \
  --model "$MODEL_CONFIG" --backend-url "http://127.0.0.1:$PORT/v1" || status=$?

echo "== measurements for planning (vLLM log)"
echo "load time: $(( READY_AT - START )) s"
grep -E "KV cache size|Maximum concurrency|Loading weights took|Model loading took" "$VLLM_LOG" | sed 's/^.*INFO/INFO/' | head -6 || true
grep "generation throughput" "$VLLM_LOG" | tail -3 | sed 's/^.*Avg prompt/Avg prompt/' || true
echo "== done, exit code $status"
exit "$status"
