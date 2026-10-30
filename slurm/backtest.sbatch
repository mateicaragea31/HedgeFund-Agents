#!/bin/bash
# §6.3 backtest: serve the experiment's model, run one repeat of the experiment in waves.
# Model path, name and GPU layout come from the yaml files (aihf.scripts.model_env).
# Output: results/<run_id>/ (see src/aihf/scripts/run_backtest.py). Exit 3 = deadline: submit again.
#
# Submit from FEP (EXPERIMENT is required; REPEAT defaults to 0; DATES limits a test run;
# MAX_WORKERS (default 15) and CELL_TIMEOUT (minutes, default 60) tune the runner):
#   cd $AIHF_WS/AI_HedgeFund && mkdir -p logs
#   sbatch --export=ALL,EXPERIMENT=configs/experiments/e1_llama70b.yaml,REPEAT=0 slurm/backtest.sbatch
#   sbatch --export=ALL,EXPERIMENT=configs/experiments/e1_llama70b.yaml,DATES=10 slurm/backtest.sbatch
#   sbatch --gres=gpu:1 --export=ALL,EXPERIMENT=configs/experiments/e0_dev.yaml slurm/backtest.sbatch
# Add --partition=dgxh100 for the H100s, --exclude=dgxa100-ncit-wn01 on the A100s (broken node).
#
#SBATCH --job-name=aihf-backtest
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:3                 # the student account allows at most 3 GPUs per job
#SBATCH --cpus-per-task=32           # up to 15 cell processes + vLLM
#SBATCH --mem=192G
#SBATCH --time=0-12                  # the maximum; the default would be only 4 h
#SBATCH --output=logs/%x-%j.out      # logs/ must already exist
set -euo pipefail
# Do NOT set CUDA_VISIBLE_DEVICES: SLURM assigns the GPUs requested with --gres.

: "${AIHF_WS:?set AIHF_WS in ~/.bashrc}"
: "${AIHF_LAB:?set AIHF_LAB in ~/.bashrc}"
: "${AIHF_STORE:?set AIHF_STORE in ~/.bashrc}"
: "${AIHF_DATA:?set AIHF_DATA in ~/.bashrc}"
: "${EXPERIMENT:?pass EXPERIMENT=configs/experiments/<name>.yaml with --export}"
cd "$AIHF_WS/AI_HedgeFund"
REPEAT=${REPEAT:-0}

SERVE_SIF=$AIHF_LAB/containers/vllm_v4.sif           # lab image, read-only use
AGENTS_SIF=$AIHF_STORE/containers/python312.sif
AGENTS_PY=$AIHF_STORE/envs/agents/bin/python
export HF_HOME=$AIHF_STORE/hf                         # never the lab space
export HF_HUB_OFFLINE=1                               # load only the pinned snapshot, never download
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
VLLM_LOG=logs/vllm-${SLURM_JOB_ID}.log

eval "$(apptainer exec "$AGENTS_SIF" "$AGENTS_PY" -m aihf.scripts.model_env "$EXPERIMENT")"
GOT=$(nvidia-smi -L | wc -l)
if [ "$GOT" -ne "$GPUS" ]; then
  echo "STOP: $SERVED_NAME needs $GPUS GPU(s) (TP $TP x PP $PP) but this job has $GOT: submit with --gres=gpu:$GPUS"
  exit 2
fi

export AIHF_GIT_COMMIT=$(git rev-parse HEAD)
if [ -z "$(git status --porcelain --untracked-files=no)" ]; then export AIHF_GIT_CLEAN=1; else export AIHF_GIT_CLEAN=0; fi
echo "== job $SLURM_JOB_ID on $(hostname) ($SLURM_JOB_PARTITION), commit ${AIHF_GIT_COMMIT:0:7} (clean=$AIHF_GIT_CLEAN), $(date -Is)"
echo "== $EXPERIMENT repeat $REPEAT${DATES:+, first $DATES dates}; model $SERVED_NAME, TP $TP x PP $PP"
nvidia-smi -L

echo "== sha256 of the serving image"
export AIHF_SERVE_IMAGE_SHA=$(sha256sum "$SERVE_SIF" | cut -d' ' -f1)
EXPECTED=$(awk '/^serving:/{s=1} s && /sha256:/{print $2; exit}' containers/images.yaml)
if [ "$AIHF_SERVE_IMAGE_SHA" != "$EXPECTED" ]; then
  echo "STOP: the serving image changed (expected $EXPECTED). Check with Stefan before trusting any result."
  exit 2
fi

echo "== the plan (dry run)"
apptainer exec "$AGENTS_SIF" "$AGENTS_PY" -m aihf.scripts.run_backtest --experiment "$EXPERIMENT" \
  --repeat "$REPEAT" ${DATES:+--dates "$DATES"} --dry-run

echo "== starting vLLM on port $PORT (log: $VLLM_LOG)"
START=$(date +%s)
# --structured-outputs-config: no whitespace between JSON tokens in forced forms. It is a server
# setting (vLLM's engines ignore the per-request field; job 289033), and needs a named backend.
apptainer exec --nv "$SERVE_SIF" vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --dtype bfloat16 \
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  --enable-auto-tool-choice --tool-call-parser llama3_json \
  --chat-template "$CHAT_TEMPLATE" \
  --seed 0 --host 127.0.0.1 --port "$PORT" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true' EXIT

ready=0
for i in $(seq 1 240); do                            # up to 40 min (a 70B from network storage)
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "vLLM ready after $(( $(date +%s) - START )) s"; ready=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM stopped while starting. Last lines of its log:"; tail -50 "$VLLM_LOG"; exit 1
  fi
  sleep 10
done
[ "$ready" -eq 1 ] || { echo "vLLM not ready after 40 min"; tail -50 "$VLLM_LOG"; exit 1; }
grep -E "KV cache size|Maximum concurrency" "$VLLM_LOG" | sed 's/^.*INFO/INFO/' || true

# 12 h job: stop starting cells 30 min before the end (vLLM start-up already used a few minutes).
DEADLINE=$(( 690 - ( $(date +%s) - START ) / 60 ))
echo "== running the backtest (deadline in $DEADLINE min)"
status=0
apptainer exec "$AGENTS_SIF" "$AGENTS_PY" -m aihf.scripts.run_backtest --experiment "$EXPERIMENT" \
  --repeat "$REPEAT" ${DATES:+--dates "$DATES"} --backend-url "http://127.0.0.1:$PORT/v1" \
  --deadline-minutes "$DEADLINE" --max-workers "${MAX_WORKERS:-15}" \
  --cell-timeout-minutes "${CELL_TIMEOUT:-60}" || status=$?

echo "== vLLM throughput (last lines)"
grep "generation throughput" "$VLLM_LOG" | tail -5 | sed 's/^.*Avg prompt/Avg prompt/' || true
echo "== done, exit code $status (3 = deadline: submit again to continue)"
exit "$status"
