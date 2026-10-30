#!/bin/bash
# §6.0 "done when": one cell (default AAPL on 2024-01-10) through the whole pipeline on the
# `dev` model (Llama-3.1-8B, 1 GPU). Same start-up as smoke_dev.sbatch, then run_cell.
# Output: results/dev/<TICKER>_<DATE>/{state.json, llm_log.jsonl, report.json}.
#
# Submit from FEP (TICKER, DATE and FORCE are optional):
#   cd $AIHF_WS/AI_HedgeFund && mkdir -p logs && sbatch slurm/cell_dev.sbatch
#   sbatch --export=ALL,TICKER=AAPL,DATE=2024-01-11 slurm/cell_dev.sbatch
#   sbatch --export=ALL,FORCE=1 slurm/cell_dev.sbatch     # run a finished cell again
#
#SBATCH --job-name=aihf-cell-dev
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00              # always set it: the default is 4 h
#SBATCH --output=logs/%x-%j.out      # logs/ must already exist
set -euo pipefail
# Do NOT set CUDA_VISIBLE_DEVICES: SLURM assigns the GPU requested with --gres.

: "${AIHF_WS:?set AIHF_WS in ~/.bashrc}"
: "${AIHF_LAB:?set AIHF_LAB in ~/.bashrc}"
: "${AIHF_STORE:?set AIHF_STORE in ~/.bashrc}"
: "${AIHF_DATA:?set AIHF_DATA in ~/.bashrc}"
cd "$AIHF_WS/AI_HedgeFund"

TICKER=${TICKER:-AAPL}
DATE=${DATE:-2024-01-10}
EXPERIMENT=configs/experiments/e0_dev.yaml
SERVE_SIF=$AIHF_LAB/containers/vllm_v4.sif           # lab image, read-only use
AGENTS_SIF=$AIHF_STORE/containers/python312.sif
AGENTS_ENV=$AIHF_STORE/envs/agents
MODEL_DIR=$AIHF_LAB/.cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
SERVED_NAME=llama-3.1-8b                              # must match `name` in configs/models/llama31_8b_dev.yaml
export HF_HOME=$AIHF_STORE/hf                         # never the lab space
export HF_HUB_OFFLINE=1                               # load only the pinned snapshot, never download
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
VLLM_LOG=logs/vllm-${SLURM_JOB_ID}.log

# The agents container has no git: record the commit and a clean/dirty flag here.
export AIHF_GIT_COMMIT=$(git rev-parse HEAD)
if [ -z "$(git status --porcelain --untracked-files=no)" ]; then export AIHF_GIT_CLEAN=1; else export AIHF_GIT_CLEAN=0; fi

echo "== job $SLURM_JOB_ID on $(hostname), commit ${AIHF_GIT_COMMIT:0:7} (clean=$AIHF_GIT_CLEAN), $(date -Is)"
echo "== cell: $TICKER on $DATE, experiment $EXPERIMENT"
nvidia-smi -L

echo "== sha256 of the serving image (15 GB, takes under a minute)"
sha256sum "$SERVE_SIF" | tee "logs/image-${SLURM_JOB_ID}.sha256"
ACTUAL=$(cut -d' ' -f1 "logs/image-${SLURM_JOB_ID}.sha256")
EXPECTED=$(awk '/^serving:/{s=1} s && /sha256:/{print $2; exit}' containers/images.yaml)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "STOP: the serving image changed (expected $EXPECTED). Check with Stefan before trusting any result."
  exit 1
fi
echo "serving image matches containers/images.yaml"

echo "== checking the config and the data (dry run)"
apptainer exec "$AGENTS_SIF" "$AGENTS_ENV/bin/python" -m aihf.scripts.run_cell \
  --experiment "$EXPERIMENT" --ticker "$TICKER" --date "$DATE" --dry-run

echo "== starting vLLM on port $PORT (log: $VLLM_LOG)"
# --structured-outputs-config: no whitespace between JSON tokens in forced forms. It is a server
# setting (vLLM's engines ignore the per-request field; job 289033), and needs a named backend.
apptainer exec --nv "$SERVE_SIF" vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  --enable-auto-tool-choice --tool-call-parser llama3_json \
  --chat-template configs/templates/tool_chat_template_llama3.1_json.jinja \
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

echo "== running the cell"
status=0
apptainer exec "$AGENTS_SIF" "$AGENTS_ENV/bin/python" -m aihf.scripts.run_cell \
  --experiment "$EXPERIMENT" --ticker "$TICKER" --date "$DATE" \
  --backend-url "http://127.0.0.1:$PORT/v1" ${FORCE:+--force} || status=$?

echo "== last lines of the vLLM log"
tail -20 "$VLLM_LOG"
echo "== done, exit code $status"
exit "$status"
