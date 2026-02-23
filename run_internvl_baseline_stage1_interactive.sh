#!/bin/bash -l
set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate "${CONDA_ENV_NAME:-vllm}"

if [[ -n "${SCRIPT_DIR_OVERRIDE:-}" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_OVERRIDE}"
elif [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
else
  SCRIPT_DIR="$(pwd)"
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/InternVL3-78B/}"
META_JSON="${META_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data/stage1_query1_val_swift.json}"
SYSTEM_FILE="${SYSTEM_FILE:-${SCRIPT_DIR}/baseline_prompts/baseline_system.txt}"
USER_TEMPLATE_FILE="${USER_TEMPLATE_FILE:-${SCRIPT_DIR}/baseline_prompts/baseline_user.txt}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid}"

OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final___baseline__strong__InternVL/}"
CACHE_BASE_DIR="${CACHE_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/test}"
FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${CACHE_BASE_DIR}/flashinfer}"
FLASHINFER_JIT_CACHE_DIR="${FLASHINFER_JIT_CACHE_DIR:-${FLASHINFER_WORKSPACE_DIR}}"
FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${CACHE_BASE_DIR}}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_BASE_DIR}}"
VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${CACHE_BASE_DIR}/vllm}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_BASE_DIR}/triton}"
HF_HOME="${HF_HOME:-${CACHE_BASE_DIR}/huggingface}"
TORCH_HOME="${TORCH_HOME:-${CACHE_BASE_DIR}/torch}"
TMPDIR="${TMPDIR:-/tmp/${USER}/internvl_baseline}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a _GPU_ARR <<< "${GPU_IDS}"
GPU_COUNT="${#_GPU_ARR[@]}"
if (( GPU_COUNT < 1 )); then
  echo "[error] GPU_IDS is empty."
  exit 2
fi
if (( TENSOR_PARALLEL_SIZE < 1 )); then
  echo "[error] TENSOR_PARALLEL_SIZE must be >= 1."
  exit 2
fi
if (( GPU_COUNT % TENSOR_PARALLEL_SIZE != 0 )); then
  echo "[error] GPU count (${GPU_COUNT}) must be divisible by TENSOR_PARALLEL_SIZE (${TENSOR_PARALLEL_SIZE})."
  echo "[hint] Example: GPU_IDS=0,1,2,3 TENSOR_PARALLEL_SIZE=2"
  exit 2
fi
GPU_GROUP_COUNT=$((GPU_COUNT / TENSOR_PARALLEL_SIZE))
SHARD_COUNT="${SHARD_COUNT:-${GPU_GROUP_COUNT}}"
if (( SHARD_COUNT > GPU_GROUP_COUNT )); then
  echo "[error] SHARD_COUNT (${SHARD_COUNT}) exceeds available parallel GPU groups (${GPU_GROUP_COUNT})."
  echo "[hint] Reduce SHARD_COUNT or set smaller TENSOR_PARALLEL_SIZE."
  exit 2
fi
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_IMAGES_PER_PROMPT="${VLLM_MAX_IMAGES_PER_PROMPT:-1}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
DTYPE="${DTYPE:-auto}"
BACKEND="${BACKEND:-vllm}"
OVERWRITE="${OVERWRITE:-1}"
SAMPLE_ID_GLOB="${SAMPLE_ID_GLOB:-}"

_model_basename="$(basename "${MODEL_ID%/}")"
MODEL_TAG="${MODEL_TAG:-$(echo "${_model_basename}" | tr -cs 'A-Za-z0-9._-' '_')}"
RUN_TAG="${RUN_TAG:-manual_$(date +%Y%m%d_%H%M%S)}"
RUN_OUTPUT_ROOT="${OUTPUT_BASE_DIR%/}/${MODEL_TAG}/${RUN_TAG}"

mkdir -p "${FLASHINFER_WORKSPACE_DIR}" "${VLLM_CONFIG_ROOT}" "${TRITON_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"
mkdir -p "${OUTPUT_BASE_DIR%/}/logs"

export FLASHINFER_WORKSPACE_BASE FLASHINFER_WORKSPACE_DIR FLASHINFER_JIT_CACHE_DIR
export XDG_CACHE_HOME VLLM_CONFIG_ROOT TRITON_CACHE_DIR HF_HOME TORCH_HOME TMPDIR

if [[ "${CACHE_BASE_DIR}" == /home/* || "${XDG_CACHE_HOME}" == /home/* || "${FLASHINFER_WORKSPACE_DIR}" == /home/* ]]; then
  echo "[error] Cache path points to /home and may hit quota."
  exit 2
fi

python - <<'PY'
import sys
try:
    import flashinfer.jit.env as env
except Exception as e:
    print(f"[warn] flashinfer precheck import failed: {e}")
    sys.exit(0)
resolved = str(getattr(env, "FLASHINFER_WORKSPACE_DIR", ""))
print(f"[run] flashinfer.resolved_workspace={resolved}")
if resolved.startswith('/home/'):
    print('[error] FlashInfer resolved workspace points to /home; will likely fail with quota.')
    sys.exit(3)
PY

echo "[run] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] SHARD_COUNT=${SHARD_COUNT}"
echo "[run] TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "[run] GPU_IDS=${GPU_IDS}"
echo "[run] RUN_TAG=${RUN_TAG}"
echo "[run] RUN_OUTPUT_ROOT=${RUN_OUTPUT_ROOT}"

pids=()
shards=()

for ((i=0; i<SHARD_COUNT; i++)); do
  group_start=$((i * TENSOR_PARALLEL_SIZE))
  group_end=$((group_start + TENSOR_PARALLEL_SIZE - 1))
  gpu_list=""
  for ((g=group_start; g<=group_end; g++)); do
    if [[ -z "${gpu_list}" ]]; then
      gpu_list="${_GPU_ARR[$g]}"
    else
      gpu_list="${gpu_list},${_GPU_ARR[$g]}"
    fi
  done

  out_dir="${RUN_OUTPUT_ROOT}/shard_${i}_of_${SHARD_COUNT}"
  out_jsonl="${RUN_OUTPUT_ROOT}/internvl_baseline_outputs_shard_${i}.jsonl"
  log_file="logs/internvl_baseline_${MODEL_TAG}_shard_${i}_${RUN_TAG}.log"

  cmd=(python -u "${SCRIPT_DIR}/internvl_baseline_artifact_prompt.py"
    --backend "${BACKEND}"
    --model-id "${MODEL_ID}"
    --meta-json "${META_JSON}"
    --image-folder "${IMAGE_FOLDER}"
    --system-file "${SYSTEM_FILE}"
    --user-template-file "${USER_TEMPLATE_FILE}"
    --dtype "${DTYPE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --num-shards "${SHARD_COUNT}"
    --shard-id "${i}"
    --output-dir "${out_dir}"
    --output-jsonl "${out_jsonl}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --vllm-max-model-len "${VLLM_MAX_MODEL_LEN}"
    --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm-max-images-per-prompt "${VLLM_MAX_IMAGES_PER_PROMPT}")

  if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
    cmd+=(--vllm-enforce-eager)
  fi
  if [[ -n "${SAMPLE_ID_GLOB}" ]]; then
    cmd+=(--sample-id-glob "${SAMPLE_ID_GLOB}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  echo "[launch] shard=${i} gpus=${gpu_list} log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu_list}" "${cmd[@]}" > "${log_file}" 2>&1 &
  pids+=("$!")
  shards+=("${i}")
done

fail=0
set +e
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  shard="${shards[$idx]}"
  wait "${pid}"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    fail=1
    echo "[fail] shard=${shard} pid=${pid} rc=${rc}"
    tail -n 80 "logs/internvl_baseline_${MODEL_TAG}_shard_${shard}_${RUN_TAG}.log" || true
  else
    echo "[ok] shard=${shard} pid=${pid}"
  fi
done
set -e

if [[ $fail -ne 0 ]]; then
  echo "[done] one or more shards failed."
  exit 1
fi

echo "[done] all shards completed successfully."
