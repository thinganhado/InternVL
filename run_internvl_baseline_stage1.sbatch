#!/bin/bash -l
#SBATCH --job-name=internvl_baseline
#SBATCH --partition=h24gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/test/logs/internvl_baseline_%A_%a.out
#SBATCH --error=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/test/logs/internvl_baseline_%A_%a.err
#SBATCH --account=OD-237777
#SBATCH --gres=gpu:4
#SBATCH --exclusive

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

conda activate vllm
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

if [[ -n "${SCRIPT_DIR_OVERRIDE:-}" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_internvl_baseline_stage1.sbatch" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -n "${SLURM_JOB_SCRIPT:-}" && -f "${SLURM_JOB_SCRIPT}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$(realpath "${SLURM_JOB_SCRIPT}")")" && pwd)"
elif [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
else
  SCRIPT_DIR="$(pwd)"
fi

mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

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

SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DTYPE="${DTYPE:-auto}"
BACKEND="${BACKEND:-vllm}"
OVERWRITE="${OVERWRITE:-0}"
SAMPLE_ID_GLOB="${SAMPLE_ID_GLOB:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-768}"
VLLM_MAX_IMAGES_PER_PROMPT="${VLLM_MAX_IMAGES_PER_PROMPT:-1}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"

_model_basename="$(basename "${MODEL_ID%/}")"
MODEL_TAG="${MODEL_TAG:-$(echo "${_model_basename}" | tr -cs 'A-Za-z0-9._-' '_')}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-manual}}"
RUN_OUTPUT_ROOT="${OUTPUT_BASE_DIR%/}/${MODEL_TAG}/${RUN_TAG}"

if [[ "${CACHE_BASE_DIR}" == /home/* || "${XDG_CACHE_HOME}" == /home/* || "${FLASHINFER_WORKSPACE_DIR}" == /home/* ]]; then
  echo "[error] Cache path points to /home and may hit quota."
  echo "[error] CACHE_BASE_DIR=${CACHE_BASE_DIR}"
  echo "[error] XDG_CACHE_HOME=${XDG_CACHE_HOME}"
  echo "[error] FLASHINFER_WORKSPACE_DIR=${FLASHINFER_WORKSPACE_DIR}"
  echo "[error] Set cache paths to /datasets or /scratch and rerun."
  exit 2
fi

if [[ "${SHARD_COUNT}" -gt 1 ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-${RUN_OUTPUT_ROOT}/shard_${SHARD_ID}_of_${SHARD_COUNT}}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-${RUN_OUTPUT_ROOT}}"
fi
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR%/}/internvl_baseline_outputs.jsonl}"

mkdir -p "${FLASHINFER_WORKSPACE_DIR}" "${VLLM_CONFIG_ROOT}" "${TRITON_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"
mkdir -p "${OUTPUT_BASE_DIR%/}/logs"
export FLASHINFER_WORKSPACE_BASE FLASHINFER_WORKSPACE_DIR FLASHINFER_JIT_CACHE_DIR XDG_CACHE_HOME VLLM_CONFIG_ROOT TRITON_CACHE_DIR HF_HOME TORCH_HOME TMPDIR

echo "[run] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] BACKEND=${BACKEND}"
echo "[run] MODEL_TAG=${MODEL_TAG}"
echo "[run] RUN_TAG=${RUN_TAG}"
echo "[run] META_JSON=${META_JSON}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run] SHARD_ID=${SHARD_ID} SHARD_COUNT=${SHARD_COUNT}"
echo "[run] SAMPLE_ID_GLOB=${SAMPLE_ID_GLOB}"
echo "[run] OVERWRITE=${OVERWRITE}"
echo "[run] DTYPE=${DTYPE}"
echo "[run] CACHE_BASE_DIR=${CACHE_BASE_DIR}"

python - <<'PY'
import os
import sys
try:
    import flashinfer.jit.env as env
except Exception as e:
    print(f"[warn] flashinfer precheck import failed: {e}")
    sys.exit(0)
resolved = str(getattr(env, "FLASHINFER_WORKSPACE_DIR", ""))
print(f"[run] flashinfer.resolved_workspace={resolved}")
if resolved.startswith("/home/"):
    print("[error] FlashInfer resolved workspace points to /home; will likely fail with quota.")
    sys.exit(3)
PY

CMD=(python -u "${SCRIPT_DIR}/internvl_baseline_artifact_prompt.py" \
  --backend "${BACKEND}" \
  --model-id "${MODEL_ID}" \
  --meta-json "${META_JSON}" \
  --image-folder "${IMAGE_FOLDER}" \
  --system-file "${SYSTEM_FILE}" \
  --user-template-file "${USER_TEMPLATE_FILE}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-shards "${SHARD_COUNT}" \
  --shard-id "${SHARD_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm-max-images-per-prompt "${VLLM_MAX_IMAGES_PER_PROMPT}")

if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
  CMD+=(--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}")
fi

if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
  CMD+=(--vllm-max-model-len "${VLLM_MAX_MODEL_LEN}")
fi

if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
  CMD+=(--vllm-enforce-eager)
fi

if [[ -n "${SAMPLE_ID_GLOB}" ]]; then
  CMD+=(--sample-id-glob "${SAMPLE_ID_GLOB}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

CMD+=("$@")
echo "[run] ${CMD[*]}"
"${CMD[@]}"
