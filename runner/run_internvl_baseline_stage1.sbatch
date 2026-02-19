#!/usr/bin/env bash
#SBATCH --job-name=internvl78b_baseline
#SBATCH --partition=h24gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=logs/internvl78b_baseline_%A_%a.out
#SBATCH --account=OD-237777

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-deepfake}"
conda activate "${CONDA_ENV_NAME}"

if [[ -n "${SCRIPT_DIR_OVERRIDE:-}" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_PATH="${SLURM_JOB_SCRIPT:-${BASH_SOURCE[0]:-$0}}"
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
fi

REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/.." && pwd))"
cd "${REPO_ROOT}"
mkdir -p logs

MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/InternVL3-78B/}"
META_JSON="${META_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_val.json}"
META_CSV="${META_CSV:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_gt.csv}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid}"
SYSTEM_FILE="${SYSTEM_FILE:-${REPO_ROOT}/baseline_prompts/baseline_system.txt}"
USER_TEMPLATE_FILE="${USER_TEMPLATE_FILE:-${REPO_ROOT}/baseline_prompts/baseline_user.txt}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_strongVLM/InternVL3-78B_test}"
NUM_SHARDS="${NUM_SHARDS:-8}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID}}"
else
  SHARD_ID="${SHARD_ID:-0}"
fi
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_ROOT}/internvl_outputs_shard_${SHARD_ID}.jsonl}"

DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
MAX_NUM="${MAX_NUM:-12}"
INPUT_SIZE="${INPUT_SIZE:-448}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

mkdir -p "${OUTPUT_ROOT}"

CMD=(
  python runner/internvl_baseline_runner.py
  --model-id "${MODEL_ID}"
  --meta-json "${META_JSON}"
  --meta-csv "${META_CSV}"
  --image-folder "${IMAGE_FOLDER}"
  --system-file "${SYSTEM_FILE}"
  --user-template-file "${USER_TEMPLATE_FILE}"
  --output-dir "${OUTPUT_ROOT}"
  --output-jsonl "${OUTPUT_JSONL}"
  --num-shards "${NUM_SHARDS}"
  --shard-id "${SHARD_ID}"
  --dtype "${DTYPE}"
  --device-map "${DEVICE_MAP}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-num "${MAX_NUM}"
  --input-size "${INPUT_SIZE}"
)

if [[ "${USE_FLASH_ATTN}" == "1" ]]; then
  CMD+=(--use-flash-attn)
fi
if [[ "${DO_SAMPLE}" == "1" ]]; then
  CMD+=(--do-sample --temperature "${TEMPERATURE}" --top-p "${TOP_P}")
fi

echo "[run] SHARD_ID=${SHARD_ID}/${NUM_SHARDS}"
echo "[run] OUTPUT_JSONL=${OUTPUT_JSONL}"
printf '[run] %q ' "${CMD[@]}"
echo

"${CMD[@]}"
