#!/usr/bin/env bash
#SBATCH --job-name=internvl_q2_p1json
#SBATCH --partition=h24gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=logs/internvl_query2_from_p1json_%A_%a.out
#SBATCH --account=OD-237777

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-deepfake}"
conda activate "${CONDA_ENV_NAME}"

if [[ -n "${SCRIPT_DIR_OVERRIDE:-}" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_internvl_query2_from_p1json.sbatch" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_PATH="${SLURM_JOB_SCRIPT:-${BASH_SOURCE[0]:-$0}}"
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/InternVL3-78B/}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_ID%/}")}"

META_CSV="${META_CSV:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/SFT_2turn/stage1_gt_with_transcript.csv}"
IMAGE_FOLDER="${IMAGE_FOLDER:-}"
P1_JSON_ROOT="${P1_JSON_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_strongVLM/InternVL3-78B_test/}"
IMG_STEM_CONTAINS="${IMG_STEM_CONTAINS:-_LA_D_}"
SYSTEM_FILE="${SYSTEM_FILE:-${SCRIPT_DIR}/query2_prompts/query2_system.txt}"
USER_TEMPLATE_FILE="${USER_TEMPLATE_FILE:-${SCRIPT_DIR}/query2_prompts/query2_user.txt}"

OUTPUT_BASE="${OUTPUT_BASE:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/ZS_Q2_P1JSON}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${MODEL_TAG}_test}"

NUM_SHARDS="${NUM_SHARDS:-8}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID}}"
else
  SHARD_ID="${SHARD_ID:-0}"
fi
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/internvl_query2_outputs_shard_${SHARD_ID}.jsonl}"

DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-600}"
MAX_NUM="${MAX_NUM:-12}"
INPUT_SIZE="${INPUT_SIZE:-448}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${OUTPUT_DIR}"

echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] SHARD_ID=${SHARD_ID}/${NUM_SHARDS}"
echo "[run] P1_JSON_ROOT=${P1_JSON_ROOT}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run] OUTPUT_JSONL=${OUTPUT_JSONL}"

CMD=(
  python -u "${SCRIPT_DIR}/internvl_query2_artifact_prompt_from_p1json.py"
  --model-id "${MODEL_ID}"
  --meta-csv "${META_CSV}"
  --image-folder "${IMAGE_FOLDER}"
  --prompt1-json-root "${P1_JSON_ROOT}"
  --img-stem-contains "${IMG_STEM_CONTAINS}"
  --system-file "${SYSTEM_FILE}"
  --user-template-file "${USER_TEMPLATE_FILE}"
  --output-dir "${OUTPUT_DIR}"
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
if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

CMD+=("$@")
printf '[run] %q ' "${CMD[@]}"
echo

"${CMD[@]}"

