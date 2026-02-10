#!/bin/bash
#SBATCH --partition=a2000-48h
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=11G
#SBATCH --time=48:00:00

MODEL_NAME=$1
PROJECT_DIR=$2
DATASET=${3:-inlegalner}

if [ -z "$MODEL_NAME" ] || [ -z "$PROJECT_DIR" ]; then
    echo "Usage: sbatch run_bert.sh <model_name> <project_dir> [dataset]"
    exit 1
fi

source "${PROJECT_DIR}/venv/bin/activate"
cd "${PROJECT_DIR}"

echo "=== ${MODEL_NAME} (${DATASET}) Started: $(date) ==="
echo "Project dir: ${PROJECT_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python3 main.py train --model "${MODEL_NAME}" --dataset "${DATASET}"
python3 main.py evaluate-bert --model "${MODEL_NAME}" --dataset "${DATASET}"

echo "=== ${MODEL_NAME} (${DATASET}) Completed: $(date) ==="
