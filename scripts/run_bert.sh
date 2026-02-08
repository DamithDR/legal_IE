#!/bin/bash
#SBATCH --partition=a2000-48h
#SBATCH --output=log/%x_%A.log
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=11G
#SBATCH --time=24:00:00

MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: sbatch run_bert.sh <model_name>"
    echo "  Models: bert-base-uncased, legal-bert, InLegalBERT"
    exit 1
fi

#SBATCH --job-name=ner-${MODEL_NAME}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
source "${PROJECT_DIR}/venv/bin/activate"
cd "${PROJECT_DIR}"

echo "=== ${MODEL_NAME} Started: $(date) ==="
echo "Project dir: ${PROJECT_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python3 main.py train --model "${MODEL_NAME}"
python3 main.py evaluate-bert --model "${MODEL_NAME}"

echo "=== ${MODEL_NAME} Completed: $(date) ==="
