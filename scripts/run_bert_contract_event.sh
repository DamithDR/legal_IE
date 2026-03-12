#!/bin/bash
#SBATCH --partition=a2000-48h
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=11G
#SBATCH --time=48:00:00

PROJECT_DIR=$1

if [ -z "$PROJECT_DIR" ]; then
    echo "Usage: sbatch run_bert_contract_event.sh <project_dir>"
    exit 1
fi

source "${PROJECT_DIR}/venv/bin/activate"
cd "${PROJECT_DIR}"

echo "=== Contract Event Detection — Legal-BERT Started: $(date) ==="
echo "Project dir: ${PROJECT_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python3 main.py train-contract-event --model legal-bert
python3 main.py evaluate-bert-contract-event --model legal-bert

echo "=== Contract Event Detection — Legal-BERT Completed: $(date) ==="
