#!/bin/bash
#SBATCH --job-name=ner-compare
#SBATCH --partition=a2000-48h
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=1
#SBATCH --mem=11G
#SBATCH --time=24:00:00

PROJECT_DIR=$1

if [ -z "$PROJECT_DIR" ]; then
    echo "Usage: sbatch run_compare.sh <project_dir>"
    exit 1
fi

source "${PROJECT_DIR}/venv/bin/activate"
cd "${PROJECT_DIR}"

echo "=== Comparison Report Started: $(date) ==="

python3 main.py compare

echo "=== Comparison Report Completed: $(date) ==="
echo "Results saved to: ${PROJECT_DIR}/results/"
