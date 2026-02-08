#!/bin/bash
#SBATCH --job-name=ner-compare
#SBATCH --partition=a2000-48h
#SBATCH --output=log/compare_%A.log
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=1
#SBATCH --mem=11G
#SBATCH --time=24:00:00

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
source "${PROJECT_DIR}/venv/bin/activate"
cd "${PROJECT_DIR}"

echo "=== Comparison Report Started: $(date) ==="

python3 main.py compare

echo "=== Comparison Report Completed: $(date) ==="
echo "Results saved to: ${PROJECT_DIR}/results/"
