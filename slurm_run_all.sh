#!/bin/bash
###############################################################################
# SLURM Master Script — Legal NER Evaluation Pipeline
# Submits separate jobs per model: each BERT model gets its own GPU.
# LLM evaluation runs in parallel (CPU-only).
# Comparison runs after all jobs complete.
#
# Usage: sbatch slurm_run_all.sh
###############################################################################

#SBATCH --job-name=legal-ner-master
#SBATCH --partition=a2000-48h
#SBATCH --output=log/master_%A.log
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${PROJECT_DIR}/venv/bin/activate"

mkdir -p "${PROJECT_DIR}/log"

echo "=== Legal NER Pipeline — Submitting jobs ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Timestamp: $(date)"

# --- BERT Model 1: bert-base-uncased (own GPU) ---
BERT1_JOB=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=ner-bert-base
#SBATCH --partition=a2000-48h
#SBATCH --output=${PROJECT_DIR}/log/bert-base_%A.log
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=11G
#SBATCH --time=24:00:00

source "${VENV}"
cd "${PROJECT_DIR}"

echo "=== bert-base-uncased Started: \$(date) ==="
python3 main.py train --model bert-base-uncased
python3 main.py evaluate-bert --model bert-base-uncased
echo "=== bert-base-uncased Completed: \$(date) ==="
EOF
)
echo "Submitted bert-base-uncased: ${BERT1_JOB}"

# --- BERT Model 2: legal-bert (own GPU) ---
BERT2_JOB=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=ner-legal-bert
#SBATCH --partition=a2000-48h
#SBATCH --output=${PROJECT_DIR}/log/legal-bert_%A.log
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=11G
#SBATCH --time=24:00:00

source "${VENV}"
cd "${PROJECT_DIR}"

echo "=== legal-bert Started: \$(date) ==="
python3 main.py train --model legal-bert
python3 main.py evaluate-bert --model legal-bert
echo "=== legal-bert Completed: \$(date) ==="
EOF
)
echo "Submitted legal-bert: ${BERT2_JOB}"

# --- BERT Model 3: InLegalBERT (own GPU) ---
BERT3_JOB=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=ner-inlegal-bert
#SBATCH --partition=a2000-48h
#SBATCH --output=${PROJECT_DIR}/log/inlegal-bert_%A.log
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=11G
#SBATCH --time=24:00:00

source "${VENV}"
cd "${PROJECT_DIR}"

echo "=== InLegalBERT Started: \$(date) ==="
python3 main.py train --model InLegalBERT
python3 main.py evaluate-bert --model InLegalBERT
echo "=== InLegalBERT Completed: \$(date) ==="
EOF
)
echo "Submitted InLegalBERT: ${BERT3_JOB}"

# --- LLM Evaluation (no GPU, API calls only) ---
LLM_JOB=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=ner-llm-eval
#SBATCH --partition=a2000-48h
#SBATCH --output=${PROJECT_DIR}/log/llm_%A.log
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=2
#SBATCH --mem=11G
#SBATCH --time=24:00:00

source "${VENV}"
cd "${PROJECT_DIR}"

echo "=== LLM Evaluation Started: \$(date) ==="
python3 main.py evaluate-llm --provider openai
python3 main.py evaluate-llm --provider deepseek
echo "=== LLM Evaluation Completed: \$(date) ==="
EOF
)
echo "Submitted LLM eval: ${LLM_JOB}"

# --- Comparison Report (runs after ALL jobs complete) ---
sbatch --dependency=afterok:${BERT1_JOB}:${BERT2_JOB}:${BERT3_JOB}:${LLM_JOB} <<EOF
#!/bin/bash
#SBATCH --job-name=ner-compare
#SBATCH --partition=a2000-48h
#SBATCH --output=${PROJECT_DIR}/log/compare_%A.log
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=1
#SBATCH --mem=11G
#SBATCH --time=24:00:00

source "${VENV}"
cd "${PROJECT_DIR}"

echo "=== Comparison Report Started: \$(date) ==="
python3 main.py compare
echo "=== Comparison Report Completed: \$(date) ==="
echo "Results saved to: ${PROJECT_DIR}/results/"
EOF

echo ""
echo "=== All jobs submitted ==="
echo "  bert-base-uncased: ${BERT1_JOB}"
echo "  legal-bert:        ${BERT2_JOB}"
echo "  InLegalBERT:       ${BERT3_JOB}"
echo "  LLM eval:          ${LLM_JOB}"
echo "  Compare:           runs after all complete"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      ${PROJECT_DIR}/log/"
