#!/bin/bash
###############################################################################
# Submit BERT NER training jobs one by one via SLURM.
# Each model runs sequentially (next job waits for previous to finish).
#
# Usage: bash slurm_run_all.sh
###############################################################################

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/log"

mkdir -p "${LOG_DIR}"

echo "=== Legal NER Pipeline — Submitting BERT jobs ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Timestamp: $(date)"

# --- BERT Model 1: bert-base-uncased ---
BERT1_JOB=$(sbatch --parsable "${PROJECT_DIR}/scripts/run_bert.sh" bert-base-uncased)
echo "Submitted bert-base-uncased: ${BERT1_JOB}"

# --- BERT Model 2: legal-bert (waits for Model 1) ---
BERT2_JOB=$(sbatch --parsable --dependency=afterok:${BERT1_JOB} "${PROJECT_DIR}/scripts/run_bert.sh" legal-bert)
echo "Submitted legal-bert: ${BERT2_JOB} (after ${BERT1_JOB})"

# --- BERT Model 3: InLegalBERT (waits for Model 2) ---
BERT3_JOB=$(sbatch --parsable --dependency=afterok:${BERT2_JOB} "${PROJECT_DIR}/scripts/run_bert.sh" InLegalBERT)
echo "Submitted InLegalBERT: ${BERT3_JOB} (after ${BERT2_JOB})"

# --- Comparison Report (waits for all) ---
COMPARE_JOB=$(sbatch --parsable --dependency=afterok:${BERT3_JOB} "${PROJECT_DIR}/scripts/run_compare.sh")
echo "Submitted compare: ${COMPARE_JOB} (after ${BERT3_JOB})"

echo ""
echo "=== All jobs submitted (sequential chain) ==="
echo "  1. bert-base-uncased: ${BERT1_JOB}"
echo "  2. legal-bert:        ${BERT2_JOB}"
echo "  3. InLegalBERT:       ${BERT3_JOB}"
echo "  4. Compare:           ${COMPARE_JOB}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      ${LOG_DIR}/"
