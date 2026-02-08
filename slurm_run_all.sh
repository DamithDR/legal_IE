#!/bin/bash
###############################################################################
# Submit BERT NER training jobs via SLURM.
# All 3 models run in parallel on separate GPUs.
#
# Usage: cd /path/to/legal_IE && bash slurm_run_all.sh
###############################################################################

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/log"

mkdir -p "${LOG_DIR}"

echo "=== Legal NER Pipeline — Submitting BERT jobs ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Timestamp: $(date)"

# --- Submit all 3 BERT models in parallel ---
BERT1_JOB=$(sbatch --parsable \
    --job-name=ner-bert-base \
    --output="${LOG_DIR}/bert-base_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert.sh" bert-base-uncased "${PROJECT_DIR}")
echo "Submitted bert-base-uncased: ${BERT1_JOB}"

BERT2_JOB=$(sbatch --parsable \
    --job-name=ner-legal-bert \
    --output="${LOG_DIR}/legal-bert_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert.sh" legal-bert "${PROJECT_DIR}")
echo "Submitted legal-bert: ${BERT2_JOB}"

BERT3_JOB=$(sbatch --parsable \
    --job-name=ner-inlegal-bert \
    --output="${LOG_DIR}/inlegal-bert_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert.sh" InLegalBERT "${PROJECT_DIR}")
echo "Submitted InLegalBERT: ${BERT3_JOB}"

# --- Comparison Report (waits for all 3 to finish) ---
COMPARE_JOB=$(sbatch --parsable \
    --dependency=afterok:${BERT1_JOB}:${BERT2_JOB}:${BERT3_JOB} \
    --output="${LOG_DIR}/compare_%A.log" \
    "${PROJECT_DIR}/scripts/run_compare.sh" "${PROJECT_DIR}")
echo "Submitted compare: ${COMPARE_JOB}"

echo ""
echo "=== All jobs submitted (parallel) ==="
echo "  bert-base-uncased: ${BERT1_JOB}"
echo "  legal-bert:        ${BERT2_JOB}"
echo "  InLegalBERT:       ${BERT3_JOB}"
echo "  Compare:           ${COMPARE_JOB} (after all 3)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      ${LOG_DIR}/"
