#!/bin/bash
###############################################################################
# Submit IE4Wills BERT NER jobs via SLURM (re-run after dataset fix).
# All 3 models run in parallel, then comparison report runs after all complete.
#
# Usage: cd /path/to/legal_IE && bash slurm_run_ie4wills.sh
###############################################################################

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/log"

mkdir -p "${LOG_DIR}"

echo "=== IE4Wills BERT Jobs — Submitting ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Timestamp: $(date)"

# Verify dataset exists
if [ ! -f "${PROJECT_DIR}/data/datasets/ie4wills/train.json" ]; then
    echo "ERROR: data/datasets/ie4wills/train.json not found!"
    echo "Download the dataset first (see README)."
    exit 1
fi

# --- IE4Wills jobs (3 models in parallel) ---
echo ""
echo "--- IE4Wills ---"

IE4_BERT1_JOB=$(sbatch --parsable \
    --job-name=ner-ie4-bert-base \
    --output="${LOG_DIR}/ie4-bert-base_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert.sh" bert-base-uncased "${PROJECT_DIR}" ie4wills)
echo "Submitted ie4wills/bert-base-uncased: ${IE4_BERT1_JOB}"

IE4_BERT2_JOB=$(sbatch --parsable \
    --job-name=ner-ie4-legal-bert \
    --output="${LOG_DIR}/ie4-legal-bert_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert.sh" legal-bert "${PROJECT_DIR}" ie4wills)
echo "Submitted ie4wills/legal-bert: ${IE4_BERT2_JOB}"

IE4_BERT3_JOB=$(sbatch --parsable \
    --job-name=ner-ie4-inlegal-bert \
    --output="${LOG_DIR}/ie4-inlegal-bert_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert.sh" InLegalBERT "${PROJECT_DIR}" ie4wills)
echo "Submitted ie4wills/InLegalBERT: ${IE4_BERT3_JOB}"

# --- Comparison Report (waits for all 3 IE4Wills jobs) ---
ALL_JOBS="${IE4_BERT1_JOB}:${IE4_BERT2_JOB}:${IE4_BERT3_JOB}"

COMPARE_JOB=$(sbatch --parsable \
    --dependency=afterany:${ALL_JOBS} \
    --output="${LOG_DIR}/compare_%A.log" \
    "${PROJECT_DIR}/scripts/run_compare.sh" "${PROJECT_DIR}")

echo ""
echo "=== All jobs submitted ==="
echo "  IE4Wills:"
echo "    bert-base-uncased: ${IE4_BERT1_JOB}"
echo "    legal-bert:        ${IE4_BERT2_JOB}"
echo "    InLegalBERT:       ${IE4_BERT3_JOB}"
echo "  Compare:             ${COMPARE_JOB} (after all 3)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      ${LOG_DIR}/"
