#!/bin/bash
###############################################################################
# Submit Legal-BERT Contract Event Detection job via SLURM.
#
# Usage: cd /path/to/legal_IE && bash slurm_run_contract_event.sh
###############################################################################

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/log"

mkdir -p "${LOG_DIR}"

echo "=== Contractual Events — Legal-BERT — Submitting ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Timestamp: $(date)"

if [ ! -d "${PROJECT_DIR}/data/datasets/contractual_events/annotator-5" ]; then
    echo "ERROR: data/datasets/contractual_events/annotator-5 not found!"
    exit 1
fi

JOB_ID=$(sbatch --parsable \
    --job-name=cevent-legal-bert \
    --output="${LOG_DIR}/cevent-legal-bert_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert_contract_event.sh" "${PROJECT_DIR}")

echo "Submitted: ${JOB_ID}"
echo "Monitor with: squeue -u \$USER"
echo "Log: ${LOG_DIR}/cevent-legal-bert_${JOB_ID}.log"
