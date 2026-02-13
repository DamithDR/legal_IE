#!/bin/bash
###############################################################################
# Submit BERT RE training jobs via SLURM (IE4Wills dataset).
# All 3 models run in parallel, then comparison report runs after all complete.
#
# Usage: cd /path/to/legal_IE && bash slurm_run_re.sh
###############################################################################

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/log"

mkdir -p "${LOG_DIR}"

echo "=== IE4Wills BERT RE Jobs — Submitting ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Timestamp: $(date)"

# Verify dataset exists
if [ ! -f "${PROJECT_DIR}/data/datasets/ie4wills/train.json" ]; then
    echo "ERROR: data/datasets/ie4wills/train.json not found!"
    exit 1
fi

# --- 3 BERT RE models in parallel ---
RE_BERT1_JOB=$(sbatch --parsable \
    --job-name=re-bert-base \
    --output="${LOG_DIR}/re-bert-base_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert_re.sh" bert-base-uncased "${PROJECT_DIR}")
echo "Submitted RE/bert-base-uncased: ${RE_BERT1_JOB}"

RE_BERT2_JOB=$(sbatch --parsable \
    --job-name=re-legal-bert \
    --output="${LOG_DIR}/re-legal-bert_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert_re.sh" legal-bert "${PROJECT_DIR}")
echo "Submitted RE/legal-bert: ${RE_BERT2_JOB}"

RE_BERT3_JOB=$(sbatch --parsable \
    --job-name=re-inlegal-bert \
    --output="${LOG_DIR}/re-inlegal-bert_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert_re.sh" InLegalBERT "${PROJECT_DIR}")
echo "Submitted RE/InLegalBERT: ${RE_BERT3_JOB}"

# --- Comparison Report (waits for all 3 RE jobs) ---
ALL_JOBS="${RE_BERT1_JOB}:${RE_BERT2_JOB}:${RE_BERT3_JOB}"

COMPARE_JOB=$(sbatch --parsable \
    --dependency=afterany:${ALL_JOBS} \
    --output="${LOG_DIR}/re-compare_%A.log" \
    "${PROJECT_DIR}/scripts/run_compare.sh" "${PROJECT_DIR}")

echo ""
echo "=== All jobs submitted ==="
echo "  RE models:"
echo "    bert-base-uncased: ${RE_BERT1_JOB}"
echo "    legal-bert:        ${RE_BERT2_JOB}"
echo "    InLegalBERT:       ${RE_BERT3_JOB}"
echo "  Compare:             ${COMPARE_JOB} (after all 3)"
echo ""
echo "LLM RE evaluation — run locally:"
echo "  python main.py prepare-re-samples"
echo "  python main.py evaluate-llm-re --provider all"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      ${LOG_DIR}/"
