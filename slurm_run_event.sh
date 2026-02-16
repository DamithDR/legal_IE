#!/bin/bash
###############################################################################
# Submit BERT Event Detection training jobs via SLURM (Events Matter dataset).
# All 3 models run in parallel, then comparison report runs after all complete.
#
# Usage: cd /path/to/legal_IE && bash slurm_run_event.sh
###############################################################################

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/log"

mkdir -p "${LOG_DIR}"

echo "=== Events Matter BERT Event Detection Jobs — Submitting ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Timestamp: $(date)"

# Verify dataset exists
if [ ! -d "${PROJECT_DIR}/data/datasets/events_matter/EventMattersCorpus/annotated" ]; then
    echo "ERROR: data/datasets/events_matter/EventMattersCorpus/annotated not found!"
    exit 1
fi

# --- 3 BERT Event Detection models in parallel ---
EV_BERT1_JOB=$(sbatch --parsable \
    --job-name=event-bert-base \
    --output="${LOG_DIR}/event-bert-base_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert_event.sh" bert-base-uncased "${PROJECT_DIR}")
echo "Submitted Event/bert-base-uncased: ${EV_BERT1_JOB}"

EV_BERT2_JOB=$(sbatch --parsable \
    --job-name=event-legal-bert \
    --output="${LOG_DIR}/event-legal-bert_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert_event.sh" legal-bert "${PROJECT_DIR}")
echo "Submitted Event/legal-bert: ${EV_BERT2_JOB}"

EV_BERT3_JOB=$(sbatch --parsable \
    --job-name=event-inlegal-bert \
    --output="${LOG_DIR}/event-inlegal-bert_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert_event.sh" InLegalBERT "${PROJECT_DIR}")
echo "Submitted Event/InLegalBERT: ${EV_BERT3_JOB}"

# --- Comparison Report (waits for all 3 Event jobs) ---
ALL_JOBS="${EV_BERT1_JOB}:${EV_BERT2_JOB}:${EV_BERT3_JOB}"

COMPARE_JOB=$(sbatch --parsable \
    --dependency=afterany:${ALL_JOBS} \
    --output="${LOG_DIR}/event-compare_%A.log" \
    "${PROJECT_DIR}/scripts/run_compare.sh" "${PROJECT_DIR}")

echo ""
echo "=== All jobs submitted ==="
echo "  Event Detection models:"
echo "    bert-base-uncased: ${EV_BERT1_JOB}"
echo "    legal-bert:        ${EV_BERT2_JOB}"
echo "    InLegalBERT:       ${EV_BERT3_JOB}"
echo "  Compare:             ${COMPARE_JOB} (after all 3)"
echo ""
echo "LLM Event evaluation — run locally:"
echo "  python main.py prepare-event-samples"
echo "  python main.py evaluate-llm-event --provider all"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      ${LOG_DIR}/"
