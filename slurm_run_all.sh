#!/bin/bash
###############################################################################
# Submit BERT NER training jobs via SLURM.
# All models run in parallel on separate GPUs, for all datasets.
# LLM evaluation is run locally (not via SLURM).
#
# Usage: cd /path/to/legal_IE && bash slurm_run_all.sh
###############################################################################

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/log"

mkdir -p "${LOG_DIR}"

echo "=== Legal NER Pipeline — Submitting BERT jobs ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Timestamp: $(date)"

# --- InLegalNER jobs ---
echo ""
echo "--- InLegalNER ---"

INL_BERT1_JOB=$(sbatch --parsable \
    --job-name=ner-inl-bert-base \
    --output="${LOG_DIR}/inl-bert-base_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert.sh" bert-base-uncased "${PROJECT_DIR}" inlegalner)
echo "Submitted inlegalner/bert-base-uncased: ${INL_BERT1_JOB}"

INL_BERT2_JOB=$(sbatch --parsable \
    --job-name=ner-inl-legal-bert \
    --output="${LOG_DIR}/inl-legal-bert_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert.sh" legal-bert "${PROJECT_DIR}" inlegalner)
echo "Submitted inlegalner/legal-bert: ${INL_BERT2_JOB}"

INL_BERT3_JOB=$(sbatch --parsable \
    --job-name=ner-inl-inlegal-bert \
    --output="${LOG_DIR}/inl-inlegal-bert_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert.sh" InLegalBERT "${PROJECT_DIR}" inlegalner)
echo "Submitted inlegalner/InLegalBERT: ${INL_BERT3_JOB}"

# --- ICDAC jobs ---
echo ""
echo "--- ICDAC ---"

ICD_BERT1_JOB=$(sbatch --parsable \
    --job-name=ner-icd-bert-base \
    --output="${LOG_DIR}/icd-bert-base_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert.sh" bert-base-uncased "${PROJECT_DIR}" icdac)
echo "Submitted icdac/bert-base-uncased: ${ICD_BERT1_JOB}"

ICD_BERT2_JOB=$(sbatch --parsable \
    --job-name=ner-icd-legal-bert \
    --output="${LOG_DIR}/icd-legal-bert_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert.sh" legal-bert "${PROJECT_DIR}" icdac)
echo "Submitted icdac/legal-bert: ${ICD_BERT2_JOB}"

ICD_BERT3_JOB=$(sbatch --parsable \
    --job-name=ner-icd-inlegal-bert \
    --output="${LOG_DIR}/icd-inlegal-bert_%A.log" \
    "${PROJECT_DIR}/scripts/run_bert.sh" InLegalBERT "${PROJECT_DIR}" icdac)
echo "Submitted icdac/InLegalBERT: ${ICD_BERT3_JOB}"

# --- IE4Wills jobs ---
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

# --- Comparison Report (waits for all 9 BERT jobs to finish) ---
ALL_JOBS="${INL_BERT1_JOB}:${INL_BERT2_JOB}:${INL_BERT3_JOB}:${ICD_BERT1_JOB}:${ICD_BERT2_JOB}:${ICD_BERT3_JOB}:${IE4_BERT1_JOB}:${IE4_BERT2_JOB}:${IE4_BERT3_JOB}"

COMPARE_JOB=$(sbatch --parsable \
    --dependency=afterok:${ALL_JOBS} \
    --output="${LOG_DIR}/compare_%A.log" \
    "${PROJECT_DIR}/scripts/run_compare.sh" "${PROJECT_DIR}")
echo ""
echo "Submitted compare: ${COMPARE_JOB}"

echo ""
echo "=== All jobs submitted (parallel) ==="
echo "  InLegalNER:"
echo "    bert-base-uncased: ${INL_BERT1_JOB}"
echo "    legal-bert:        ${INL_BERT2_JOB}"
echo "    InLegalBERT:       ${INL_BERT3_JOB}"
echo "  ICDAC:"
echo "    bert-base-uncased: ${ICD_BERT1_JOB}"
echo "    legal-bert:        ${ICD_BERT2_JOB}"
echo "    InLegalBERT:       ${ICD_BERT3_JOB}"
echo "  IE4Wills:"
echo "    bert-base-uncased: ${IE4_BERT1_JOB}"
echo "    legal-bert:        ${IE4_BERT2_JOB}"
echo "    InLegalBERT:       ${IE4_BERT3_JOB}"
echo "  Compare:             ${COMPARE_JOB} (after all 9)"
echo ""
echo "LLM evaluation is run locally:"
echo "  python main.py prepare-samples --dataset all"
echo "  python main.py evaluate-llm --dataset all"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      ${LOG_DIR}/"
