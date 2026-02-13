#!/bin/bash
###############################################################################
# Legal NER Pipeline — All available commands
# Reference for local execution. Not meant to be run as a single script.
###############################################################################

# =============================================================================
# 1. Dataset Statistics
# =============================================================================

python main.py stats --dataset inlegalner
python main.py stats --dataset icdac
python main.py stats --dataset ie4wills

# =============================================================================
# 2. BERT Fine-tuning (requires GPU)
# =============================================================================

# Single model, single dataset
python main.py train --model bert-base-uncased --dataset inlegalner
python main.py train --model legal-bert --dataset inlegalner
python main.py train --model InLegalBERT --dataset inlegalner

python main.py train --model bert-base-uncased --dataset icdac
python main.py train --model legal-bert --dataset icdac
python main.py train --model InLegalBERT --dataset icdac

python main.py train --model bert-base-uncased --dataset ie4wills
python main.py train --model legal-bert --dataset ie4wills
python main.py train --model InLegalBERT --dataset ie4wills

# All models on one dataset
python main.py train --model all --dataset inlegalner
python main.py train --model all --dataset icdac
python main.py train --model all --dataset ie4wills

# =============================================================================
# 3. BERT Evaluation
# =============================================================================

# Single model, single dataset
python main.py evaluate-bert --model bert-base-uncased --dataset inlegalner
python main.py evaluate-bert --model legal-bert --dataset inlegalner
python main.py evaluate-bert --model InLegalBERT --dataset inlegalner

python main.py evaluate-bert --model bert-base-uncased --dataset icdac
python main.py evaluate-bert --model legal-bert --dataset icdac
python main.py evaluate-bert --model InLegalBERT --dataset icdac

python main.py evaluate-bert --model bert-base-uncased --dataset ie4wills
python main.py evaluate-bert --model legal-bert --dataset ie4wills
python main.py evaluate-bert --model InLegalBERT --dataset ie4wills

# All models on one dataset
python main.py evaluate-bert --model all --dataset inlegalner
python main.py evaluate-bert --model all --dataset icdac
python main.py evaluate-bert --model all --dataset ie4wills

# From a specific checkpoint
python main.py evaluate-bert --model bert-base-uncased --dataset inlegalner --checkpoint path/to/checkpoint

# =============================================================================
# 4. LLM Sample Preparation (cache test sentences + few-shot examples)
# =============================================================================

# Single dataset
python main.py prepare-samples --dataset inlegalner
python main.py prepare-samples --dataset icdac
python main.py prepare-samples --dataset ie4wills

# All datasets at once
python main.py prepare-samples --dataset all

# Regenerate cache
python main.py prepare-samples --dataset all --resample

# Use full test set instead of sampling
python main.py prepare-samples --dataset inlegalner --full-test

# Custom sample size
python main.py prepare-samples --dataset inlegalner --sample-size 100

# =============================================================================
# 5. LLM Evaluation (requires API keys in .env, uses cached samples)
# =============================================================================

# Single provider, single dataset
python main.py evaluate-llm --provider openai --dataset inlegalner
python main.py evaluate-llm --provider deepseek --dataset inlegalner
python main.py evaluate-llm --provider claude --dataset inlegalner

python main.py evaluate-llm --provider openai --dataset icdac
python main.py evaluate-llm --provider deepseek --dataset icdac
python main.py evaluate-llm --provider claude --dataset icdac

python main.py evaluate-llm --provider openai --dataset ie4wills
python main.py evaluate-llm --provider deepseek --dataset ie4wills
python main.py evaluate-llm --provider claude --dataset ie4wills

# All providers, single dataset
python main.py evaluate-llm --provider all --dataset inlegalner
python main.py evaluate-llm --provider all --dataset icdac
python main.py evaluate-llm --provider all --dataset ie4wills

# All providers, all datasets
python main.py evaluate-llm --provider all --dataset all

# =============================================================================
# 6. Comparison & Export
# =============================================================================

# Generate comparison report from all saved results
python main.py compare

# Export LaTeX tables (all datasets)
python main.py export-latex

# Export LaTeX for specific datasets
python main.py export-latex --datasets inlegalner,icdac
python main.py export-latex --datasets ie4wills

# Custom output filename
python main.py export-latex --output my_tables.tex

# =============================================================================
# 7. Full Pipeline (train + evaluate BERT + prepare samples + LLM eval + compare)
# =============================================================================

python main.py run-all --dataset inlegalner
python main.py run-all --dataset icdac
python main.py run-all --dataset ie4wills

# =============================================================================
# 8. SLURM (submit BERT jobs to cluster, LLM eval is local)
# =============================================================================

# Submit all 9 BERT jobs + comparison (from project root)
bash slurm_run_all.sh

# Monitor jobs
squeue -u $USER

# Check logs
ls log/
