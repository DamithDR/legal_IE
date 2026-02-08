#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --output=log/output_%A_%a.log
#SBATCH --mail-type=END,FAIL

source venv/bin/activate

python3 -m experiments.pre-experiments.embedding_generator