import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

# --- Dataset ---
DATASET_NAME = "opennyaiorg/InLegalNER"

# --- BERT Models to fine-tune ---
BERT_MODELS = {
    "bert-base-uncased": "bert-base-uncased",
    "legal-bert": "nlpaueb/legal-bert-base-uncased",
    "InLegalBERT": "law-ai/InLegalBERT",
}

# --- Training Hyperparameters ---
TRAIN_CONFIG = {
    "learning_rate": 2e-5,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_seq_length": 512,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "overall_f1",
    "seed": 42,
}

# --- LLM Configuration ---
LLM_SAMPLE_SIZE = 200
LLM_FEW_SHOT_COUNT = 5
LLM_MAX_RETRIES = 3
LLM_RETRY_WAIT = 2  # seconds
LLM_REQUEST_DELAY = 0.5  # seconds between API calls

# API Keys (from .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# LLM Model IDs
OPENAI_MODEL = "gpt-4o"
DEEPSEEK_MODEL = "deepseek-chat"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
