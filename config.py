import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)


def save_results(name: str, data: dict) -> Path:
    """
    Save results with a timestamped copy and a latest copy.

    Saves to:
        results/{name}_results.json          (latest, used by comparison)
        results/{name}_results_{timestamp}.json  (archived run)

    Returns:
        Path to the latest results file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data["timestamp"] = timestamp

    # Save timestamped archive
    archive_path = RESULTS_DIR / f"{name}_results_{timestamp}.json"
    with open(archive_path, "w") as f:
        json.dump(data, f, indent=2)

    # Save latest (for comparison module)
    latest_path = RESULTS_DIR / f"{name}_results.json"
    with open(latest_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {latest_path}")
    print(f"Archived to {archive_path}")
    return latest_path

# --- Dataset ---
DATASET_NAME = "opennyaiorg/InLegalNER"

# --- ICDAC Dataset ---
ICDAC_EXCEL_PATH = PROJECT_ROOT / "data" / "datasets" / "Indian_Court_Decision_Annotated_Corpus_xlsx.xlsx"

# --- IE4Wills Dataset ---
IE4WILLS_DATA_DIR = PROJECT_ROOT / "data" / "datasets" / "ie4wills"
ICDAC_SPLIT_SEED = 42
ICDAC_SPLIT_RATIOS = (0.8, 0.1, 0.1)

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
    "warmup_steps": 500,
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

# --- RE Configuration ---
RE_TRAIN_CONFIG = {
    **TRAIN_CONFIG,
    "metric_for_best_model": "eval_f1_macro",
    "num_train_epochs": 10,
}
RE_NEGATIVE_RATIO = 1.0   # ratio of negative to positive examples for BERT RE training
RE_FEW_SHOT_COUNT = 3     # fewer than NER because RE examples are larger in prompt
