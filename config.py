# config.py
from __future__ import annotations
from pathlib import Path
import os

# ---- Paths
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "Plus14_filtered_bug_report_scores_Summary.xlsx"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
VALIDATION_DATA_PATH = DATA_DIR / "validation_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

# --- [MODIFIED] DPO Data Paths (str로 변경) ---
DPO_CANDIDATES_PATH_TEMPLATE = str(DATA_DIR / "dpo_candidates_{model_name}.csv")
DPO_TRAIN_DATA_PATH_TEMPLATE = str(DATA_DIR / "dpo_train_{model_name}.jsonl")
# ----------------------------

RESULTS_DIR = PROJECT_ROOT / "results"
INDIVIDUAL_DIR = RESULTS_DIR / "individual"
SUMMARY_DIR = RESULTS_DIR / "summary" 
REPORTS_DIR = RESULTS_DIR / "reports"
CACHE_DIR = RESULTS_DIR / "cache"

# ---- Base models (WSL local)
DEFAULT_MODELS = {
    "qwen":   "/mnt/c/Users/selab/LLM_Models/models--unsloth--qwen2.5-7b-instruct-unsloth-bnb-4bit",
    "llama":  "/mnt/c/Users/selab/LLM_Models/models--unsloth--llama-3.2-3b-instruct-unsloth-bnb-4bit",
    "mistral":"/mnt/c/Users/selab/LLM_Models/models--unsloth--mistral-7b-instruct-v0.3-bnb-4bit",
}
# Allow env override: QWEN_MODEL_ID, LLAMA_MODEL_ID, MISTRAL_MODEL_ID
MODEL_ID = {
    "qwen": os.getenv("QWEN_MODEL_ID", DEFAULT_MODELS["qwen"]),
    "llama": os.getenv("LLAMA_MODEL_ID", DEFAULT_MODELS["llama"]),
    "mistral": os.getenv("MISTRAL_MODEL_ID", DEFAULT_MODELS["mistral"]),
}

# ---- [MODIFIED] Adapters (SFT) ----
ADAPTER_SFT_ROOT = Path(os.getenv("ADAPTER_SFT_ROOT", PROJECT_ROOT / "adapters_sft"))
ADAPTER_SFT_DIRS = {
    "qwen": ADAPTER_SFT_ROOT / "qwen2.5-7b",
    "llama": ADAPTER_SFT_ROOT / "llama-3.2-3b",
    "mistral": ADAPTER_SFT_ROOT / "mistral-7b",
}
for p in ADAPTER_SFT_DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

# --- [NEW] Adapters (DPO) ---
ADAPTER_DPO_ROOT = Path(os.getenv("ADAPTER_DPO_ROOT", PROJECT_ROOT / "adapters_dpo"))
ADAPTER_DPO_DIRS = {
    "qwen": ADAPTER_DPO_ROOT / "qwen2.5-7b",
    "llama": ADAPTER_DPO_ROOT / "llama-3.2-3b",
    "mistral": ADAPTER_DPO_ROOT / "mistral-7b",
}
for p in ADAPTER_DPO_DIRS.values():
    p.mkdir(parents=True, exist_ok=True)
# ----------------------------


# ---- SFT Params (기존 Finetuning) ----
# [MODIFIED] Unsloth 관련 파라미터(grad_ckpt) 제거
SFT_PARAMS = {
    "max_epochs": 3,
    "batch_size": 1,
    "grad_accum_steps": 8,
    "max_seq_len": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lr": 2e-4,
    # "grad_ckpt": "unsloth", # <-- 제거됨
}

# --- [NEW] DPO Params ---
# [MODIFIED] Unsloth 관련 파라미터(beta) 제거 (HF DPOTrainer 기본값 사용)
DPO_PARAMS = {
    # 'build_dpo_dataset.py' filters
    "QUALITY_FLOOR": 0.5,
    "NOISE_MARGIN": 0.01,
    
    # 'training_agent.py' DPO trainer params
    "max_epochs": 1,
    "batch_size": 1,
    "grad_accum_steps": 8,
    "lr": 5e-5,
    "beta": 0.1,           # (HF DPOTrainer도 beta를 지원하므로 유지)
    "max_seq_len": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}
# -------------------------

# ---- Eval/Prompt
SBERT_MODEL = "all-mpnet-base-v2"
BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"

# ---- Reporting profile
INDIVIDUAL_COLUMNS = [
    "bug_id", "tuning", "model", "prompt", "decode_mode",
    "Generated_Report", "CTQRS", "ROUGE1_R", "ROUGE1_F1", "SBERT", "SCORE_V2"
]

# ---- Misc
RANDOM_SEED = 42