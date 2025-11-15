# training_agent.py
# [MODIFIED] SFT와 DPO 학습을 모두 포함합니다. (HF-Only)
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import random, numpy as np, torch, sys
from typing import Literal

# [MODIFIED] `try` 블록 추가 (SyntaxError 수정)
try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
        BitsAndBytesConfig
    )
    from peft import (
        LoraConfig, get_peft_model, TaskType,
        prepare_model_for_kbit_training, PeftModel
    )
    from trl import DPOConfig, DPOTrainer as HFDpoTrainer
except ImportError:
    print("[ERROR] transformers, peft, and trl libraries must be installed.", file=sys.stderr)
    sys.exit(1)
    
try:
    from datasets import load_dataset
except ImportError:
    print("[ERROR] 'datasets' library must be installed for DPO training.", file=sys.stderr)
    sys.exit(1)

from config import (
    SFT_PARAMS, DPO_PARAMS, ADAPTER_SFT_DIRS, ADAPTER_DPO_DIRS,
    DPO_TRAIN_DATA_PATH_TEMPLATE, MODEL_ID, RANDOM_SEED, DATA_DIR
)

# [MODIFIED] BASE_TEMPLATE을 prompt_templates.py에서 가져옴
BASE_TEMPLATE = """You are an expert QA engineer and explainability analyst.
Convert the user's unstructured input into a CTQRS-maximized bug report while using explainable evidence and step-by-step internal reasoning.

## Structured Output
[Summary]
<one sentence: component/module + trigger condition + failure symptom>

[Steps to Reproduce]
1. <user action> - <immediate result/observation>
2. <user action> - <immediate result/observation>
3. <user action> - <immediate result/observation>
4. <optional extra user action> - <result>

[Expected Behavior]
<precise, testable outcome using the same key nouns as in Actual>

[Actual Behavior]
<what actually happens; include any error text or visible symptom; reuse the same key nouns>

[Environment]
- OS: <name version>
- Device/Browser: <model or browser name+version>
- App Version/Build: <semver or build tag>
- Network: <Wi-Fi/LTE/VPN/Proxy>
- Locale/Region: <e.g., en-US>

[Evidence]
<ONE of: short log line with error code | concrete file path | screenshot filename>
e.g., "ERROR 503 at startup", "/var/log/app/error_2025-09-07.log", "screenshot_2025-09-07_102314.png"

[Additional Info]
Frequency: always / often / sometimes
Workaround: <if any>
"""

def set_seed(seed:int=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ======================================================================
# --- 1. SFT (Supervised Fine-Tuning) 용 클래스 ---
# ======================================================================

class PairCausalDataset(torch.utils.data.Dataset):
    """ (v1 유지) 알파카 형식 SFT 데이터셋 """
    def __init__(self, df, tokenizer, max_len:int, system_prompt:str):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        self.system_prompt = system_prompt # [MODIFIED] BASE_TEMPLATE이 여기로 전달됨

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        inp = str(r["NEW_llama_output"]).strip()
        tgt = str(r["text"]).strip()
        prompt_text = self.system_prompt + "\n\n### Input:\n" + inp + "\n\n### Response:\n"
        prompt_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        target_ids = self.tok(tgt, add_special_tokens=False).input_ids
        eos_id = self.tok.eos_token_id
        input_ids = prompt_ids + target_ids
        if eos_id is not None: input_ids = input_ids + [eos_id]
        labels = [-100] * len(prompt_ids) + target_ids
        if eos_id is not None: labels = labels + [eos_id]
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels    = labels[:self.max_len]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }

def make_causal_collator(tokenizer):
    """ (v1 유지) SFT용 collator """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.eos_token_id
    def collate(features):
        max_len = max(f["input_ids"].size(0) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            L = f["input_ids"].size(0)
            pad_len = max_len - L
            input_ids.append(torch.nn.functional.pad(f["input_ids"], (0, pad_len), value=pad_id))
            attention_mask.append(torch.nn.functional.pad(f["attention_mask"], (0, pad_len), value=0))
            labels.append(torch.nn.functional.pad(f["labels"], (0, pad_len), value=-100))
        return { "input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask), "labels": torch.stack(labels) }
    return collate


class UnifiedSFTTrainer:
    """
    [MODIFIED] SFT (QLoRA / LoRA) 파인튜너. (HF-Only)
    """
    def __init__(self, 
                 model_name: str,
                 seed: int = RANDOM_SEED,
                 lora_type: Literal["qlora", "lora"] = "qlora",
                ):
        assert model_name in MODEL_ID, f"Unknown model_name {model_name}"
        
        self.model_name = model_name
        self.model_id = MODEL_ID[model_name]
        self.seed = seed
        self.lora_type = lora_type
        
        self.components_str = "base"
        self.adapter_root = ADAPTER_SFT_DIRS[model_name] / self.components_str

    def _load_model_tokenizer(self, params: dict):
        """Helper: HF 백엔드/LoRA 타입에 따라 모델과 토크나이저 로드"""
        max_seq_len = params["max_seq_len"]
        load_in_4bit = (self.lora_type == "qlora")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        print(f"[INFO] SFT: Loading model via HF backend (4-bit: {load_in_4bit})")
        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype,
            )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, quantization_config=quant_config,
            torch_dtype=compute_dtype, device_map="auto", trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, use_fast=True, trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        return model, tokenizer

    def _apply_peft_sft(self, model, params: dict):
        """Helper: SFT PEFT 적용 (HF-Only)"""
        
        print("[INFO] SFT: Applying PEFT via HF")
        if self.lora_type == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            
        peft_config = LoraConfig(
            r=params["lora_r"], lora_alpha=params["lora_alpha"], lora_dropout=params["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    def run(self, train_df) -> str:
        """SFT 학습 실행"""
        set_seed(self.seed)
        self.adapter_root.mkdir(parents=True, exist_ok=True)
        run_dir = self.adapter_root / f"RUN_SFT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        params = SFT_PARAMS
        system_prompt = BASE_TEMPLATE
        model, tokenizer = self._load_model_tokenizer(params)
        # [MODIFIED] 오타 수정 (sFT -> sft)
        model = self._apply_peft_sft(model, params)
        ds = PairCausalDataset(train_df, tokenizer, params["max_seq_len"], system_prompt=system_prompt)
        data_collator = make_causal_collator(tokenizer)

        use_grad_ckpt = False
        if self.lora_type == "qlora": 
             use_grad_ckpt = True

        args = TrainingArguments(
            output_dir=str(run_dir),
            per_device_train_batch_size=params.get("batch_size", 1),
            gradient_accumulation_steps=params.get("grad_accum_steps", 1),
            num_train_epochs=params["max_epochs"],
            learning_rate=params["lr"],
            lr_scheduler_type="cosine", warmup_ratio=0.03,
            bf16=torch.cuda.is_available() and self.lora_type == "lora",
            fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) and self.lora_type == "lora",
            logging_steps=params.get("logging_steps", 10),
            save_steps=params.get("save_steps", 200),
            save_total_limit=params.get("save_total_limit", 2),
            report_to="none", group_by_length=True,
            gradient_checkpointing=use_grad_ckpt,
        )
        trainer = Trainer(
            model=model, args=args, train_dataset=ds, data_collator=data_collator,
        )
        trainer.train()
        model.save_pretrained(str(run_dir))
        tokenizer.save_pretrained(str(run_dir))
        (self.adapter_root / "LATEST_ADAPTER.txt").write_text(str(run_dir), encoding="utf-8")
        
        print(f"[INFO] SFT Finetuning complete. Adapter saved to: {run_dir}")
        return str(run_dir)

# ======================================================================
# --- 2. DPO (Direct Preference Optimization) 용 클래스 ---
# ======================================================================

class UnifiedDPOTrainer:
    """
    [MODIFIED] DPO (QLoRA / LoRA) 파인튜너. (HF-Only)
    SFT 어댑터를 로드하여 DPO 학습을 수행합니다.
    """
    def __init__(self, 
                 model_name: str,
                 sft_adapter_path: str, # DPO의 기반이 될 SFT 어댑터
                 seed: int = RANDOM_SEED,
                 lora_type: Literal["qlora", "lora"] = "qlora",
                ):
        assert model_name in MODEL_ID, f"Unknown model_name {model_name}"
        if not Path(sft_adapter_path).exists():
            print(f"[ERROR] SFT adapter path not found: {sft_adapter_path}")
            raise SystemExit(1)

        self.model_name = model_name
        self.model_id = MODEL_ID[model_name]
        self.sft_adapter_path = sft_adapter_path
        self.seed = seed
        self.lora_type = lora_type
        
        self.components_str = "base"
        self.adapter_root = ADAPTER_DPO_DIRS[model_name] / self.components_str

    def _load_model_tokenizer_for_dpo(self, params: dict):
        """Helper: DPO용 모델/토크나이저 로드 및 SFT 어댑터 병합 (HF-Only)"""
        max_seq_len = params["max_seq_len"]
        load_in_4bit = (self.lora_type == "qlora")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        print(f"[INFO] DPO: Loading model via HF backend (4-bit: {load_in_4bit})")
        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype,
            )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id, quantization_config=quant_config,
            torch_dtype=compute_dtype, device_map="auto", trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, use_fast=True, trust_remote_code=True
        )
        
        print(f"[INFO] DPO: Loading SFT adapter onto base model from {self.sft_adapter_path}")
        model = PeftModel.from_pretrained(base_model, self.sft_adapter_path)
        print(f"[INFO] DPO: Merging SFT adapter into base model...")
        model = model.merge_and_unload() 

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        tokenizer.padding_side = "left" 
        return model, tokenizer

    def _apply_peft_dpo(self, model, params: dict):
        """Helper: DPO용 새 PEFT 어댑터 적용 (HF-Only)"""
        
        print("[INFO] DPO: Applying *new* PEFT adapter via HF")
        if self.lora_type == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            
        peft_config = LoraConfig(
            r=params["lora_r"], lora_alpha=params["lora_alpha"], lora_dropout=params["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    def _dpo_data_formatter(self, system_prompt: str):
        """DPO 데이터셋에 시스템 프롬프트를 적용하는 포매터"""
        def format_row(row):
            row["prompt"] = system_prompt + "\n\n### Input:\n" + row["prompt"] + "\n\n### Response:\n"
            row["chosen"] = row["chosen"]
            row["rejected"] = row["rejected"]
            return row
        return format_row

    def run(self) -> str:
        """DPO 학습 실행"""
        set_seed(self.seed)
        self.adapter_root.mkdir(parents=True, exist_ok=True)
        run_dir = self.adapter_root / f"RUN_DPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        params = DPO_PARAMS
        system_prompt = BASE_TEMPLATE
        
        dpo_dataset_path = Path(DPO_TRAIN_DATA_PATH_TEMPLATE.format(model_name=self.model_name))
                                 
        if not dpo_dataset_path.exists():
            print(f"[ERROR] DPO dataset not found: {dpo_dataset_path}")
            print(f"[ERROR] Run 'python data_agent.py' first.")
            raise SystemExit(1)

        model, tokenizer = self._load_model_tokenizer_for_dpo(params)
        model = self._apply_peft_dpo(model, params)
        
        dataset = load_dataset("json", data_files=str(dpo_dataset_path), split="train")
        formatter = self._dpo_data_formatter(system_prompt)
        dataset = dataset.map(formatter)

        train_args_dict = {
            "output_dir": str(run_dir),
            "per_device_train_batch_size": params.get("batch_size", 1),
            "gradient_accumulation_steps": params.get("grad_accum_steps", 4),
            "num_train_epochs": params["max_epochs"],
            "learning_rate": params["lr"],
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "bf16": torch.cuda.is_available() and self.lora_type == "lora",
            "fp16": not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) and self.lora_type == "lora",
            "logging_steps": params.get("logging_steps", 10),
            "save_steps": params.get("save_steps", 100),
            "save_total_limit": params.get("save_total_limit", 2),
            "report_to": "none",
            "remove_unused_columns": False,
            "label_names": None,
        }
        
        args = DPOConfig(**train_args_dict)

        dpo_trainer = HFDpoTrainer(
            model=model,
            ref_model=None, 
            args=args,
            #beta=params["beta"],
            train_dataset=dataset,
            #tokenizer=tokenizer,
            #max_length=params["max_seq_len"],
            #max_prompt_length=params["max_seq_len"] // 2, 
        )

        dpo_trainer.train()
        model.save_pretrained(str(run_dir))
        tokenizer.save_pretrained(str(run_dir))
        (self.adapter_root / "LATEST_ADAPTER.txt").write_text(str(run_dir), encoding="utf-8")
        
        print(f"[INFO] DPO Training complete. Adapter saved to: {run_dir}")
        return str(run_dir)