# generation_agent.py
# [MODIFIED] Unsloth 환경 변수 로직 제거 (HF-Only)
import os, sys
import torch, gc, math, numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    GenerationConfig, LogitsProcessorList
)
from peft import PeftModel
# [MODIFIED] config 임포트 변경
from config import RANDOM_SEED, SFT_PARAMS, MODEL_ID

def set_reproducible(seed: int = RANDOM_SEED):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FiniteClampLogitsProcessor:
    def __init__(self, min_val: float = -1e4, max_val: float = 1e4):
        self.min_val = float(min_val); self.max_val = float(max_val)
    def __call__(self, input_ids, scores):
        finite = torch.isfinite(scores)
        if not finite.all():
            scores = torch.where(finite, scores, torch.full_like(scores, -1e9))
        return torch.clamp(scores, min=self.min_val, max=self.max_val)


def _score_v2(ctqrs: float, r1_f1: float, r1_r: float, sbert: float) -> float:
    eps = 1e-6
    m_ctqrs = max(eps, ctqrs)
    m_f1 = max(eps, r1_f1)
    m_r  = max(eps, r1_r)
    m_s  = max(eps, (sbert + 1.0) / 2.0)  # SBERT cosine -> [0,1]
    w_ctqrs, w_f1, w_r, w_s = 0.40, 0.30, 0.20, 0.10
    val = (w_ctqrs * math.log(m_ctqrs) +
           w_f1    * math.log(m_f1) +
           w_r     * math.log(m_r) +
           w_s     * math.log(m_s))
    return float(math.exp(val))


class UnifiedGenAgent:
    """Single-try generation → evaluation pipeline for a model."""
    def __init__(self, model_name: str, adapter_path: str | None, prompt_factory, eval_agent,
                 batch_size: int = 4, greedy: bool = False, tuning_label: str = "qlora4",
                 lora_type: str = "qlora"): 
        
        assert model_name in MODEL_ID, f"Unknown model_name {model_name}"
        self.model_name = model_name
        self.model_id = MODEL_ID[model_name]
        self.adapter_path = adapter_path
        self.prompt_factory = prompt_factory
        self.eval_agent = eval_agent
        self.batch_size = int(batch_size)
        self.greedy = bool(greedy)
        self.tuning_label = tuning_label
        self.lora_type = lora_type

        set_reproducible()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ===== Tokenizer =====
        self.tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True, trust_remote_code=True)
        self.tok.padding_side = "left"
        self.tok.truncation_side = "right"
        if getattr(self.tok, "pad_token_id", None) is None:
            self.tok.pad_token_id = self.tok.eos_token_id

        # ===== [MODIFIED] lora_type에 따른 모델 로드 (HF-Only) =====
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = None
        
        if self.lora_type == "qlora":
            print(f"[INFO] Loading model in 4-bit (QLoRA mode)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif self.lora_type == "lora":
            print(f"[INFO] Loading model in 16-bit (LoRA mode)")
            # bnb_config는 None (양자화 안 함)
        else:
            print(f"[WARN] Unknown lora_type '{self.lora_type}'. Defaulting to 4-bit (qlora).")
            self.lora_type = "qlora" # Fallback
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )

        # [MODIFIED] Unsloth 분기 제거, HF 로직만 사용
        base = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map=("cuda:0" if torch.cuda.is_available() else "auto"),
            trust_remote_code=True,
        )
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        # ===== [수정 완료] =====

        # ===== Optional PEFT adapter =====
        if self.adapter_path is None or str(self.adapter_path).lower() == "base":
            print("[INFO] Using base model (no adapter).")
            self.model = base
        else:
            print(f"[INFO] Loading adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(base, self.adapter_path)
            print(f"[INFO] Adapter loaded successfully.")

        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _gen_cfg(self) -> GenerationConfig:
        # [MODIFIED] SFT_PARAMS (SFT 학습 시)의 max_new_tokens 참조
        gen_max = int(SFT_PARAMS.get("gen_max_new_tokens", 2048))
        
        common = dict(
            max_new_tokens=gen_max,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
            repetition_penalty=1.15,
        )
        if self.greedy:
            return GenerationConfig(do_sample=False, **common)
        return GenerationConfig(do_sample=True, temperature=0.7, top_p=0.9, top_k=50, **common)

    def run(self, df):
        import pandas as pd
        df = df.reset_index(drop=True)
        total = len(df)
        sum_ctqrs = sum_r = sum_f1 = sum_s = sum_score = 0.0
        processed = 0
        print(
            f"Progress: 0/{total} | CTQRS(avg)=0.000 | R1-R(avg)=0.000 | R1-F1(avg)=0.000 | SBERT(avg)=0.000 | SCORE(avg)=0.000",
            end="", flush=True
        )

        results = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gen_cfg = self._gen_cfg()

        for i in range(0, total, self.batch_size):
            batch = df.iloc[i:i + self.batch_size]
            prompts = [self.prompt_factory.create_prompt(r["NEW_llama_output"]) for _, r in batch.iterrows()]
            
            if i == 0 and processed == 0:
                print(f"\n[DEBUG] First prompt (k={self.prompt_factory.k}):\n------\n{prompts[0]}\n------")
            
            # [MODIFIED] SFT_PARAMS (SFT 학습 시)의 max_seq_len 참조
            inputs = self.tok(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=SFT_PARAMS["max_seq_len"] 
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_cfg,
                        logits_processor=LogitsProcessorList([FiniteClampLogitsProcessor(-1e4, 1e4)]),
                        return_dict_in_generate=True,
                        output_scores=False
                    )
                mode = "greedy" if not gen_cfg.do_sample else "sample"
            except RuntimeError as e:
                print(f"\n[ERROR] Generation failed (e.g., OOM): {e}. Falling back to greedy.")
                torch.cuda.empty_cache()
                greedy_cfg = GenerationConfig(
                    do_sample=False,
                    max_new_tokens=gen_cfg.max_new_tokens,
                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                    eos_token_id=self.tok.eos_token_id,
                    repetition_penalty=1.20
                )
                try:
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs, generation_config=greedy_cfg, return_dict_in_generate=True
                        )
                    mode = "greedy(fallback)"
                except Exception as e2:
                    print(f"\n[ERROR] Greedy fallback also failed: {e2}. Skipping batch.")
                    texts = ["GENERATION_FAILED"] * len(batch)
                    mode = "greedy(failed)"


            if mode != "greedy(failed)":
                seqs = outputs.sequences
                in_lens = inputs["attention_mask"].sum(dim=1)
                texts = []
                for bi in range(seqs.size(0)):
                    gen_ids = seqs[bi, in_lens[bi]:]
                    texts.append(self.tok.decode(gen_ids, skip_special_tokens=True).strip())

            for j, (_, row) in enumerate(batch.iterrows()):
                gen_text = texts[j].strip()
                gt_text = str(row["text"]).strip()

                if gen_text == "GENERATION_FAILED":
                    m = {"ctqrs": 0.0, "sbert": -1.0, "rouge1_r": 0.0, "rouge1_f1": 0.0}
                elif hasattr(self.eval_agent, "evaluate_report"):
                    m = self.eval_agent.evaluate_report(gen_text, gt_text)
                else:
                    m = self.eval_agent.evaluate(gen_text, gt_text)

                ctqrs = float(m["ctqrs"]); r1_r = float(m["rouge1_r"]); r1_f1 = float(m["rouge1_f1"]); sbert = float(m["sbert"])
                score_v2 = _score_v2(ctqrs, r1_f1, r1_r, sbert)

                results.append({
                    "bug_id": int(row["bug_id"]) if "bug_id" in row else int(i + j),
                    "tuning": self.tuning_label,
                    "model": self.model_name,
                    "prompt": prompts[j],
                    "decode_mode": mode,
                    "Generated_Report": gen_text,
                    "CTQRS": ctqrs,
                    "ROUGE1_R": r1_r,
                    "ROUGE1_F1": r1_f1,
                    "SBERT": sbert,
                    "SCORE_V2": score_v2,
                })
                
                processed += 1
                if processed > 0:
                    sum_ctqrs += ctqrs; sum_r += r1_r; sum_f1 += r1_f1; sum_s += sbert; sum_score += score_v2
                    print(
                        f"\rProgress: {processed}/{total} | "
                        f"CTQRS(avg)={sum_ctqrs/processed:.3f} | R1-R(avg)={sum_r/processed:.3f} | "
                        f"R1-F1(avg)={sum_f1/processed:.3f} | SBERT(avg)={sum_s/processed:.3f} | "
                        f"SCORE(avg)={sum_score/processed:.3f}",
                        end="", flush=True
                    )
        print()
        return results