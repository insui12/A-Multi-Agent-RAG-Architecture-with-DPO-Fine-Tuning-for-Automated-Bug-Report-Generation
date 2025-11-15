# controller_agent.py
from __future__ import annotations
import argparse, time, gc, sys, os, subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

from config import RANDOM_SEED
# [MODIFIED] data_agent에서 load_and_split_data 임포트 (경로 동일)
from data_agent import load_and_split_data
from training_agent import UnifiedSFTTrainer, UnifiedDPOTrainer
from prompt_factory import QABugPromptFactory
from evaluation_agent import EvaluationAgent
from reporting import save_individual, save_consolidated
from config import ADAPTER_SFT_DIRS, ADAPTER_DPO_DIRS, MODEL_ID

# [MODIFIED] _get_components_str 함수 제거

def _latest_adapter_or_fail(
    model_name: str, 
    # [MODIFIED] components_str 인수 제거
    adapter_type: Literal["sft", "dpo"]
) -> str:
    """
    [MODIFIED] 지정된 타입(SFT/DPO)의 'base' 어댑터 경로를 반환.
    """
    # [MODIFIED] components_str 하드코딩
    components_str = "base"
    
    if adapter_type == "sft":
        d = ADAPTER_SFT_DIRS[model_name] / components_str
        prefix = "RUN_SFT_"
    elif adapter_type == "dpo":
        d = ADAPTER_DPO_DIRS[model_name] / components_str
        prefix = "RUN_DPO_"
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")
        
    latest_file = d / "LATEST_ADAPTER.txt"
    
    if not d.exists():
        print(f"[ERROR] Adapter directory not found: {d}")
        print(f"[ERROR] Run SFT/DPO training first for 'base' components.")
        raise SystemExit(1)
        
    if latest_file.exists():
        p = latest_file.read_text(encoding="utf-8").strip()
        if p and Path(p).exists() and Path(p).name.startswith(prefix):
            print(f"[INFO] Found LATEST_ADAPTER ({adapter_type}): {p}")
            return p
            
    cands = [x for x in d.iterdir() if x.is_dir() and x.name.startswith(prefix)]
    if cands:
        latest = sorted(cands, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        print(f"[INFO] Found latest {adapter_type} adapter directory: {latest}")
        return str(latest)
        
    print(f"[ERROR] No {adapter_type} adapter found for {model_name} (components={components_str}).")
    print(f"[ERROR] Looked in: {d}")
    print(f"[ERROR] Run with --train_{adapter_type} first.")
    raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description="Unified v3 pipeline (SFT / DPO / Inference) (HF-Only)")
    
    ap.add_argument("--train_sft", action="store_true", help="Run SFT (Finetuning) per model.")
    ap.add_argument("--train_dpo", action="store_true", help="Run DPO training per model (requires SFT adapter).")
    
    ap.add_argument("--adapter", choices=["sft", "dpo", "base"], default="dpo",
                    help="Adapter type for INFERENCE: 'sft', 'dpo', or 'base' model.")
    ap.add_argument("--few_shot_k", type=int, default=0, choices=[0, 1, 2, 3])
    
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size for INFERENCE (HF-only).")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--test_run", action="store_true")
    ap.add_argument("--force_resplit", action="store_true")
    ap.add_argument("--models", nargs="+", default=["qwen", "llama", "mistral"],
                    choices=["qwen", "llama", "mistral"])

    ap.add_argument("--lora_type", choices=["qlora", "lora"], default="qlora",
                    help="Training/Inference method: 'qlora' (4-bit) or 'lora' (16-bit).")
    
    # [MODIFIED] --prompt_components 인수 제거
    
    args = ap.parse_args()

    is_training = args.train_sft or args.train_dpo
    if is_training and (args.train_sft and args.train_dpo):
        print("[ERROR] Cannot run --train_sft and --train_dpo simultaneously.")
        sys.exit(1)
    
    start = time.time()
    base_run_id = datetime.now().strftime("%Ym%d_%H%M%S")
    
    # [MODIFIED] components_str 하드코딩
    components_str = "base"
    
    run_id = f"{base_run_id}_{components_str}_k{args.few_shot_k}"

    train_df, val_df, _ = load_and_split_data(force_resplit=args.force_resplit, test_run=args.test_run)

    # === SFT 학습 분기 ===
    if args.train_sft:
        print("\n[INFO] === SFT TRAINING MODE (HF-Only) ===")
        
        # [MODIFIED] prompt_components 인수 제거
        _ = QABugPromptFactory(
            train_df=train_df, k=args.few_shot_k
        )

        for model_name in args.models:
            print(f"\n=== [TRAIN_SFT] Model: {model_name} | lora_type={args.lora_type} ===")
            print(f"    Components: {components_str} (Hardcoded)")
            
            # [MODIFIED] prompt_components 인수 제거
            tuner = UnifiedSFTTrainer(
                model_name=model_name,
                seed=RANDOM_SEED,
                lora_type=args.lora_type,
            )
            tuner.run(train_df)
            del tuner
            gc.collect()

        total_sec = time.time() - start
        print(f"\n[INFO] SFT Training finished. Elapsed: {total_sec:.1f}s")
        return

    # === DPO 학습 분기 ===
    if args.train_dpo:
        print("\n[INFO] === DPO TRAINING MODE (HF-Only) ===")
        
        for model_name in args.models:
            print(f"\n=== [TRAIN_DPO] Model: {model_name} | lora_type={args.lora_type} ===")
            print(f"    Components: {components_str} (Hardcoded)")

            # [MODIFIED] components_str 인수 제거
            sft_adapter_path = _latest_adapter_or_fail(model_name, "sft")
            print(f"    Using base SFT adapter: {sft_adapter_path}")
            
            # [MODIFIED] prompt_components 인수 제거
            tuner = UnifiedDPOTrainer(
                model_name=model_name,
                sft_adapter_path=sft_adapter_path,
                seed=RANDOM_SEED,
                lora_type=args.lora_type,
            )
            tuner.run() 
            del tuner
            gc.collect()

        total_sec = time.time() - start
        print(f"\n[INFO] DPO Training finished. Elapsed: {total_sec:.1f}s")
        return

    # === 추론 전용 ===
    from generation_agent import UnifiedGenAgent

    all_results = []
    
    eval_agent = EvaluationAgent()
    # [MODIFIED] prompt_components 인수 제거
    prompt_factory = QABugPromptFactory(
        train_df=train_df,
        k=args.few_shot_k,
    )

    for model_name in args.models:
        print(f"\n=== [INFER] Model: {model_name} | k={args.few_shot_k} | adapter={args.adapter} | lora_type={args.lora_type} ===")
        print(f"    Components: {components_str} (Hardcoded)")

        adapter_path = None
        tuning_tag = "base"
        
        if args.adapter == "sft":
            # [MODIFIED] components_str 인수 제거
            adapter_path = _latest_adapter_or_fail(model_name, "sft")
            tuning_tag = "sft"
        elif args.adapter == "dpo":
            # [MODIFIED] components_str 인수 제거
            adapter_path = _latest_adapter_or_fail(model_name, "dpo")
            tuning_tag = "dpo"
        else: # "base"
            print("    Using BASE model (no adapter).")

        gen = UnifiedGenAgent(model_name=model_name, adapter_path=adapter_path,
                              prompt_factory=prompt_factory, eval_agent=eval_agent,
                              batch_size=args.batch_size, greedy=args.greedy,
                              tuning_label=tuning_tag,
                              lora_type=args.lora_type)
        results = gen.run(val_df)
        all_results.append(results)

        # [MODIFIED] components_str은 'base'로 고정됨
        tag_str = f"{tuning_tag}_{components_str}"
        save_individual(results, model_name, args.few_shot_k, run_id, tuning_tag=tag_str)

        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            print("Final Averages -",
                  model_name,
                  f"CTQRS={df['CTQRS'].mean():.3f}",
                  f"R1-R={df['ROUGE1_R'].mean():.3f}",
                  f"R1-F1={df['ROUGE1_F1'].mean():.3f}",
                  f"SBERT={df['SBERT'].mean():.3f}",
                  f"SCORE_V2={df['SCORE_V2'].mean():.3f}",
                  sep=" | ")

        import torch
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    total_sec = time.time() - start
    save_consolidated(all_results, total_sec, f"{run_id}_consolidated")


if __name__ == "__main__":
    main()