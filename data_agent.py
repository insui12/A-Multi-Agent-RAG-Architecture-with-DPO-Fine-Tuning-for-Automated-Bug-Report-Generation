# data_agent.py
"""
데이터 로드/분할 및 DPO 데이터셋 구축 스크립트 (통합 버전) (HF-Only)

1. SFT 모델로 'Generated' 응답 생성 (캐시)
2. 'Golden' vs 'Generated' 점수 비교
3. 'Quality Floor' 및 'Noise Margin' 필터 적용
4. 최종 (prompt, chosen, rejected) .jsonl 파일 생성
"""
from __future__ import annotations
import argparse
import pandas as pd
import torch
import math
import json
import gc
import sys
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split # [MODIFIED] 원본 data_agent에서 임포트

from config import (
    MODEL_ID, ADAPTER_SFT_DIRS, DPO_PARAMS, SFT_PARAMS,
    DPO_CANDIDATES_PATH_TEMPLATE, DPO_TRAIN_DATA_PATH_TEMPLATE, DATA_DIR,
    # [MODIFIED] 원본 data_agent의 경로 임포트
    RAW_DATA_PATH, TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH,
    RANDOM_SEED
)
from prompt_factory import QABugPromptFactory
from evaluation_agent import EvaluationAgent

# --- generation_agent.py에서 핵심 기능 임포트 ---
try:
    from generation_agent import UnifiedGenAgent, _score_v2, FiniteClampLogitsProcessor
    print("[INFO] Imported components from generation_agent.")
except ImportError as e:
    print(f"[WARN] Could not import from generation_agent ({e}). Redefining _score_v2...")
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
# -----------------------------------------------

# [NEW] 원본 data_agent.py의 `load_and_split_data` 함수
REQ_COLS = ["NEW_llama_output", "text"]

def load_and_split_data(test_size=0.1, validation_size=0.1, force_resplit=False, test_run=False):
    if TRAIN_DATA_PATH.exists() and VALIDATION_DATA_PATH.exists() and TEST_DATA_PATH.exists() and not force_resplit:
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        val_df = pd.read_csv(VALIDATION_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)
    else:
        try:
            df = pd.read_excel(RAW_DATA_PATH)
        except Exception as e:
            print(f"[ERROR] Failed to load: {RAW_DATA_PATH} -> {e}")
            sys.exit(1)
        for c in REQ_COLS:
            if c not in df.columns:
                print(f"[ERROR] Missing required column: {c}")
                sys.exit(1)
        if "bug_id" not in df.columns:
            df.insert(0, "bug_id", range(len(df)))

        train_val, test_df = train_test_split(df, test_size=test_size, random_state=RANDOM_SEED)
        train_df, val_df = train_test_split(
            train_val, test_size=validation_size/(1-test_size), random_state=RANDOM_SEED
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(TRAIN_DATA_PATH, index=False)
        val_df.to_csv(VALIDATION_DATA_PATH, index=False)
        test_df.to_csv(TEST_DATA_PATH, index=False)

    if test_run:
        train_df = train_df.head(100).copy()
        val_df = val_df.head(10).copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

# -----------------------------------------------
# --- DPO 데이터셋 구축 로직 (기존 build_dpo_dataset.py) ---
# -----------------------------------------------

def _clean_generated_text(text: str) -> str:
    """
    Apply priority-based filtering to clean known template contamination
    from 'generated' text.
    """
    if not isinstance(text, str):
        return ""
        
    marker1 = "### Response:"
    marker2 = "created attachment"
    marker3 = "found in"
    
    idx1 = text.lower().find(marker1.lower())
    if idx1 != -1:
        return text[idx1 + len(marker1):].strip()

    idx2 = text.lower().find(marker2.lower())
    if idx2 != -1:
        return text[idx2:].strip()

    idx3 = text.lower().find(marker3.lower())
    if idx3 != -1:
        return text[idx3:].strip()
            
    return text.strip()


def _get_latest_sft_adapter(model_name: str) -> str:
    """
    [MODIFIED] 'base' 컴포넌트로 학습된 최신 SFT 어댑터 경로를 찾습니다.
    """
    components_str = "base" # [MODIFIED] 하드코딩
    d = ADAPTER_SFT_DIRS[model_name] / components_str
    
    if not d.exists():
        print(f"[ERROR] SFT adapter directory not found: {d}")
        print(f"[ERROR] Run SFT training first (e.g., via controller_agent.py --train_sft).")
        raise SystemExit(1)
    
    latest_file = d / "LATEST_ADAPTER.txt"
    
    if latest_file.exists():
        p = latest_file.read_text(encoding="utf-8").strip()
        if p and Path(p).exists():
            print(f"[INFO] Found LATEST_ADAPTER: {p}")
            return p
            
    cands = [x for x in d.iterdir() if x.is_dir() and x.name.startswith("RUN_SFT_")] 
    if cands:
        latest = sorted(cands, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        print(f"[INFO] Found latest SFT adapter directory: {latest}")
        return str(latest)
        
    print(f"[ERROR] No SFT adapter found for {model_name} (components={components_str}).")
    print(f"[ERROR] Looked in: {d}")
    print(f"[ERROR] Run SFT training first (e.g., via controller_agent.py --train_sft).")
    raise SystemExit(1)


def generate_candidates(model_name: str, batch_size: int) -> pd.DataFrame:
    """
    (1단계) SFT 모델을 로드하여 'Generated' 응답을 생성합니다.
    """
    print(f"--- 1. Generating Candidates for {model_name} ---")
    
    components_str = "base"
    
    # 1. 데이터 로드 (같은 파일 내의 함수 호출)
    train_df, _, _ = load_and_split_data(force_resplit=False, test_run=False)
    
    # 2. SFT 모델 로드 (GenerationAgent 사용)
    from generation_agent import UnifiedGenAgent

    adapter_path = _get_latest_sft_adapter(model_name)
    eval_agent = EvaluationAgent()
    
    prompt_factory = QABugPromptFactory(
        train_df=train_df,
        k=0,
    )

    gen_agent = UnifiedGenAgent(
        model_name=model_name,
        adapter_path=adapter_path,
        prompt_factory=prompt_factory,
        eval_agent=eval_agent,
        batch_size=batch_size,
        greedy=False,
        tuning_label=f"sft_{components_str}",
        lora_type="qlora"
    )

    # 3. 추론 실행
    print(f"Generating {len(train_df)} responses using SFT model...")
    results = gen_agent.run(train_df)
    results_df = pd.DataFrame(results)
    
    if "bug_id" not in results_df:
        results_df["bug_id"] = train_df["bug_id"]

    merged_df = pd.merge(
        train_df[["bug_id", "NEW_llama_output", "text"]],
        results_df[["bug_id", "Generated_Report"]],
        on="bug_id"
    )
    
    candidates_df = merged_df.rename(columns={
        "NEW_llama_output": "prompt_input",
        "text": "golden",
        "Generated_Report": "generated"
    })
    
    del gen_agent, eval_agent, prompt_factory
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Generation complete. {len(candidates_df)} candidates created.")
    return candidates_df


def build_dpo_dataset(model_name: str, candidates_df: pd.DataFrame):
    """
    (2단계) 점수 비교, 필터 적용 후 최종 .jsonl DPO 데이터셋 생성
    """
    print(f"--- 2. Building DPO Dataset for {model_name} ---")
    
    output_path = Path(DPO_TRAIN_DATA_PATH_TEMPLATE.format(model_name=model_name))
    
    quality_floor = DPO_PARAMS.get("QUALITY_FLOOR", 0.5)
    noise_margin = DPO_PARAMS.get("NOISE_MARGIN", 0.01)
    
    print(f"Filters: QualityFloor > {quality_floor}, NoiseMargin > {noise_margin}")
    
    eval_agent = EvaluationAgent()
    dpo_data = []
    
    total = len(candidates_df)
    kept, floor_dropped, margin_dropped, clean_failed = 0, 0, 0, 0
    
    pbar = tqdm(candidates_df.iterrows(), total=total, desc="Filtering DPO Pairs")
    for _, row in pbar:
        prompt = str(row["prompt_input"])
        
        # [MODIFIED] 'golden' 텍스트에도 정제(clean) 로직 적용
        golden_raw = str(row["golden"])
        golden = _clean_generated_text(golden_raw) # <-- 수정된 부분
        
        generated_raw = str(row["generated"])
        generated = _clean_generated_text(generated_raw)
        
        if not golden or not generated or generated == "GENERATION_FAILED":
            clean_failed += 1
            continue
            
        try:
            m_golden = eval_agent.evaluate_report(golden, golden)
            m_generated = eval_agent.evaluate_report(generated, golden)
            
            s_golden = _score_v2(m_golden["ctqrs"], m_golden["rouge1_f1"], m_golden["rouge1_r"], m_golden["sbert"])
            s_generated = _score_v2(m_generated["ctqrs"], m_generated["rouge1_f1"], m_generated["rouge1_r"], m_generated["sbert"])
        except Exception as e:
            print(f"[WARN] Scoring failed for one row: {e}. Skipping.")
            continue
            
        if max(s_golden, s_generated) < quality_floor:
            floor_dropped += 1
            continue
            
        if abs(s_golden - s_generated) < noise_margin:
            margin_dropped += 1
            continue
            
        if s_golden > s_generated:
            chosen = golden
            rejected = generated
        else:
            chosen = generated
            rejected = golden
            
        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
        kept += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item) + "\n")
            
    print("--- DPO Build Complete ---")
    print(f"Total processed: {total}")
    print(f"Kept (Final Set): {kept}")
    print(f"Dropped (Quality Floor): {floor_dropped}")
    print(f"Dropped (Noise Margin): {margin_dropped}")
    print(f"Dropped (Clean/Empty): {clean_failed}")
    print(f"Saved DPO dataset to: {output_path}")
    

def main():
    ap = argparse.ArgumentParser(description="Build DPO Dataset")
    ap.add_argument("--models", nargs="+", default=["qwen", "llama", "mistral"],
                    choices=["qwen", "llama", "mistral"])
    ap.add_argument("--force_generate", action="store_true",
                    help="Force re-generation of SFT model outputs even if cache exists.")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Batch size for SFT model inference.")
    args = ap.parse_args()
    
    components_str = "base"

    for model_name in args.models:
        print(f"\n===== Processing Model: {model_name} (Components: {components_str}) =====")
        
        cache_path = Path(DPO_CANDIDATES_PATH_TEMPLATE.format(model_name=model_name))
        
        if not cache_path.exists() or args.force_generate:
            if args.force_generate:
                print(f"[INFO] --force_generate enabled. Regenerating candidates...")
            else:
                print(f"[INFO] Cache file not found: {cache_path}. Generating candidates...")
            
            candidates_df = generate_candidates(model_name, args.batch_size)
            candidates_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
            print(f"Saved candidates cache to: {cache_path}")
        else:
            print(f"[INFO] Loading candidates from cache: {cache_path}")
            candidates_df = pd.read_csv(cache_path)

        build_dpo_dataset(model_name, candidates_df)


if __name__ == "__main__":
    main()