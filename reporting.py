# reporting.py
from __future__ import annotations
from datetime import datetime
import pandas as pd
from config import INDIVIDUAL_COLUMNS, INDIVIDUAL_DIR, SUMMARY_DIR, REPORTS_DIR # (SUMMARY_COLUMNS 삭제)

def save_individual(results: list[dict], model_name: str, few_shot_k:int, run_id:str, tuning_tag:str="qlora4"):
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    
    # config.py의 INDIVIDUAL_COLUMNS (수정된 이름)을 기준으로 DataFrame을 재정렬/필터링
    out = pd.DataFrame(columns=INDIVIDUAL_COLUMNS)
    for c in INDIVIDUAL_COLUMNS:
        if c in df.columns: out[c] = df[c]
            
    path = INDIVIDUAL_DIR / f"{model_name}_{tuning_tag}_k{few_shot_k}_{run_id}.xlsx"
    try:
        out.to_excel(path, index=False)
    except Exception as e:
        print(f"[WARN] Failed to save individual report (filename too long?): {e}")
        # 파일 이름 축약 시도
        path = INDIVIDUAL_DIR / f"{model_name}_{tuning_tag[:20]}_k{few_shot_k}_{run_id[:15]}.xlsx"
        try:
            out.to_excel(path, index=False)
            print(f"[INFO] Saved with truncated name: {path}")
        except Exception as e2:
            print(f"[ERROR] Failed to save individual report even with truncated name: {e2}")
    return path

# --- save_summary (삭제됨) ---

# --- save_best_of_id (삭제됨) ---

def save_consolidated(all_results:list[list[dict]], total_sec:float, run_id:str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {"run_id": run_id, "elapsed_sec": float(total_sec)}
    
    flat_results = [r for lst in all_results for r in lst]
    if not flat_results:
        print("[WARN] No data to save in save_consolidated.")
        return

    # [수정됨] generation_agent의 새 키(CTQRS, SBERT 등)를 사용
    for model_name, rows in _iter_by_model(all_results):
        df = pd.DataFrame(rows)
        if len(df) == 0: continue
        row[f"{model_name}_avg_CTQRS"] = float(df["CTQRS"].mean())
        row[f"{model_name}_avg_ROUGE1_R"] = float(df["ROUGE1_R"].mean())
        row[f"{model_name}_avg_ROUGE1_F1"] = float(df["ROUGE1_F1"].mean())
        row[f"{model_name}_avg_SBERT"] = float(df["SBERT"].mean())
        row[f"{model_name}_avg_SCORE_V2"] = float(df["SCORE_V2"].mean())
    
    out = pd.DataFrame([row])
    path = REPORTS_DIR / f"consolidated_report_{run_id}.xlsx"
    out.to_excel(path, index=False)
    return path

def _iter_by_model(all_results):
    by = {}
    for lst in all_results:
        for r in lst:
            by.setdefault(r["model"], []).append(r)
    return by.items()