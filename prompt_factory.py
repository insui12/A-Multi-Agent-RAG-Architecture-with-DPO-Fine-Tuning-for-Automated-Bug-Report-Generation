# prompt_factory.py
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np
import faiss, sys
from sentence_transformers import SentenceTransformer
from config import BGE_MODEL_ID
# [MODIFIED] prompt_templates 임포트 제거

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

# [MODIFIED] _assemble_system_prompt 함수 제거


class QABugPromptFactory:
    """
    [MODIFIED] v3: Builds Alpaca-formatted prompts, optionally attaching k example pairs
    via dense retrieval. System prompt is fixed to BASE_TEMPLATE.
    """
    def __init__(
        self,
        train_df: pd.DataFrame,
        k: int = 0,
        # [MODIFIED] prompt_components 및 v1 DEPRECATED 인수 모두 제거
    ):
        self.k = max(0, min(3, int(k)))
        self.train_df = train_df.reset_index(drop=True) if train_df is not None else pd.DataFrame()
        
        try:
            import torch
            RAG_AVAILABLE = True
        except ImportError:
            RAG_AVAILABLE = False
            
        self.use_few_shot = (self.k > 0 and len(self.train_df) > 0 and RAG_AVAILABLE)

        # [MODIFIED] v1 경고 제거
        # [MODIFIED] v2 동적 조립 로직 제거
        
        # [MODIFIED] 시스템 프롬프트를 BASE_TEMPLATE으로 고정
        self.system_prompt = BASE_TEMPLATE
        
        print(f"[INFO] PromptFactory initialized. k={self.k}. Components=base (Hardcoded)")

        self.index = None
        self.search_model = None
        if self.use_few_shot:
            self._build_faiss()

    # ---- Retrieval index (v1 유지)
    def _build_faiss(self):
        print("[INFO] Building FAISS index for RAG (k > 0)...")
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            device = "cuda" if self._has_cuda() else "cpu"
            self.search_model = SentenceTransformer(BGE_MODEL_ID, device=device)
        except ImportError as e:
            print(f"[ERROR] Failed to load RAG components: {e}", file=sys.stderr)
            self.use_few_shot = False
            return
            
        corpus_texts = [str(x).strip() for x in self.train_df["NEW_llama_output"].tolist()]
        embs = self.search_model.encode(
            corpus_texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
        )
        embs = embs.astype(np.float32)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        print("[INFO] FAISS index built.")

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # ---- Few-shot block (v2: Alpaca 형식으로 수정)
    def _few_shot_block(self, input_summary: str) -> str:
        if not self.use_few_shot or self.index is None or self.search_model is None:
            return ""
        k = min(self.k, len(self.train_df))
        if k <= 0:
            return ""
        
        q = self.search_model.encode([input_summary], show_progress_bar=False, normalize_embeddings=True)
        q = q.astype(np.float32)
        _, idxs = self.index.search(q, k)
        
        exs = []
        for idx in reversed(idxs[0]): # (가장 유사도가 낮은 것부터 순서대로)
            row = self.train_df.iloc[int(idx)]
            ex_in = str(row.get("NEW_llama_output", "")).strip()
            ex_out = str(row.get("text", "")).strip()
            if ex_in and ex_out:
                exs.append(f"### Input:\n{ex_in}\n\n### Response:\n{ex_out}\n")
        
        if not exs:
            return ""
            
        return "\n".join(exs) + "\n" # 마지막에 개행 추가

    # ---- Final prompt (v2: Alpaca 형식으로 수정)
    def create_prompt(self, input_summary: str) -> str:
        """
        v3: 최종 프롬프트를 Alpaca 형식으로 조립합니다.
        [SYSTEM] + [FEW_SHOT_TURNS] + [ACTUAL_INPUT_TURN]
        """
        return (
            self.system_prompt
            + "\n\n"
            + self._few_shot_block(input_summary) # (RAG 예제 턴)
            + f"### Input:\n{input_summary}\n\n### Response:\n" # (실제 입력 턴)
        )