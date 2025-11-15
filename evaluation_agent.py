# evaluation_agent.py
from __future__ import annotations
import sys, importlib.util, types, torch
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from config import SBERT_MODEL

class EvaluationAgent:
    def __init__(self, ctqrs_path: str = "evaluation/perfect_ctqrs.py"):
        self.ctqrs_eval = self._load_ctqrs(ctqrs_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sbert = SentenceTransformer(SBERT_MODEL, device=self.device)
        self.rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    def _load_ctqrs(self, path:str):
        spec = importlib.util.spec_from_file_location("perfect_ctqrs", path)
        if spec is None or spec.loader is None:
            print(f"[ERROR] Cannot import CTQRS module at {path}")
            sys.exit(1)
        mod = importlib.util.module_from_spec(spec)  # type: types.ModuleType
        spec.loader.exec_module(mod)
        if not hasattr(mod, "evaluate_bug_report"):
            print(f"[ERROR] 'evaluate_bug_report' not found in {path}")
            sys.exit(1)
        return mod.evaluate_bug_report

    # --- compatibility shim for generation_agent ---
    def evaluate(self, gen_text: str, gt_text: str):
        """Backwards-compatible: delegate to evaluate_report()."""
        return self.evaluate_report(gen_text, gt_text)

    def evaluate_report(self, gen_text:str, gt_text:str):
        # CTQRS (normalized to [0,1])
        r = self.ctqrs_eval(gen_text)
        total = float(r.get("total_score", 0.0))
        max_possible = float(r.get("max_possible_score", 16.0))
        ctqrs = (total / max_possible) if max_possible > 0 else 0.0

        # SBERT cosine similarity
        e_gen = self.sbert.encode(gen_text, convert_to_tensor=True, show_progress_bar=False)
        e_gt  = self.sbert.encode(gt_text,  convert_to_tensor=True, show_progress_bar=False)
        sbert = util.pytorch_cos_sim(e_gen, e_gt).item()

        # ROUGE-1: recall & f1
        rs = self.rouge.score(gt_text, gen_text)["rouge1"]
        r1_r = float(rs.recall)
        r1_f1 = float(rs.fmeasure)

        return {"ctqrs": ctqrs, "sbert": sbert, "rouge1_r": r1_r, "rouge1_f1": r1_f1}