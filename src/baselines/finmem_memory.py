from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.zvec_store import ZvecStore

CHAIN = ("short", "mid", "long")

LAYER_CFG: Dict[str, dict] = {
    "short": {"recency_factor": 3.0, "importance_factor": 0.92, "jump_lower": -1e9, "jump_upper": 60.0,
              "imp_scores": [50, 70, 90], "imp_probs": [0.5, 0.45, 0.05]},
    "mid": {"recency_factor": 90.0, "importance_factor": 0.967, "jump_lower": 60.0, "jump_upper": 80.0,
            "imp_scores": [40, 60, 80], "imp_probs": [0.05, 0.8, 0.15]},
    "long": {"recency_factor": 365.0, "importance_factor": 0.988, "jump_lower": 80.0, "jump_upper": 1e9,
             "imp_scores": [40, 60, 80], "imp_probs": [0.05, 0.15, 0.8]},
    "reflection": {"recency_factor": 365.0, "importance_factor": 0.988, "jump_lower": -1e9, "jump_upper": 1e9,
                   "imp_scores": [40, 60, 80], "imp_probs": [0.05, 0.15, 0.8]},
}
CLEANUP_RECENCY = 0.05
CLEANUP_IMPORTANCE = 5.0
TOP_K = 3
FEEDBACK_STEP = 5.0


@dataclass
class _Mem:
    mem_id: str
    layer: str
    text: str
    date: str
    importance: float
    recency: float = 1.0
    delta: int = 0


class FinMemBrain:

    def __init__(self, experiment_id: str, ticker: str, root: str | Path = "runtime/vector_memory/finmem", seed: int = 0, embed_fn=None):
        self.ticker = str(ticker)
        path = Path(root) / str(experiment_id) / self.ticker
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        self.store = ZvecStore(path, scalar_fields=["ticker"], embed_fn=embed_fn)
        self.book: Dict[str, _Mem] = {}
        self._next = 0
        self._rng = np.random.default_rng(seed)

    def _sample_importance(self, layer: str) -> float:
        cfg = LAYER_CFG[layer]
        return float(self._rng.choice(cfg["imp_scores"], p=cfg["imp_probs"]))

    def add(self, layer: str, text: str, date: str) -> Optional[str]:
        if not (text or "").strip() or layer not in LAYER_CFG:
            return None
        mem_id = f"m{self._next}"
        self._next += 1
        self.store.add(mem_id, text, {"ticker": self.ticker, "t_write": str(date)[:10]})
        self.book[mem_id] = _Mem(mem_id, layer, str(text), str(date)[:10], self._sample_importance(layer))
        return mem_id

    def query(self, layer: str, query_text: str, t_now: str, top_k: int = TOP_K) -> List[Tuple[str, str]]:
        live = {m.mem_id: m for m in self.book.values() if m.layer == layer}
        if not live:
            return []
        rows = self.store.query(query_text or self.ticker, t_now=t_now, k=max(len(self.book), 1), extra_filter=f"ticker = '{self.ticker}'")
        scored: List[Tuple[float, str, str]] = []
        for r in rows:
            m = live.get(r["id"])
            if m is None:
                continue
            final = float(r["sim"]) + m.recency + min(m.importance, 100.0) / 100.0
            scored.append((final, m.mem_id, m.text))
        scored.sort(key=lambda x: -x[0])
        return [(mid, txt) for _, mid, txt in scored[:top_k]]

    def step(self) -> None:
        self._decay()
        self._clean_up()
        self._jump()
        self._jump()

    def _decay(self) -> None:
        for m in self.book.values():
            m.delta += 1
            rf = LAYER_CFG[m.layer]["recency_factor"]
            m.recency = math.exp(-(m.delta / rf))
            m.importance *= LAYER_CFG[m.layer]["importance_factor"]

    def _clean_up(self) -> None:
        dead = [mid for mid, m in self.book.items() if m.recency < CLEANUP_RECENCY or m.importance < CLEANUP_IMPORTANCE]
        for mid in dead:
            self.book.pop(mid, None)

    def _jump(self) -> None:
        for m in self.book.values():
            cfg = LAYER_CFG[m.layer]
            if m.importance >= cfg["jump_upper"] and m.layer in CHAIN:
                i = CHAIN.index(m.layer)
                if i + 1 < len(CHAIN):
                    m.layer = CHAIN[i + 1]
                    m.recency, m.delta = 1.0, 0
            elif m.importance < cfg["jump_lower"] and m.layer in CHAIN:
                i = CHAIN.index(m.layer)
                if i - 1 >= 0:
                    m.layer = CHAIN[i - 1]
                    m.recency, m.delta = 1.0, 0

    def feedback(self, cited_ids: List[str], sign: int) -> None:
        if not sign:
            return
        for mid in cited_ids:
            m = self.book.get(mid)
            if m is not None:
                m.importance += FEEDBACK_STEP * float(sign)
