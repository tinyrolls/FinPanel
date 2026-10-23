from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional

from data.zvec_store import ZvecStore

STORES = ("market_intelligence", "low_level_reflection", "high_level_reflection")

QUERY_TRANSFORMS: Dict[str, str] = {
    "plain": "{q}",
    "short_term": "{q}. SHORT-TERM impact of the stock.",
    "medium_term": "{q}. MEDIUM-TERM impact of the stock.",
    "long_term": "{q}. LONG-TERM impact of the stock.",
}


class FinAgentMemory:

    def __init__(self, experiment_id: str, ticker: str, root: str | Path = "runtime/vector_memory/finagent", embed_fn=None):
        self.ticker = str(ticker)
        base = Path(root) / str(experiment_id) / self.ticker
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)
        self.stores: Dict[str, ZvecStore] = {
            s: ZvecStore(base / s, scalar_fields=["ticker"], embed_fn=embed_fn) for s in STORES
        }
        self._n: Dict[str, int] = {s: 0 for s in STORES}

    def add(self, store: str, text: str, date: str) -> Optional[str]:
        if store not in self.stores or not (text or "").strip():
            return None
        doc_id = f"{store[:2]}{self._n[store]}"
        self._n[store] += 1
        self.stores[store].add(doc_id, text, {"ticker": self.ticker, "t_write": str(date)[:10]})
        return doc_id

    def retrieve(self, store: str, query: str, t_now: str, k: int, query_types: Optional[List[str]] = None) -> List[Dict[str, str]]:
        if store not in self.stores:
            return []
        qtypes = query_types or ["plain"]
        seen: set = set()
        out: List[Dict[str, str]] = []
        for qt in qtypes:
            qtext = QUERY_TRANSFORMS.get(qt, "{q}").format(q=query or self.ticker)
            for r in self.stores[store].query(qtext, t_now=t_now, k=k, extra_filter=f"ticker = '{self.ticker}'"):
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                out.append({"id": r["id"], "date": (r.get("fields") or {}).get("t_write", ""), "text": r["text"]})
        return out
