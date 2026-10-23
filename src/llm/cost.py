from __future__ import annotations

import atexit
import json
import threading
from datetime import datetime
from pathlib import Path

_LOCK = threading.Lock()
_TALLY: dict[str, dict] = {}
_FLUSHED = False

_COST_PATH = Path(__file__).resolve().parents[2] / "logs" / "exp_cost.jsonl"
_FIELDS = ("requests", "prompt_tokens", "completion_tokens", "total_tokens")


def record(model: str | None, prompt_tokens, completion_tokens, total_tokens, cost_usd) -> None:
    with _LOCK:
        t = _TALLY.setdefault(model or "unknown", {**{f: 0 for f in _FIELDS}, "cost_usd": 0.0})
        t["requests"] += 1
        t["prompt_tokens"] += int(prompt_tokens or 0)
        t["completion_tokens"] += int(completion_tokens or 0)
        t["total_tokens"] += int(total_tokens or 0)
        t["cost_usd"] += float(cost_usd or 0.0)


def flush() -> None:
    global _FLUSHED
    with _LOCK:
        if _FLUSHED or not _TALLY:
            return
        _FLUSHED = True
        from utils.run_context import get_run_name

        by_model = {m: dict(v) for m, v in _TALLY.items()}
        rec = {"timestamp": datetime.now().isoformat(), "run_name": get_run_name()}
        for f in _FIELDS:
            rec[f] = sum(v[f] for v in by_model.values())
        rec["cost_usd"] = round(sum(v["cost_usd"] for v in by_model.values()), 6)
        rec["by_model"] = {m: {**{f: v[f] for f in _FIELDS}, "cost_usd": round(v["cost_usd"], 6)} for m, v in by_model.items()}
    try:
        _COST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _COST_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


atexit.register(flush)
