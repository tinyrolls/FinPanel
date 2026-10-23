from __future__ import annotations

import fcntl
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

DEFAULT_ROOT = Path("runtime") / "reputation_memory"

LOOK_AHEAD_GAP_DAYS = 1


BUCKET_RESOLUTIONS = ("month", "quarter", "half", "year", "alltime")
_RES_ORDER = {r: i for i, r in enumerate(BUCKET_RESOLUTIONS)}
_BAND_BOUNDS = (("month", 6), ("quarter", 12), ("half", 24))


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def band_for_age(age_months: int) -> str:
    for res, ub in _BAND_BOUNDS:
        if age_months < ub:
            return res
    return "year"


def period_label(d: date, resolution: str) -> str:
    if resolution == "month":
        return f"{d.year}-M{d.month:02d}"
    if resolution == "quarter":
        return f"{d.year}-Q{(d.month - 1) // 3 + 1}"
    if resolution == "half":
        return f"{d.year}-H{1 if d.month <= 6 else 2}"
    if resolution == "year":
        return f"{d.year}"
    return "ALL"


def period_start(period: str) -> Optional[date]:
    p = (period or "").strip()
    if not p or p == "ALL":
        return None
    try:
        if "-M" in p:
            y, m = p.split("-M")
            return date(int(y), int(m), 1)
        if "-Q" in p:
            y, q = p.split("-Q")
            return date(int(y), (int(q) - 1) * 3 + 1, 1)
        if "-H" in p:
            y, h = p.split("-H")
            return date(int(y), 1 if int(h) == 1 else 7, 1)
        return date(int(p), 1, 1)
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class MemoryBucket:
    actor: str
    ticker: str
    resolution: str
    period: str
    content: str
    n_evidences: int
    last_updated: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryBucket":
        return cls(
            actor=str(d["actor"]),
            ticker=str(d["ticker"]),
            resolution=str(d["resolution"]),
            period=str(d["period"]),
            content=str(d.get("content", "")),
            n_evidences=int(d.get("n_evidences", 1)),
            last_updated=str(d["last_updated"]),
        )


def _is_superseded(b: "MemoryBucket", current: dict) -> bool:
    ps = period_start(b.period)
    if ps is None:
        return False
    for r in BUCKET_RESOLUTIONS:
        if r == "alltime" or _RES_ORDER[r] <= _RES_ORDER[b.resolution]:
            continue
        if (r, period_label(ps, r)) in current:
            return True
    return False


def render_resume(
    persona_text: str,
    sections: dict,
    actor_name: str = "",
    focus_ticker: str = "",
) -> str:
    parts: List[str] = []
    name = actor_name or "this actor"
    if (persona_text or "").strip():
        parts.append(f"[Philosophy] {persona_text.strip()}")
    if sections:
        order = sorted(
            sections.keys(),
            key=lambda tk: (tk != focus_ticker, tk),
        )
        for tk in order:
            spans = []
            for b in sections[tk] or []:
                body = (getattr(b, "content", "") or "").strip()
                if body:
                    spans.append(f"  ({getattr(b, 'period', '')}) {body}")
            if not spans:
                continue
            tag = f"{tk} — today's name" if tk == focus_ticker else tk
            parts.append(f"[{tag}]\n" + "\n".join(spans))
    if not parts:
        return ""
    header = f"=== Track record: {name} ===" if name != "this actor" else "=== Track record ==="
    return header + "\n" + "\n".join(parts)


def parse_date(s: str) -> Optional[date]:
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


class ReputationMemory:

    def __init__(
        self,
        root: Path = DEFAULT_ROOT,
        experiment_id: str = "default",
    ):
        self.experiment_id = str(experiment_id)
        self.dir = Path(root) / self.experiment_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.buckets_path = self.dir / "buckets.jsonl"


    def append_bucket(self, bucket: MemoryBucket) -> None:
        line = json.dumps(bucket.to_dict(), ensure_ascii=False) + "\n"
        with open(self.buckets_path, "a", encoding="utf-8") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass

    def _iter_buckets(self) -> Iterable[MemoryBucket]:
        if not self.buckets_path.exists():
            return
        with open(self.buckets_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield MemoryBucket.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

    def _current_buckets(self, actor: str, ticker: str, cutoff: date) -> dict:
        best: dict = {}
        for b in self._iter_buckets():
            if b.actor != actor or b.ticker != ticker:
                continue
            d = parse_date(b.last_updated)
            if d is None or d > cutoff:
                continue
            key = (b.resolution, b.period)
            cur = best.get(key)
            if cur is None or (parse_date(cur.last_updated) or d) <= d:
                best[key] = b
        return best

    def full_spectrum(
        self, actor: str, ticker: str, t_now: str,
    ) -> List[MemoryBucket]:
        now_d = parse_date(t_now)
        if now_d is None:
            return []
        cutoff = now_d - timedelta(days=LOOK_AHEAD_GAP_DAYS)
        cur = self._current_buckets(actor, ticker, cutoff)
        out = [
            b for (res, _p), b in cur.items()
            if res == "alltime" or not _is_superseded(b, cur)
        ]

        def _key(b: MemoryBucket):
            if b.resolution == "alltime":
                return (1, date.max)
            return (0, period_start(b.period) or date.min)

        out.sort(key=_key, reverse=True)
        return out

    def resumes_for_manager(self, actors: Sequence[str], t_now: str) -> dict:
        now_d = parse_date(t_now)
        if now_d is None:
            return {a: {} for a in actors}
        cutoff = now_d - timedelta(days=LOOK_AHEAD_GAP_DAYS)
        wanted = set(actors)
        grouped: dict = {}
        for b in self._iter_buckets():
            if b.actor not in wanted or b.resolution == "alltime":
                continue
            d = parse_date(b.last_updated)
            if d is None or d > cutoff:
                continue
            cur = grouped.setdefault((b.actor, b.ticker), {})
            key = (b.resolution, b.period)
            prev = cur.get(key)
            if prev is None or (parse_date(prev.last_updated) or d) <= d:
                cur[key] = b
        out: dict = {a: {} for a in actors}
        for (actor, ticker), cur in grouped.items():
            live = [b for b in cur.values() if not _is_superseded(b, cur)]
            live.sort(key=lambda b: period_start(b.period) or date.min, reverse=True)
            if live:
                out[actor][ticker] = live
        return out

    def pending_rollups(self, actor: str, ticker: str, t_now: str) -> List[dict]:
        now_d = parse_date(t_now)
        if now_d is None:
            return []
        cur = self._current_buckets(actor, ticker, now_d)
        groups: dict = {}
        for (res, period), b in cur.items():
            if res == "alltime" or _is_superseded(b, cur):
                continue
            ps = period_start(period)
            if ps is None:
                continue
            target = band_for_age(_months_between(ps, now_d))
            if _RES_ORDER[target] <= _RES_ORDER[res]:
                continue
            key = (target, period_label(ps, target))
            groups.setdefault(key, []).append(b)
        return [
            {"target_resolution": r, "target_period": p, "sources": src}
            for (r, p), src in sorted(groups.items())
        ]


_EXPERIMENT_ENV_VAR = "FINPANEL_EXPERIMENT_ID"


def derive_experiment_id(
    label: str,
    start_date: str,
    end_date: str,
    model_name: str = "",
) -> str:
    cid = (label or "default").lower()
    s = (start_date or "").strip()[:10]
    e = (end_date or "").strip()[:10]
    base = f"{cid}_{s}_{e}" if s and e else cid
    m = re.sub(r"[^a-z0-9.-]+", "-", (model_name or "").strip().lower()).strip("-")
    return f"{base}_{m}" if m else base


def get_default_ledger(experiment_id: Optional[str] = None) -> ReputationMemory:
    eid = experiment_id or os.environ.get(_EXPERIMENT_ENV_VAR) or "default"
    return ReputationMemory(root=DEFAULT_ROOT, experiment_id=eid)
