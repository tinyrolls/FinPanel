from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

DISTILL_OK = "ok"
DISTILL_LLM_FAILED = "llm_failed"
DISTILL_EXCEPTION = "exception"

_DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

Obs = Tuple[float, float]


def _sign(x: float) -> int:
    return 1 if x > 1e-9 else (-1 if x < -1e-9 else 0)


def track_record_line(obs: Sequence[Obs]) -> str:
    obs = list(obs)
    if len(obs) < 2:
        return ""
    signs = [_sign(s) for s, _ in obs]
    directional = [(s, r) for s, r in obs if _sign(s) != 0]
    if not directional:
        return f"{len(obs)} scored days, never took a side"

    stances = 1 + sum(1 for a, b in zip(signs, signs[1:]) if a != b)

    n = len(directional)
    hit = sum(1 for s, r in directional if _sign(s) == _sign(r)) / n
    base_hit = sum(1 for _, r in directional if r > 0) / n
    gap = sum(_sign(s) * r for s, r in directional)
    window = sum(r for _, r in obs)
    long_frac = sum(1 for s, _ in directional if _sign(s) > 0) / n

    if stances == 1:
        side = "long" if long_frac > 0.5 else "short"
        return (
            f"one stance ({side}) held across all {len(obs)} scored days — no "
            f"discriminating record: with the view never changing, its action-gap "
            f"({gap:+.1%}) is fixed by the instrument's own drift over the window, "
            f"not by any judgement it made"
        )
    return (
        f"{stances} distinct stances over {len(obs)} scored days, long {long_frac:.0%} "
        f"of the days it took a side; action-gap {gap:+.1%} summed over its sided days "
        f"(each BUY day scored by the next day's move, each SELL day by the move it "
        f"stepped aside from) while the instrument moved {window:+.1%} over the same "
        f"window (hit rate {hit:.0%} vs {base_hit:.0%} for always-long)"
    )


@dataclass
class DistillOutcome:

    section: str
    status: str

    @property
    def failed(self) -> bool:
        return self.status != DISTILL_OK


def compose_episode(
    reasoning_samples: Optional[list] = None,
    outcome_phrase: str = "",
    descriptor: str = "",
) -> str:
    lines: list = []
    samples = [str(s).strip() for s in (reasoning_samples or []) if str(s).strip()]
    if samples:
        lines.append("What it argued:")
        for s in samples[-3:]:
            lines.append(f"  - {s[:300]}")
    if outcome_phrase:
        lines.append(f"How the calls turned out: {outcome_phrase}.")
    if descriptor:
        lines.append(f"Outcome signal: {descriptor}")
    return "\n".join(lines)


def _distill_call(
    prompt: "str | list", agent_name: str, fallback: str,
    model_name: str, model_provider: str,
) -> DistillOutcome:
    try:
        from llm.functions import call_llm

        d = call_llm(
            prompt=prompt,
            model_name=model_name,
            model_provider=model_provider,
            tags=["section"],
            agent_name=agent_name,
        ).tags
        text = (d.get("section") or "").strip()
        if not text or "Error in analysis, using default" in text:
            return DistillOutcome(fallback, DISTILL_LLM_FAILED)
        return DistillOutcome(text[:1200], DISTILL_OK)
    except Exception:
        return DistillOutcome(fallback, DISTILL_EXCEPTION)


def distill_actor_bucket(
    actor_name: str,
    ticker: str,
    scope_label: str,
    source_texts: list,
    today: str,
    model_name: str = _DEFAULT_MODEL,
    model_provider: str = "OpenAI",
) -> DistillOutcome:
    joined = "\n\n".join(
        f"[{i + 1}] {str(t).strip()}" for i, t in enumerate(source_texts) if str(t).strip()
    )
    if not joined.strip():
        return DistillOutcome("", DISTILL_LLM_FAILED)
    from utils.prompt_loader import render_messages
    prompt = render_messages(
        "finpanel/bucket_distiller",
        ticker=ticker,
        scope_label=scope_label,
        source_texts=joined,
    )
    return _distill_call(
        prompt, f"bucket_distiller:{actor_name}:{ticker}", "", model_name, model_provider
    )
