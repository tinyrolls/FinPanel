from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from finpanel.actors.actor_configs import ActorConfig, register_actors

PANEL_SIZE = 6

@dataclass(frozen=True)
class Panel:

    name: str
    ticker: str
    slugs: Tuple[str, ...]
    information: str

    def __len__(self) -> int:
        return len(self.slugs)


def _slugify(name: str, taken: set) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_") or "actor"
    slug = base
    i = 2
    while slug in taken:
        slug = f"{base}_{i}"
        i += 1
    return slug


def instrument_character(ticker: str, start: str, end: str) -> str:
    from data import api

    df = api.get_price_data(ticker, start, end)
    if df is None or df.empty:
        return ""
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    close = d["close"].astype(float)
    ret = close.pct_change().dropna()
    if len(ret) < 30:
        return ""

    vol_ann = float(ret.std() * (252 ** 0.5) * 100)
    mean_abs = float(ret.abs().mean() * 100)
    big_days = float((ret.abs() >= 0.03).mean() * 100)
    autocorr = float(ret.autocorr(lag=1)) if len(ret) > 2 else 0.0
    rng = float(((d["high"].astype(float) - d["low"].astype(float)) / close).mean() * 100)
    gap = float((d["open"].astype(float) / close.shift(1) - 1.0).abs().mean() * 100)
    vol_cv = 0.0
    if "volume" in d and float(d["volume"].astype(float).mean() or 0) > 0:
        v = d["volume"].astype(float)
        vol_cv = float(v.std() / v.mean())

    persistence = ("moves tend to follow through day to day" if autocorr > 0.03
                   else "moves tend to reverse day to day" if autocorr < -0.03
                   else "day-to-day moves are close to independent")
    return "\n".join([
        "- price and volume only; no fundamentals",
        f"- annualised volatility {vol_ann:.0f}%; the average day moves "
        f"{mean_abs:.1f}% in absolute terms",
        f"- {big_days:.0f}% of days move 3% or more",
        f"- average intraday range {rng:.1f}% of price; average overnight gap "
        f"{gap:.1f}%",
        f"- 1-day return autocorrelation {autocorr:+.2f} — {persistence}",
        f"- volume is uneven: its standard deviation is {vol_cv:.2f}x its mean",
    ])


def ensure_panel(
    run_dir: Optional[str],
    ticker: str,
    size: int = PANEL_SIZE,
    train_start: str = "",
    train_end: str = "",
    model_name: str = "gpt-oss-120b",
    model_provider: str = "OpenAI",
    roster: Optional[Sequence[str]] = None,
) -> Panel:
    path = Path(run_dir) / "panel.yaml" if run_dir else None
    if path is not None and path.exists():
        manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return _panel_from_manifest(manifest)

    if roster:
        names = [str(n).strip() for n in roster if str(n).strip()]
        if len(names) != size:
            raise RuntimeError(f"fixed roster has {len(names)} names but panel_size={size}: {names}")
        actors, taken = [], set()
        for n in names:
            slug = _slugify(n, taken)
            taken.add(slug)
            actors.append({"slug": slug, "actor_name": n})
        manifest = {"name": f"panel{size}", "ticker": str(ticker).upper(), "information": "fixed",
                    "size": len(actors), "actors": actors}
        panel = _panel_from_manifest(manifest)
    else:
        brief, window = "", ""
        if train_start and train_end and train_start < train_end:
            brief = instrument_character(ticker, train_start, train_end)
            window = f"daily bars {train_start} to {train_end}"
        panel, manifest = recruit_panel(
            ticker=ticker, size=size, instrument_brief=brief, window=window,
            model_name=model_name, model_provider=model_provider,
        )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, width=88),
            encoding="utf-8",
        )
    return panel


def recruit_panel(
    ticker: str,
    size: int = PANEL_SIZE,
    instrument_brief: str = "",
    window: str = "",
    model_name: str = "gpt-oss-120b",
    model_provider: str = "OpenAI",
) -> Tuple[Panel, Dict[str, any]]:
    information = "cond" if (instrument_brief or "").strip() else "blind"
    instrument_block = (
        f"How {ticker} has traded, {window or 'over the training window'}:\n"
        f"{instrument_brief.strip()}\n" if information == "cond" else
        f"You are told nothing about how {ticker} trades. Staff it from trading\n"
        "practice alone.\n")

    from utils.prompt_loader import render_messages
    from llm.functions import call_llm

    prompt = render_messages(
        "finpanel/recruiter",
        ticker=str(ticker).upper(), size=size, instrument_block=instrument_block,
    )
    resp = call_llm(
        prompt=prompt, model_name=model_name, model_provider=model_provider,
        agent_name="recruiter",
    )
    entries = _parse_hires((resp.content or "").strip(), size)

    actors, taken = [], set()
    for e in entries:
        slug = _slugify(e["actor_name"], taken)
        taken.add(slug)
        row = {"slug": slug, "actor_name": e["actor_name"]}
        if e.get("fit"):
            row["fit"] = e["fit"]
        actors.append(row)

    manifest = {
        "name": f"panel{size}",
        "ticker": str(ticker).upper(),
        "information": information,
        "size": len(actors),
        "model_name": model_name,
        "prompt_sha256": hashlib.sha256("\n---\n".join(m["content"] for m in prompt).encode("utf-8")).hexdigest(),
        "instrument_brief": instrument_brief,
        "window": window,
        "actors": actors,
    }
    return _panel_from_manifest(manifest), manifest


def _panel_from_manifest(manifest: Dict[str, any]) -> Panel:
    actors = manifest.get("actors") or []
    if not actors:
        raise RuntimeError(f"panel manifest for {manifest.get('ticker')} lists no actors")
    slugs: List[str] = []
    configs: Dict[str, ActorConfig] = {}
    for a in actors:
        slug = str(a.get("slug") or "").strip()
        actor_name = str(a.get("actor_name") or "").strip()
        if not slug or not actor_name:
            raise RuntimeError(f"panel manifest for {manifest.get('ticker')} has an actor without slug/name")
        slugs.append(slug)
        configs[slug] = ActorConfig(slug=slug, actor_name=actor_name)
    register_actors(configs)
    return Panel(
        name=str(manifest.get("name") or f"panel{len(slugs)}"),
        ticker=str(manifest.get("ticker", "")).upper(),
        slugs=tuple(slugs),
        information=str(manifest.get("information", "cond")),
    )


_HIRE_TAG_RE = re.compile(
    r"<(fit|name)\s*>(.*?)(?=</\s*\w*\s*>|<(?:fit|name)\s*>|\Z)", re.S | re.I)


def _parse_hires(body: str, size: int) -> List[dict]:
    out: List[dict] = []
    fit = ""
    for tag, text in _HIRE_TAG_RE.findall(body):
        text = text.strip()
        if tag.lower() == "fit":
            fit = text
            continue
        if text:
            out.append({"actor_name": text, "fit": fit})
        fit = ""
    if not out:
        raise RuntimeError("recruiter returned no <name> entries")
    if len(out) > size:
        print(f"⚠️ recruiter returned {len(out)} hires for a panel of {size} — keeping the first {size}")
        out = out[:size]
    if len(out) != size:
        raise RuntimeError(
            f"recruiter returned {len(out)} usable hires, expected {size} — re-run rather "
            "than padding: a short panel is not the one the experiment was set up around"
        )
    return out
