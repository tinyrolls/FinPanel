from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import BaseModel, Field, field_validator

SINGLE_PASS_CHARS = 350_000
FILING_LABEL = {"filing_k": "10-K (annual)", "filing_q": "10-Q (quarterly)"}


class _StrCoercingModel(BaseModel):

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return "; ".join(f"{k}: {val}" for k, val in v.items())
        if isinstance(v, (list, tuple)):
            return "; ".join(str(x) for x in v)
        return str(v)


class FilingBrief(_StrCoercingModel):

    fiscal_period: str = Field(description="The fiscal period this filing covers, e.g. 'FY2022' or 'Q2 2023'. Empty string if unclear.")
    key_financials: str = Field(description="Concrete numbers: revenue and YoY growth, gross/operating margin, net income/EPS, operating cash flow / FCF, key balance-sheet items. Cite the actual figures.")
    business_narrative: str = Field(description="What drove results: segment performance, demand, pricing, costs — in plain prose.")
    outlook_guidance: str = Field(description="Forward guidance / management outlook stated in the filing. Empty string if none given.")
    key_risks: str = Field(description="The most material risk factors emphasised in this filing.")
    notable_changes: str = Field(description="Anything notably changed vs prior periods the filing discusses (accounting, segments, litigation, capital structure). Empty string if none.")

    def render(self) -> str:
        parts = []
        if self.fiscal_period:
            parts.append(f"Period: {self.fiscal_period}")
        parts.append(f"Key financials: {self.key_financials}")
        parts.append(f"Business: {self.business_narrative}")
        if self.outlook_guidance:
            parts.append(f"Outlook/guidance: {self.outlook_guidance}")
        parts.append(f"Key risks: {self.key_risks}")
        if self.notable_changes:
            parts.append(f"Notable changes: {self.notable_changes}")
        return "\n".join(parts)


def _key(ticker: str, ftype: str, fdate: str) -> str:
    return f"{ticker}|{ftype}|{fdate}"


def _iter_filings(data: dict) -> Iterator[tuple[str, str, str, str]]:
    for day, day_data in data.items():
        fdate = day.isoformat() if hasattr(day, "isoformat") else str(day)
        for ftype in ("filing_k", "filing_q"):
            for ticker, text in (day_data.get(ftype, {}) or {}).items():
                if text:
                    yield ticker, ftype, fdate, text


def _chunks(text: str, size: int) -> Iterator[str]:
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _summarise_one(text: str, ticker: str, ftype: str, model: str, provider: str) -> FilingBrief:
    from llm.functions import call_llm

    label = FILING_LABEL.get(ftype, ftype)
    if len(text) <= SINGLE_PASS_CHARS:
        body = text
    else:
        notes = []
        for j, ch in enumerate(_chunks(text, SINGLE_PASS_CHARS)):
            d = call_llm(
                prompt=[
                    {"role": "system", "content": f"You are a financial analyst reading part {j + 1} of a long SEC {label} for {ticker}."},
                    {"role": "user", "content": f"Extract dense factual notes (all figures, guidance, risks) from this excerpt. Be terse. Wrap your notes in a single tag:\n<notes>…</notes>\n\n{ch}"},
                ],
                model_name=model, model_provider=provider, tags=["notes"],
                agent_name=f"filing_notes_{ticker}_{j}",
            ).tags
            notes.append(d.get("notes", "") or "")
        body = "\n\n".join(notes)

    field_names = list(FilingBrief.model_fields)
    d = call_llm(
        prompt=[
            {"role": "system", "content": f"You are a financial analyst. Produce a compact, numbers-rich brief of this SEC {label} for {ticker}. Prefer concrete figures over adjectives. Base it ONLY on the provided text."},
            {"role": "user", "content": (
                "Write each of the following fields as free-form text wrapped in its own tag, IN THIS ORDER:\n"
                "<fiscal_period>the period covered</fiscal_period>\n"
                "<key_financials>concrete revenue/margin/EPS/cash figures</key_financials>\n"
                "<business_narrative>what drove results</business_narrative>\n"
                "<outlook_guidance>stated guidance (empty if none)</outlook_guidance>\n"
                "<key_risks>material risks</key_risks>\n"
                "<notable_changes>changes vs prior (empty if none)</notable_changes>\n\n"
                f"{body}"
            )},
        ],
        model_name=model, model_provider=provider, tags=field_names,
        agent_name=f"filing_brief_{ticker}",
    ).tags
    brief = FilingBrief(**{k: d.get(k, "") for k in field_names})
    if not brief.key_financials.strip() and not brief.business_narrative.strip():
        raise RuntimeError("LLM returned no parseable brief content — not caching")
    return brief


def _load_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add(_key(r["ticker"], r["filing_type"], r["filing_date"]))
    return done


def ensure_briefs(
    data: dict | str | Path,
    briefs_path: str | Path,
    *,
    model: str = "gpt-4o-mini-2024-07-18",
    provider: str = "OpenAI",
    tickers: Iterable[str] | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> int:
    if isinstance(data, (str, Path)):
        with Path(data).open("rb") as fh:
            data = pickle.load(fh)

    keep = {t.upper() for t in tickers} if tickers else None
    out_path = Path(briefs_path)
    done = _load_done(out_path)

    todo = [
        (tk, ft, fd, tx)
        for tk, ft, fd, tx in _iter_filings(data)
        if (keep is None or tk in keep) and _key(tk, ft, fd) not in done
    ]
    if limit:
        todo = todo[:limit]

    if not todo:
        if verbose:
            print(f"✓ fundamentals briefs up to date ({len(done)} cached) — skipping summarisation")
        return 0

    if verbose:
        print(f"📝 summarising {len(todo)} filings → {out_path} (model={model}); {len(done)} already cached")

    n_ok = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for ticker, ftype, fdate, text in todo:
            try:
                brief = _summarise_one(text, ticker, ftype, model, provider)
            except Exception as exc:
                if verbose:
                    print(f"   ⚠️ {ticker} {ftype} {fdate} failed: {type(exc).__name__}: {exc}")
                continue
            fh.write(json.dumps({
                "ticker": ticker, "filing_type": ftype, "filing_date": fdate,
                "summarizer_model": model, "source_chars": len(text),
                "generated_at": dt.date.today().isoformat(), "brief": brief.model_dump(),
            }, ensure_ascii=False) + "\n")
            fh.flush()
            n_ok += 1
            if verbose:
                print(f"   ✓ {ticker} {ftype} {fdate} ({len(text)} chars)")

    if verbose:
        print(f"💾 wrote {n_ok} new briefs → {out_path}")
    return n_ok


class FundamentalsBriefStore:

    def __init__(self, by_ticker: dict[str, list[tuple[dt.date, FilingBrief]]]):
        self._by_ticker = by_ticker

    @classmethod
    def load(cls, briefs_path: str | Path) -> "FundamentalsBriefStore":
        by_ticker: dict[str, list[tuple[dt.date, FilingBrief]]] = {}
        p = Path(briefs_path)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    fdate = dt.date.fromisoformat(r["filing_date"])
                    by_ticker.setdefault(r["ticker"], []).append((fdate, FilingBrief(**r["brief"])))
        for rows in by_ticker.values():
            rows.sort(key=lambda x: x[0])
        return cls(by_ticker)

    def get_brief(self, ticker: str, t_now: dt.date | str) -> FilingBrief | None:
        if isinstance(t_now, str):
            t_now = dt.date.fromisoformat(t_now)
        cutoff = t_now - dt.timedelta(days=1)
        latest = None
        for fdate, brief in self._by_ticker.get(ticker, ()):
            if fdate <= cutoff:
                latest = brief
            else:
                break
        return latest
