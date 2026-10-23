from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from backtest.engine import action_from_direction

MODEL = "gpt-4o-mini-2024-07-18"
PROVIDER = "OpenAI"

_ROOT = Path(__file__).resolve().parents[2]
NEWS_PATH = _ROOT / "data" / "news_selection.jsonl"
BRIEF_PATH = _ROOT / "data" / "fundamentals_briefs.jsonl"


class LLMStrategy:

    name: str = "llm_strategy"
    warmup_days: int = 80
    model_name: str = MODEL
    model_provider: str = PROVIDER

    def prepare(self, close, ohlc=None) -> None:
        self._news = self._brief = None
        try:
            from data.news_retrieval import NewsSelectionStore

            self._news = NewsSelectionStore.load(str(NEWS_PATH))
        except Exception:
            self._news = None
        try:
            from data.fundamentals_briefs import FundamentalsBriefStore

            self._brief = FundamentalsBriefStore.load(str(BRIEF_PATH))
        except Exception:
            self._brief = None
        self.reset_memory()

    def reset_memory(self) -> None:
        pass

    def on_evaluation_start(self) -> None:
        pass

    def on_run_start(self, run_id: str, run_dir=None) -> None:
        pass

    def on_run_end(self) -> dict:
        return {}

    def decide(self, ctx) -> float:
        raise NotImplementedError

    def news(self, ctx) -> List[str]:
        store = getattr(self, "_news", None)
        if not store:
            return []
        return list(store.get(ctx.ticker, self.date_str(ctx)) or [])

    def brief(self, ctx) -> str:
        store = getattr(self, "_brief", None)
        if not store:
            return ""
        b = store.get_brief(ctx.ticker, self.date_str(ctx))
        return b.render() if b else ""

    def technical(self, ctx) -> str:
        try:
            from data.technical_features import compute_technical_brief

            return compute_technical_brief(ctx.ticker, self.date_str(ctx)) or ""
        except Exception:
            return ""

    def call(self, prompt: Any, tags: List[str], agent: str, validate=None) -> Dict[str, str]:
        from llm.functions import call_llm

        r = call_llm(prompt=prompt, model_name=self.model_name, model_provider=self.model_provider,
                     tags=tags, agent_name=agent, validate=validate)
        return dict(r.tags)

    def decide_action(self, prompt: Any, agent: str, extra_tags: List[str] = ("reasoning",),
                      choices: List[str] = ("BUY", "SELL", "HOLD")) -> tuple:
        from llm.tag_parse import choice_validator, normalize_choice

        tags = self.call(
            prompt, ["action", *extra_tags], agent,
            validate=choice_validator("action", list(choices)),
        )
        action = normalize_choice(tags.get("action") or "", list(choices), "HOLD")
        return action, tags

    @staticmethod
    def date_str(ctx) -> str:
        d = ctx.date
        return d.date().isoformat() if hasattr(d, "date") else str(d)[:10]

    def decision_to_weight(self, direction: str, ctx) -> float:
        return action_from_direction(direction, ctx.current_weight)
