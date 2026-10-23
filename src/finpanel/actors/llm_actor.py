from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from finpanel.state import AgentState
from llm.functions import call_llm
from llm.tag_parse import extract_choice, strip_tags
from finpanel.actors.actor_configs import get_actor_config


def filter_state_for_actor(state: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    filtered_state = {
        "data": dict(state.get("data", {})),
        "metadata": dict(state.get("metadata", {})),
    }
    metrics = state.get("data", {}).get("metrics_by_ticker", {}).get(ticker, {})
    filtered_state["data"]["metrics_by_ticker"] = {ticker: metrics} if metrics else {}
    filtered_state["data"]["tickers"] = [ticker]
    return filtered_state


class DiscreteRecommendation(BaseModel):

    ticker: str
    reasoning: str = Field(description="Your reasoning for the call.")
    direction: str = Field(
        description="BUY, SELL, or HOLD.",
        json_schema_extra={"enum": ["BUY", "SELL", "HOLD"]},
    )


class DiscreteActorOutput(BaseModel):
    recommendations: List[DiscreteRecommendation] = []


class LLMActor:

    def __init__(
        self,
        slug: str,
        model_name: str | None = None,
        model_provider: str | None = None,
    ):
        self.slug = slug
        self.config = get_actor_config(slug)
        self.model_name = model_name
        self.model_provider = model_provider

    @property
    def agent_name(self) -> str:
        return self.slug

    def run(self, state: AgentState, ticker: str | None = None) -> Dict[str, Any]:
        tickers = [ticker] if ticker else state["data"]["tickers"]

        metadata = state.get("metadata", {})
        model_name = self.model_name or metadata.get("model_name", "gpt-4o-mini-2024-07-18")
        model_provider = self.model_provider or metadata.get("model_provider", "OpenAI")

        print(
            f"🤖 LLM {self.config.actor_name} analyzing {tickers} "
            f"(slug={self.slug})"
        )


        _day = state["data"]["_day"]

        def _brief(name: str, t: str) -> str:
            return _day.get("pre", name, ticker=t) or ""

        recommendations: List[Any] = []
        for t in tickers:
            filtered_state = filter_state_for_actor(state, t)
            metrics = filtered_state["data"].get("metrics_by_ticker", {}).get(t, {})
            metrics_json = self._format_metrics_for_prompt(metrics)

            fundamentals_text = _brief("fundamentals", t)
            fundamentals_section = ""
            if fundamentals_text:
                fundamentals_section = (
                    f"### Fundamentals for {t} (latest 10-K/10-Q, dated <= decision day)\n"
                    f"{fundamentals_text}"
                )

            technical_text = _brief("technical", t)
            technical_section = ""
            if technical_text:
                technical_section = (
                    f"### Technical indicators for {t} (price-derived, dated <= decision day)\n"
                    f"{technical_text}"
                )

            news_text = _brief("news", t)
            news_section = ""
            if news_text:
                news_section = (
                    f"### Recent news for {t} (dated <= decision day)\n"
                    f"{news_text}"
                )

            prompt = self.config.build_messages(
                ticker=t,
                metrics_data=metrics_json,
                news_section=news_section,
                fundamentals_section=fundamentals_section,
                technical_section=technical_section,
                date=str(state["data"].get("end_date", "")),
            )

            try:
                resp = call_llm(
                    prompt=prompt,
                    model_name=model_name,
                    model_provider=model_provider,
                    tags=["action", "reasoning"],
                    agent_name=f"{self.slug}_{t}",
                )
                raw_action = (resp.tags.get("action") or "").strip()
                reasoning = (resp.tags.get("reasoning") or "").strip() or strip_tags(resp.content, ["action", "reasoning"])
                if not reasoning and not raw_action:
                    raise RuntimeError("LLM returned no parseable output after retries — recording abstain")
                direction = extract_choice(resp.content, ["BUY", "SELL", "HOLD"], "HOLD")
                rec = DiscreteRecommendation(
                    ticker=t,
                    reasoning=reasoning or "(no reasoning parsed)",
                    direction=direction,
                )
                recommendations.append(rec)

                if metadata.get("show_reasoning", False):
                    print(
                        f"  {rec.ticker}: {rec.reasoning[:120]}... "
                        f"(dir={rec.direction})"
                    )
            except Exception as exc:
                print(f"❌ {self.config.actor_name} LLM call failed for {t}: {exc}")
                recommendations.append(DiscreteRecommendation(
                    ticker=t, reasoning=f"Error: {exc}",
                    direction="HOLD",
                ))

        output = DiscreteActorOutput(recommendations=[
            r if isinstance(r, DiscreteRecommendation) else DiscreteRecommendation(
                ticker=getattr(r, "ticker", ""),
                reasoning=getattr(r, "reasoning", ""),
                direction="HOLD",
            )
            for r in recommendations
        ])

        return {
            "output": output,
            "agent_name": self.slug,
            "enhanced_analysis": True,
            "llm_based": True,
        }

    @staticmethod
    def _format_metrics_for_prompt(metrics: Dict[str, Any]) -> str:
        cleaned: Dict[str, Any] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, (dict, list)) and not value:
                continue
            cleaned[key] = value
        try:
            return json.dumps(cleaned, indent=2, default=str)
        except Exception:
            return json.dumps({"error": "could not serialize metrics"})


_agent_cache: Dict[tuple, LLMActor] = {}


def get_llm_actor(
    slug: str,
    model_name: str = "gpt-4o-mini-2024-07-18",
    model_provider: str = "OpenAI",
) -> LLMActor:
    cache_key = (slug, model_name, model_provider)
    if cache_key not in _agent_cache:
        _agent_cache[cache_key] = LLMActor(
            slug=slug, model_name=model_name, model_provider=model_provider
        )
    return _agent_cache[cache_key]


def run_llm_actor(
    state: AgentState,
    slug: str,
    ticker: str | None = None,
    model_name: str | None = None,
    model_provider: str | None = None,
) -> Dict[str, Any]:
    agent = get_llm_actor(
        slug,
        model_name or "gpt-4o-mini-2024-07-18",
        model_provider or "OpenAI",
    )
    return agent.run(state, ticker)
