from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ActorConfig:

    slug: str
    actor_name: str

    def build_messages(
        self,
        ticker: str,
        metrics_data: str,
        news_section: str = "",
        fundamentals_section: str = "",
        technical_section: str = "",
        date: str = "",
    ) -> list:
        from utils.prompt_loader import render_messages

        return render_messages(
            "finpanel/actor",
            actor_name=self.actor_name,
            ticker=ticker,
            date=date or "today",
            metrics_data=metrics_data,
            news_section=news_section,
            fundamentals_section=fundamentals_section,
            technical_section=technical_section,
        )


ACTOR_CONFIGS: Dict[str, ActorConfig] = {}


def register_actors(configs: Dict[str, ActorConfig]) -> None:
    ACTOR_CONFIGS.update(configs)


def get_actor_config(slug: str) -> ActorConfig:
    cfg = ACTOR_CONFIGS.get(slug)
    if cfg is None:
        raise ValueError(
            f"no actor configured for slug {slug!r}; "
            f"available: {sorted(ACTOR_CONFIGS.keys())}"
        )
    return cfg
