from __future__ import annotations

import os

from backtest.engine import DayContext, action_from_direction
from finpanel.panel import PANEL_SIZE, ensure_panel
from finpanel.single_stock_workflow import SingleStockWorkflow, _reflection_store_path
from finpanel.state import AgentState
from llm.strategy import LLMStrategy


class FinPanelStrategy(LLMStrategy):

    name = "finpanel"
    warmup_days = 420

    def prepare(self, close, ohlc=None) -> None:
        return None

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        model_name: str,
        model_provider: str,
        experiment_settings: dict,
        execution_protocol: dict,
        offline_mode: bool,
        initial_nav: float,
        show_reasoning: bool = False,
        ablation_stage: str = "a4",
        execution_profile: str = "base",
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.model_name = model_name
        self.model_provider = model_provider
        self.experiment_settings = experiment_settings or {}
        self.execution_protocol = execution_protocol
        self.offline_mode = offline_mode
        self.initial_nav = float(initial_nav)
        self.show_reasoning = show_reasoning
        self.ablation_stage = ablation_stage
        self.execution_profile = execution_profile
        self.run_id: str | None = None
        self.run_dir: str | None = None
        self.persistent_signal_history: list = []
        self.persistent_last_direction: str | None = None
        self.persistent_decision_history: list = []
        self.panel = None
        self.finpanel_coordination_log: list = []
        self.finpanel_distill_stats: dict[str, int] = {"distilled": 0, "failed": 0}

    def on_evaluation_start(self) -> None:
        self.persistent_decision_history = []
        self.persistent_last_direction = None

    def on_run_start(self, run_id: str, run_dir=None) -> None:
        self.run_id = run_id
        self.run_dir = str(run_dir) if run_dir is not None else None
        try:
            _p = _reflection_store_path(self.run_dir, run_id)
            if os.path.exists(_p):
                os.remove(_p)
        except OSError:
            pass
        es = self.experiment_settings
        roster = [n for n in str(es.get("panel_roster") or "").split(",") if n.strip()]
        blind = bool(es.get("panel_blind", False))
        self.panel = ensure_panel(
            run_dir=self.run_dir, ticker=self.ticker,
            size=int(es.get("panel_size", PANEL_SIZE)),
            train_start="" if blind else str(es.get("train_start") or self.start_date),
            train_end="" if blind else str(es.get("train_end") or ""),
            model_name=self.model_name, model_provider=self.model_provider,
            roster=roster or None,
        )
        es["panel_members"] = list(self.panel.slugs)
        print(f"🎯 panel for {self.ticker}: {', '.join(self.panel.slugs)} "
              f"(information={self.panel.information})")

    def on_run_end(self) -> dict:
        from backtest.artifacts import finpanel_mechanism_surface

        return {
            "ablation_stage": self.ablation_stage,
            "execution_profile": self.execution_profile,
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "experiment_settings": dict(self.experiment_settings),
            "execution_protocol": dict(self.execution_protocol),
            "metrics_extra": {
                "finpanel_mechanism_surface": finpanel_mechanism_surface(
                    self.finpanel_coordination_log),
                "distill_stats": dict(self.finpanel_distill_stats),
            },
        }

    def decide(self, ctx: DayContext) -> float:
        es = self.experiment_settings
        portfolio = {
            "cash": float(ctx.cash),
            "positions": {ctx.ticker: {"shares": float(ctx.position_shares), "cost_basis": 0.0, "margin_used": 0.0}},
            "margin_requirement": 0.5,
            "realized_gains": {},
        }
        day_str = ctx.date.strftime("%Y-%m-%d")
        initial_state = AgentState(
            messages=[],
            data={
                "tickers": [ctx.ticker],
                "portfolio": portfolio,
                "start_date": self.start_date,
                "end_date": day_str,
                "phase": getattr(ctx, "phase", "test"),
                "analyst_signals": {},
                "persistent_signal_history": self.persistent_signal_history,
                "persistent_last_direction": self.persistent_last_direction,
                "persistent_decision_history": self.persistent_decision_history,
                "initial_nav": self.initial_nav,
                "use_fundamentals": bool(es.get("use_fundamentals", False)),
                "use_news_selection": bool(es.get("use_news_selection", False)),
            },
            metadata={
                "show_reasoning": self.show_reasoning,
                "model_name": self.model_name,
                "model_provider": self.model_provider,
                "experiment_settings": es,
                "offline_mode": self.offline_mode,
                "run_id": self.run_id,
                "backtest_run_id": self.run_id,
                "run_dir": self.run_dir,
            },
        )
        workflow = SingleStockWorkflow(
            model_name=self.model_name,
            model_provider=self.model_provider,
            collect_reasoning=bool(es.get("collect_reasoning", False)),
            panel=self.panel,
        )
        final_state = workflow.run(initial_state)
        d = final_state["data"]
        final_dir = d.get("_manager_final_dir", "HOLD")

        self.persistent_signal_history = d.get("persistent_signal_history", self.persistent_signal_history)
        self.persistent_last_direction = d.get("persistent_last_direction", self.persistent_last_direction)
        self.persistent_decision_history = d.get("persistent_decision_history", self.persistent_decision_history)
        self.finpanel_coordination_log.extend(d.get("coordination_log", []) or [])
        for k, v in (d.get("distill_stats") or {}).items():
            self.finpanel_distill_stats[k] = self.finpanel_distill_stats.get(k, 0) + int(v or 0)

        return action_from_direction(final_dir, ctx.current_weight)
