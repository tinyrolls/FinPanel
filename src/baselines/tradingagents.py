from __future__ import annotations

import os

from llm.strategy import LLMStrategy
from utils.prompt_loader import render_prompt

_ACTIONS = {"BUY", "SELL", "HOLD"}


def _rounds(env_name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(env_name, "").strip()))
    except (TypeError, ValueError):
        return default


class TradingAgentsStrategy(LLMStrategy):

    name = "tradingagents"
    warmup_days = 80

    MAX_DEBATE_ROUNDS = 1
    MAX_RISK_ROUNDS = 1


    def decide(self, ctx) -> float:
        if getattr(ctx, "phase", "test") == "warmup":
            return 0.0
        date = self.date_str(ctx)
        tkr = ctx.ticker
        debate_rounds = _rounds("TA_DEBATE_ROUNDS", self.MAX_DEBATE_ROUNDS)
        risk_rounds = _rounds("TA_RISK_ROUNDS", self.MAX_RISK_ROUNDS)

        news = self.news(ctx)
        news_txt = "\n".join(f"- {n.strip()}" for n in news[:8]) if news else "No fresh news available."
        tech = self.technical(ctx) or "Insufficient price history for technicals."
        fund = self.brief(ctx) or "No fundamentals brief available."

        reports = {
            "market_report": self._agent("tradingagents/market_analyst", f"ta_market_{tkr}", ticker=tkr, date=date, technical=tech),
            "sentiment_report": self._agent("tradingagents/sentiment_analyst", f"ta_sentiment_{tkr}", ticker=tkr, date=date, news=news_txt),
            "news_report": self._agent("tradingagents/news_analyst", f"ta_news_{tkr}", ticker=tkr, date=date, news=news_txt),
            "fundamentals_report": self._agent("tradingagents/fundamentals_analyst", f"ta_fundamentals_{tkr}", ticker=tkr, date=date, fundamentals=fund),
        }

        history, current, count = "", "", 0
        while count < 2 * debate_rounds:
            side = "bull" if count % 2 == 0 else "bear"
            arg = self._agent(
                f"tradingagents/{side}_researcher", f"ta_{side}_{tkr}",
                ticker=tkr, history=history.strip() or "(debate opens)",
                current_response=current or "(none yet)", **reports,
            )
            history += f"\n{'Bull' if side == 'bull' else 'Bear'} Analyst: {arg}"
            current = arg
            count += 1

        investment_plan = self._agent("tradingagents/research_manager", f"ta_resmgr_{tkr}", ticker=tkr, history=history.strip())

        pos = self._position_context(ctx)
        trader_action, trader_reasoning = self._decision(
            "tradingagents/trader", f"ta_trader_{tkr}",
            ticker=tkr, date=date, investment_plan=investment_plan, position_context=pos,
        )
        trader_plan = trader_reasoning or investment_plan

        risk_roles = ("aggressive", "conservative", "neutral")
        risk_hist, risk_cur, rc = "", "", 0
        while rc < 3 * risk_rounds:
            role = risk_roles[rc % 3]
            arg = self._agent(
                f"tradingagents/risk_{role}", f"ta_risk_{role}_{tkr}",
                ticker=tkr, trader_action=trader_action, trader_plan=trader_plan,
                history=risk_hist.strip() or "(debate opens)", current_response=risk_cur or "(none yet)",
            )
            risk_hist += f"\n{role.capitalize()} Analyst: {arg}"
            risk_cur = arg
            rc += 1

        final_dir, _ = self._decision(
            "tradingagents/portfolio_manager", f"ta_pm_{tkr}",
            ticker=tkr, trader_action=trader_action, trader_plan=trader_plan,
            risk_history=risk_hist.strip(), position_context=pos,
        )
        return self.decision_to_weight(final_dir, ctx)

    def _agent(self, prompt_name: str, agent: str, **kw) -> str:
        tags = self.call(render_prompt(prompt_name, **kw), ["report", "argument", "plan"], agent)
        for t in ("report", "argument", "plan"):
            v = (tags.get(t) or "").strip()
            if v:
                return v
        return ""

    def _decision(self, prompt_name: str, agent: str, **kw):
        action, tags = self.decide_action(render_prompt(prompt_name, **kw), agent, choices=_ACTIONS)
        return action, (tags.get("reasoning") or "").strip()

    def _position_context(self, ctx) -> str:
        w = float(ctx.current_weight)
        nav = float(ctx.nav_pre)
        if w > 0.01:
            return f"You currently hold a LONG position (~{w * 100:.0f}% of NAV) in {ctx.ticker}; NAV ~${nav:,.0f}."
        return f"You are currently FLAT (no position) in {ctx.ticker}; NAV ~${nav:,.0f}."
