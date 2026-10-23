from __future__ import annotations

import hashlib
import os
from collections import deque
from typing import Any, Dict, List, Optional

from llm.strategy import LLMStrategy
from utils.prompt_loader import render_prompt

TRADER_PREFERENCE_AGGRESSIVE = (
    "As an aggressive trader, you are willing to take on higher risks in order to seek greater potential "
    "profits. You may engage in frequent buying and selling, attempting to capture short-term price "
    "fluctuations, and you might allocate a larger portion of your capital to that stock."
)

_MOMENTUM_EXPLANATION = (
    'The information below provides a summary of stock price fluctuations over the previous few days, which is the "Momentum" of a stock.\n'
    "        It reflects the trend of a stock.\n"
    "        Momentum is based on the idea that securities that have performed well in the past will continue to perform well, and conversely, securities that have performed poorly will continue to perform poorly."
)
_FEEDBACK_WINDOW = 7

def _date_str(ctx) -> str:
    return LLMStrategy.date_str(ctx)


def _momentum(ctx, window: int = 3) -> int:
    h = ctx.history
    if h is None or len(h) <= window:
        return 0
    r = float(h.iloc[-1]) / float(h.iloc[-1 - window]) - 1.0
    return 1 if r > 1e-9 else (-1 if r < -1e-9 else 0)


class FinMemEquiv(LLMStrategy):

    name = "finmem"

    def prepare(self, close, ohlc=None) -> None:
        super().prepare(close, ohlc)
        self._close = close

    def reset_memory(self) -> None:
        self._brain: Optional[Any] = None
        self._eid = os.environ.get("FINPANEL_EXPERIMENT_ID", "finmem_default")
        self._filings_seen: set = set()
        self._pending: deque = deque()

    def on_evaluation_start(self) -> None:
        self._pending.clear()

    @staticmethod
    def _seed(ticker: str) -> int:
        return int(hashlib.sha256(ticker.encode("utf-8")).hexdigest()[:8], 16)

    def _ensure_brain(self, ctx):
        if self._brain is None:
            from baselines.finmem_memory import FinMemBrain

            self._brain = FinMemBrain(self._eid, ctx.ticker, seed=self._seed(ctx.ticker))
        return self._brain

    def _next_day_delta(self, ctx) -> float:
        try:
            idx = self._close.index.get_loc(ctx.date)
            if isinstance(idx, int) and idx + 1 < len(self._close):
                return float(self._close.iloc[idx + 1]) - float(self._close.iloc[idx])
        except Exception:
            pass
        return 0.0

    def _ingest(self, brain, ctx, date: str) -> None:
        for n in self.news(ctx):
            brain.add("short", n, date)
        brief = self.brief(ctx)
        if brief and brief not in self._filings_seen:
            brain.add("mid", brief, date)
            self._filings_seen.add(brief)

    def _process_feedback(self, brain, ctx) -> None:
        still: deque = deque()
        for days_left, cited, px0 in self._pending:
            days_left -= 1
            if days_left <= 0:
                sign = 1 if ctx.close > px0 * (1 + 1e-9) else (-1 if ctx.close < px0 * (1 - 1e-9) else 0)
                brain.feedback(cited, sign)
            else:
                still.append((days_left, cited, px0))
        self._pending = still

    def _render_layers(self, brain, ctx, date: str) -> str:
        out: List[str] = []
        for layer, label in (("short", "The short-term information:"), ("mid", "The mid-term information:"),
                             ("long", "The long-term information:"), ("reflection", "The reflection-term information:")):
            items = brain.query(layer, ctx.ticker, date)
            if not items:
                continue
            out.append(label)
            out.extend(f"{mid}. {text}" for mid, text in items)
        return "\n".join(out) if out else "No memories retrieved yet."

    def _assemble_test_info(self, brain, ctx, date: str, mom: int) -> str:
        prefix = f"The ticker of the stock to be analyzed is {ctx.ticker} and the current date is {date}"
        pos = f"You currently hold ~{ctx.current_weight * 100:.0f}% of NAV in {ctx.ticker}."
        mom_word = {1: "positive", 0: "zero", -1: "negative"}[mom]
        mom_block = f"{_MOMENTUM_EXPLANATION}\nThe cumulative return of past 3 days for this stock is {mom_word}."
        return "\n".join([prefix, pos, self._render_layers(brain, ctx, date), mom_block])

    def _assemble_train_info(self, brain, ctx, date: str, fut: float) -> str:
        prefix = (f"The current date is {date}. Here are the observed financial market facts: for {ctx.ticker}, "
                  f"the price difference between the next trading day and the current trading day is: {fut:.4f}")
        return "\n".join([prefix, self._render_layers(brain, ctx, date)])

    @staticmethod
    def _parse_cited(s) -> List[str]:
        return [x.strip() for x in str(s or "").replace(";", ",").split(",") if x.strip()]

    def decide(self, ctx) -> float:
        brain = self._ensure_brain(ctx)
        date = _date_str(ctx)
        self._process_feedback(brain, ctx)
        self._ingest(brain, ctx, date)

        if ctx.phase == "warmup":
            fut = self._next_day_delta(ctx)
            r = self.call(render_prompt("finmem/train", investment_info=self._assemble_train_info(brain, ctx, date, fut)),
                          ["reasoning", "cited"], f"finmem_train_{ctx.ticker}")
            summary = (r.get("reasoning") or "").strip()
            if summary:
                brain.add("reflection", summary, date)
            brain.step()
            direction = "BUY" if fut > 0 else ("SELL" if fut < 0 else "HOLD")
            return self.decision_to_weight(direction, ctx)

        mom = _momentum(ctx)
        direction, r = self.decide_action(
            render_prompt("finmem/test", investment_info=self._assemble_test_info(brain, ctx, date, mom)),
            f"finmem_{ctx.ticker}", extra_tags=["reasoning", "cited"])
        summary = (r.get("reasoning") or "").strip()
        if summary:
            brain.add("reflection", summary, date)
        self._pending.append((_FEEDBACK_WINDOW, self._parse_cited(r.get("cited")), float(ctx.close)))
        brain.step()
        return self.decision_to_weight(direction, ctx)


_ASSET_INFO_CACHE: Optional[dict] = None


def _asset_params(ticker: str) -> Dict[str, str]:
    global _ASSET_INFO_CACHE
    if _ASSET_INFO_CACHE is None:
        import json

        from utils.prompt_loader import _PROMPTS_DIR

        try:
            _ASSET_INFO_CACHE = json.loads((_PROMPTS_DIR / "finagent" / "asset_infos.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _ASSET_INFO_CACHE = {}
    info = _ASSET_INFO_CACHE.get(ticker) or {}
    return {
        "asset_name": info.get("companyName", ticker),
        "asset_exchange": info.get("exchange", ticker),
        "asset_sector": info.get("sector", ticker),
        "asset_industry": info.get("industry", ticker),
        "asset_description": info.get("description", ticker),
    }


class FinAgentEquiv(LLMStrategy):

    name = "finagent"
    INITIAL_CAPITAL = 100000.0

    def reset_memory(self) -> None:
        self._mem: Optional[Any] = None
        self._eid = os.environ.get("FINPANEL_EXPERIMENT_ID", "finagent_default")
        self._trades: deque = deque(maxlen=14)
        self._prev_nav: Optional[float] = None

    def on_evaluation_start(self) -> None:
        self._trades.clear()
        self._prev_nav = None

    def _ensure_mem(self, ctx):
        if self._mem is None:
            from baselines.finagent_memory import FinAgentMemory

            self._mem = FinAgentMemory(self._eid, ctx.ticker)
        return self._mem

    def _kline_text(self, ctx) -> str:
        df = getattr(ctx, "ohlc", None)
        if df is None or len(df) < 2:
            return "insufficient price history"
        d = df.tail(14).copy()
        d.columns = [str(c).lower() for c in d.columns]
        c = d["close"].astype(float)
        ma5 = c.rolling(5, min_periods=1).mean()
        std5 = c.rolling(5, min_periods=1).std().fillna(0.0)
        lines = ["date | O | H | L | C | MA5 | BBU(+2σ) | BBL(-2σ)"]
        n = len(d)
        for i in range(n):
            row = d.iloc[i]
            idx = d.index[i]
            dt = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
            mark = "  <-- today" if i == n - 1 else ""
            lines.append(f"{dt} | {float(row['open']):.2f} | {float(row['high']):.2f} | {float(row['low']):.2f} | {float(row['close']):.2f} | {ma5.iloc[i]:.2f} | {ma5.iloc[i] + 2 * std5.iloc[i]:.2f} | {ma5.iloc[i] - 2 * std5.iloc[i]:.2f}{mark}")
        return "\n".join(lines)

    def _movement(self, ctx, days: int) -> str:
        h = ctx.history
        if h is None or len(h) <= days:
            return "N/A"
        return f"{(float(h.iloc[-1]) / float(h.iloc[-1 - days]) - 1.0) * 100.0:+.2f}%"

    def _trading_text(self, ctx) -> str:
        if not self._trades:
            return "No prior trades."
        rows = "\n".join(f"{d}: {a} — {why}" for d, a, why in self._trades)
        cum = (float(ctx.nav_pre) / self.INITIAL_CAPITAL - 1.0) * 100.0
        return (f"{rows}\nCurrent position: ~{ctx.current_weight * 100:.0f}% of NAV. "
                f"Cumulative return of the trade so far: {cum:+.2f}%.")

    @staticmethod
    def _prices_row(row) -> str:
        o, h, l, c = (float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]))
        return f"Prices: Open: ({o:.2f}), High: ({h:.2f}), Low: ({l:.2f}), Close: ({c:.2f}), Adj Close: ({c:.2f})"

    def _mi_block(self, ctx, news: List[str], brief: str, date: str) -> str:
        if not news and not brief:
            return "There is no latest market_intelligence."
        lines = [f"Date: Today is {date}."]
        df = getattr(ctx, "ohlc", None)
        if df is not None and len(df):
            row = {str(k).lower(): v for k, v in df.iloc[-1].items()}
            lines.append(self._prices_row(row))
        else:
            lines.append("Prices: Today is closed for trading.")
        for i, n in enumerate(news[:20]):
            lines.append(f"ID: {i:06d}\nContent: {n.strip()}")
        if brief:
            lines.append(f"ID: filing\nHeadline: {ctx.ticker} Filing Brief\nContent: {brief}")
        return "\n".join(lines)

    def _past_mi_block(self, ctx, items: List[Dict[str, str]]) -> str:
        if not items:
            return "There is no past market_intelligence."
        px: Dict[str, Any] = {}
        df = getattr(ctx, "ohlc", None)
        if df is not None and len(df):
            d = df.copy()
            d.columns = [str(c).lower() for c in d.columns]
            for idx, row in d.iterrows():
                px[idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]] = row
        out = []
        for it in items:
            text = it["text"]
            if text.startswith("Heading: "):
                text = text[len("Heading: "):]
            row = px.get(it["date"])
            out.append(f"Date: {it['date']}.\nID: {it['id']}\nContent: {text}\n"
                       + (self._prices_row(row) if row is not None else "Prices: Today is closed for trading."))
        return "\n".join(out)

    def _state_line(self, ctx) -> str:
        nav, close = float(ctx.nav_pre), float(ctx.close)
        shares = int(round(nav * ctx.current_weight / close)) if close > 0 else 0
        cash = nav * (1.0 - float(ctx.current_weight))
        profit = nav - self.INITIAL_CAPITAL
        ret = "N/A" if self._prev_nav is None or self._prev_nav <= 0 else f"{nav / self._prev_nav - 1.0:+.4f}"
        return (f"For the current situation, the Adj Close price of the asset is {close:.2f}, "
                f"your current POSITION is {shares}, and your available CASH is {cash:.2f}. "
                f"Your total profit from the opening of the position is {profit:.2f}, "
                f"and your return yesterday was {ret}.")

    def decide(self, ctx) -> float:

        mem = self._ensure_mem(ctx)
        date = _date_str(ctx)
        tkr = ctx.ticker
        news = self.news(ctx)
        brief = self.brief(ctx)
        asset = _asset_params(tkr)

        r1 = self.call(render_prompt("finagent/latest_mi", ticker=tkr, date=date,
                                     latest_market_intelligence=self._mi_block(ctx, news, brief, date), **asset),
                       ["analysis", "summary", "short_query", "medium_query", "long_query"], f"finagent_mi_{tkr}")
        mi_sum = (r1.get("summary") or "").strip() or "None."
        for n in news:
            mem.add("market_intelligence", f"Heading: {n.strip()}", date)
        if brief:
            mem.add("market_intelligence", f"Heading: filing\n{brief}", date)
        seen: set = set()
        past_items: List[Dict[str, str]] = []
        for q in (r1.get("short_query"), r1.get("medium_query"), r1.get("long_query")):
            if not (q or "").strip():
                continue
            for it in mem.retrieve("market_intelligence", q, date, k=3):
                if it["id"] not in seen:
                    seen.add(it["id"])
                    past_items.append(it)

        past_sum = (self.call(render_prompt("finagent/past_mi", ticker=tkr,
                                            past_market_intelligence=self._past_mi_block(ctx, past_items), **asset),
                              ["analysis", "summary"], f"finagent_pastmi_{tkr}").get("summary") or "").strip() or "None."

        r3 = self.call(render_prompt("finagent/low_reflection", ticker=tkr, date=date,
                                     past_mi_summary=past_sum, latest_mi_summary=mi_sum,
                                     kline_text=self._kline_text(ctx),
                                     short_term_past_price_movement=self._movement(ctx, 1),
                                     medium_term_past_price_movement=self._movement(ctx, 7),
                                     long_term_past_price_movement=self._movement(ctx, 14), **asset),
                       ["short_term_reasoning", "medium_term_reasoning", "long_term_reasoning", "query"],
                       f"finagent_low_{tkr}")
        low = "\n".join(f"{lbl}: {(r3.get(k) or '').strip()}"
                        for lbl, k in (("Short-Term", "short_term_reasoning"),
                                       ("Medium-Term", "medium_term_reasoning"),
                                       ("Long-Term", "long_term_reasoning"))
                        if (r3.get(k) or "").strip())
        if low:
            mem.add("low_level_reflection", low, date)
        past_low = "\n".join(f"- {it['text']}" for it in mem.retrieve("low_level_reflection", r3.get("query") or tkr, date, k=5)) or "None."

        r4 = self.call(render_prompt("finagent/high_reflection", ticker=tkr,
                                     past_mi_summary=past_sum, latest_mi_summary=mi_sum,
                                     past_low_level_reflection=past_low, latest_low_level_reflection=low or "None.",
                                     trading_text=self._trading_text(ctx), **asset),
                       ["reasoning", "improvement", "summary", "query"], f"finagent_high_{tkr}")
        high_sum = (r4.get("summary") or "").strip()
        high_imp = (r4.get("improvement") or "").strip()
        high = "\n".join(x for x in (high_sum, f"Improvements: {high_imp}" if high_imp else "") if x)
        if high_sum:
            mem.add("high_level_reflection", high_sum, date)
        past_high = "\n".join(f"- {it['text']}" for it in mem.retrieve("high_level_reflection", r4.get("query") or tkr, date, k=5)) or "None."

        action, r5 = self.decide_action(
            render_prompt("finagent/decision", ticker=tkr, trader_preference=TRADER_PREFERENCE_AGGRESSIVE,
                          past_mi_summary=past_sum, latest_mi_summary=mi_sum,
                          past_low_level_reflection=past_low, latest_low_level_reflection=low or "None.",
                          past_high_level_reflection=past_high, latest_high_level_reflection=high or "None.",
                          state=self._state_line(ctx), **asset),
            f"finagent_decide_{tkr}", extra_tags=("analysis", "reasoning"))
        self._trades.append((date, action, (r5.get("reasoning") or "").strip()))
        self._prev_nav = float(ctx.nav_pre)
        return self.decision_to_weight(action, ctx)


LLM_NAMES = ("finmem", "finagent", "tradingagents")
_CACHE: Dict[str, Any] = {}


def get_llm_strategy(name: str):
    if not _CACHE:
        from baselines.tradingagents import TradingAgentsStrategy

        _CACHE.update({
            FinMemEquiv.name: FinMemEquiv,
            FinAgentEquiv.name: FinAgentEquiv,
            TradingAgentsStrategy.name: TradingAgentsStrategy,
        })
    return _CACHE[name]
