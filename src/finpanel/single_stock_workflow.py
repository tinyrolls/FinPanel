import math
from datetime import datetime, timedelta

from dotenv import load_dotenv

from finpanel.metrics_calculator import metrics_calculator_agent
from finpanel.risk_manager import risk_management_agent
from finpanel.actors.llm_actor import run_llm_actor
from finpanel.panel import Panel
from data import api
from finpanel.state import AgentState, DayState


def _reflection_store_path(run_dir, run_id):
    import os as _os
    if run_dir:
        return _os.path.join(str(run_dir), "reflection.jsonl")
    return _os.path.join("results", "finpanel_reflection", f"{run_id}.jsonl")

load_dotenv()


def _tally_directions(emissions: dict) -> tuple:
    d = [str(e.get("direction", "")).upper() for e in emissions.values()]
    return d.count("BUY"), d.count("SELL"), d.count("HOLD")


def ballot_spread(n_buy: int, n_sell: int, n_hold: int) -> float:
    counts = [c for c in (n_buy, n_sell, n_hold) if c > 0]
    n = sum(counts)
    if n <= 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / n) * math.log(c / n) for c in counts)
    return round(h / math.log(3), 4)


def _classify_signal(target_pct: float) -> str:
    if target_pct > 0.10:
        return "BULLISH"
    if target_pct < -0.10:
        return "BEARISH"
    return "NEUTRAL"


NEWS_LOOKBACK_DAYS = 7
NEWS_MAX_ARTICLES = 12
NEWS_ARTICLE_CHARS = 300


def _build_news_evidence(ticker: str, as_of: str) -> str:
    if not as_of:
        return ""
    try:
        d = datetime.strptime(as_of, "%Y-%m-%d")
        start = (d - timedelta(days=NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        items = api.get_company_news(ticker, start, as_of, limit=100000)
    except Exception as exc:
        print(f"   ⚠️ news fetch failed for {ticker}: {type(exc).__name__}")
        return ""
    if not items:
        return ""
    items = sorted(items, key=lambda n: str(getattr(n, "date", "")), reverse=True)[:NEWS_MAX_ARTICLES]
    lines = []
    for n in items:
        title = str(getattr(n, "title", "") or "").strip()
        if title:
            lines.append(f"[{getattr(n, 'date', '')}] {title[:NEWS_ARTICLE_CHARS]}")
    return "\n".join(lines)


class SingleStockWorkflow:

    def __init__(
        self,
        panel: Panel,
        model_name: str = "gpt-4o-mini-2024-07-18",
        model_provider: str = "OpenAI",
        collect_reasoning: bool = False,
    ):
        self.model_name = model_name
        self.model_provider = model_provider
        self.collect_reasoning = collect_reasoning
        self.reasoning_log_path: str | None = None
        self.panel = panel

    def _label_of(self, slug: str) -> str:
        from finpanel.actors.actor_configs import ACTOR_CONFIGS
        cfg = ACTOR_CONFIGS.get(slug)
        return cfg.actor_name if cfg else slug

    def manager_agent_execute_trades(self, state: AgentState) -> AgentState:
        print("🎯 === SINGLE-STOCK MANAGER EXECUTING DAILY TRADE ===")

        tickers = state["data"]["tickers"]
        if len(tickers) != 1:
            raise ValueError(f"SingleStockWorkflow supports exactly 1 ticker, got {len(tickers)}: {tickers}")
        ticker = tickers[0]

        portfolio = state["data"].get("portfolio", {}) or {}
        positions = portfolio.get("positions", {}) or {}
        analyst_signals = state["data"].setdefault("analyst_signals", {})

        persistent_signal_history: list[tuple[str, float]] = state["data"].get("persistent_signal_history", [])

        price_map: dict[str, float] = {t: 0.0 for t in tickers}
        end_date = state["data"].get("end_date")

        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            fallback_start = (end_dt - timedelta(days=7)).strftime("%Y-%m-%d")
            for t in tickers:
                try:
                    prices_df = api.get_price_data(t, fallback_start, end_date)
                    if not prices_df.empty:
                        price_map[t] = float(prices_df["close"].iloc[-1])
                except Exception:
                    continue

        risk_info = analyst_signals.get("risk_management_agent", {}) or {}
        risk_limits = risk_info.get("risk_limits_by_ticker", {}) or {}
        for t in tickers:
            if price_map.get(t, 0.0) > 0:
                continue
            fallback_price = risk_limits.get(t, {}).get("current_price")
            try:
                fallback_price = float(fallback_price)
            except (TypeError, ValueError):
                fallback_price = 0.0
            if fallback_price > 0:
                price_map[t] = fallback_price

        if end_date:
            for t in tickers:
                if price_map.get(t, 0.0) > 0:
                    continue
                try:
                    day_df = api.get_price_data(t, end_date, end_date)
                    if not day_df.empty:
                        price_map[t] = float(day_df["close"].iloc[-1])
                except Exception:
                    continue

        cash = float(portfolio.get("cash", 0.0) or 0.0)
        portfolio_metrics = risk_info.get("portfolio_metrics", {}) or {}
        total_value = float(portfolio_metrics.get("total_value", cash) or cash)

        if total_value <= 0:
            position_value = 0.0
            for t in tickers:
                pos = positions.get(t, {}) or {}
                shares = float(pos.get("shares", 0.0) or 0.0)
                position_value += shares * price_map.get(t, 0.0)
            total_value = cash + position_value

        if total_value <= 0:
            total_value = cash

        prior_dir = state["data"].get("persistent_last_direction")
        _today_close = float(price_map.get(ticker, 0.0) or 0.0) or None
        state["data"]["_today_close_price"] = _today_close
        final_dir, _surface = self._finpanel_discrete_decide(
            state, ticker, prior_direction=prior_dir,
            today_close=_today_close,
        )
        if final_dir == "BUY":
            target_pct = 1.0
        elif final_dir == "SELL":
            target_pct = 0.0
        else:
            _shares = float((positions.get(ticker, {}) or {}).get("shares", 0.0) or 0.0)
            _px = float(price_map.get(ticker, 0.0) or 0.0)
            _cur_w = (_shares * _px) / total_value if total_value > 0 and _px > 0 else 0.0
            target_pct = max(0.0, min(1.0, _cur_w))
        state["data"]["persistent_last_direction"] = final_dir
        state["data"].setdefault("coordination_log", []).append(
            {**_surface, "as_of": state["data"].get("end_date", "")}
        )
        state["data"]["_day"].put("proc", "manager", {
            "direction": final_dir,
            "n_buy": _surface.get("n_buy"),
            "n_sell": _surface.get("n_sell"),
            "n_hold": _surface.get("n_hold"),
            "disputed": _surface.get("disputed"),
            "ballot_spread": _surface.get("ballot_spread"),
        }, ticker=ticker)
        print(f"\U0001F4CA Discrete decision: {ticker} final={final_dir} -> target={target_pct:+.1%}")

        current_date = state["data"].get("end_date", "")

        state["data"]["_manager_final_dir"] = final_dir
        state["data"]["_manager_target_pct"] = target_pct

        persistent_signal_history.append((current_date, target_pct))
        if len(persistent_signal_history) > 252:
            persistent_signal_history = persistent_signal_history[-252:]
        state["data"]["persistent_signal_history"] = persistent_signal_history

        return state

    def _decision_manager_synthesize_discrete(
        self,
        ticker: str,
        emissions: dict,
        prior_direction: str | None = None,
        memory_block: str = "",
        records_block: str = "",
        account_state_block: str = "",
    ) -> tuple:
        fallback = (self._discrete_majority_fallback(emissions), "")
        try:
            from llm.functions import call_llm
            from llm.tag_parse import extract_choice, strip_tags

            analyses = []
            for slug, e in emissions.items():
                d = str(e.get("direction") or "HOLD").upper()
                r = str(e.get("reasoning") or "")
                analyses.append(f"- {self._label_of(slug)}: leans {d} — {r}")
            analyses_text = "\n".join(analyses) if analyses else "(no analyses)"

            ctx = ""
            if prior_direction is not None:
                ctx = (f"\nFor context, yesterday's final direction was "
                       f"{prior_direction}. INFORMATIONAL ONLY — today's call is "
                       "based on today's full evidence set, not yesterday's.")

            from utils.prompt_loader import render_messages
            prompt = render_messages(
                "finpanel/decision_manager",
                ticker=ticker,
                ctx=ctx,
                analyses_text=analyses_text,
                records_section=records_block or "",
                account_state_block=account_state_block or "(no account state)",
                memory_block=memory_block or "(no prior decisions yet)",
            )
            resp = call_llm(
                prompt=prompt, model_name=self.model_name,
                model_provider=self.model_provider,
                tags=["action", "reasoning"],
                agent_name="decision_manager",
            )
            direction = extract_choice(resp.content, ["BUY", "SELL", "HOLD"], "")
            if direction not in {"BUY", "SELL", "HOLD"}:
                print(f"   ⚠️ decision_manager returned invalid direction "
                      f"({direction[:40]!r}) — falling back to majority vote "
                      f"-> {fallback[0]}")
                return fallback
            rationale = (resp.tags.get("reasoning") or "").strip() or strip_tags(resp.content, ["action", "reasoning"])
            return (direction, rationale)
        except Exception as _exc:
            print(f"   ⚠️ decision_manager LLM failed ({type(_exc).__name__}): "
                  f"falling back to majority vote -> {fallback[0]}")
            return fallback

    @staticmethod
    def _discrete_majority_fallback(emissions: dict) -> str:
        scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        for e in emissions.values():
            d = str(e.get("direction") or "HOLD").upper()
            if d not in scores:
                d = "HOLD"
            scores[d] += 1.0
        top = max(scores, key=lambda k: scores[k])
        if scores["BUY"] == scores["SELL"] and (scores["BUY"] + scores["SELL"]) > 0:
            return "HOLD"
        return top

    def _build_memory_block(self, state: AgentState, ticker: str) -> str:
        try:
            hist = list(state["data"].get("persistent_decision_history", []) or [])
            if not hist:
                return ""
            hist = hist[-10:]
            lines = []
            for entry in hist:
                date = str(entry.get("date", "?"))
                direction = str(entry.get("direction", "?"))
                rationale = str(entry.get("rationale", "") or "").strip()
                rationale = rationale.replace("\n", " ")[:260]
                lines.append(f"- {date} → {direction}")
                if rationale:
                    lines.append(f"    rationale: {rationale}")
            return "\n".join(lines) if lines else ""
        except Exception:
            return ""

    def _bucket_rollup(self, rm, slug: str, ticker: str, today: str, agg_line=None) -> tuple:
        from finpanel.distiller import distill_actor_bucket
        from data.reputation_memory import MemoryBucket as _MB
        from finpanel.actors.actor_configs import get_actor_config
        _c = get_actor_config(slug)
        aname = _c.actor_name if _c else slug

        n_ok, n_failed = 0, 0
        for grp in rm.pending_rollups(slug, ticker, today):
            srcs = [b.content for b in grp["sources"]]
            if agg_line:
                hdr = agg_line(grp["target_resolution"], grp["target_period"])
                if hdr:
                    srcs = [hdr] + srcs
            oc = distill_actor_bucket(aname, ticker, grp["target_period"], srcs, today,
                                      self.model_name, self.model_provider)
            if not oc.failed and (oc.section or "").strip():
                rm.append_bucket(_MB(slug, ticker, grp["target_resolution"], grp["target_period"],
                                     oc.section, len(srcs), today))
                n_ok += 1
            else:
                n_failed += 1
        return n_ok, n_failed

    def _build_manager_account_block(
        self, state: AgentState, ticker: str,
    ) -> str:
        try:
            import json as _json

            data = state.get("data", {}) or {}
            portfolio = data.get("portfolio", {}) or {}
            positions = portfolio.get("positions", {}) or {}
            pos = positions.get(ticker, {}) or {}
            shares = int(pos.get("shares", 0) or 0)
            avg_entry = pos.get("avg_entry_price")
            if avg_entry is None and shares:
                cost_basis = pos.get("cost_basis", 0) or 0
                try:
                    avg_entry = float(cost_basis) / float(shares)
                except Exception:
                    avg_entry = None
            metrics = data.get("metrics_by_ticker", {}).get(ticker, {}) or {}
            basic = metrics.get("basic_metrics", {}) or {}
            cur_price = float(basic.get("current_price") or 0.0)
            cash = float(portfolio.get("cash", 0) or 0)

            unrealised = None
            if avg_entry and cur_price and float(avg_entry) > 0:
                unrealised = round(
                    ((float(cur_price) - float(avg_entry)) / float(avg_entry)) * 100.0, 2
                )
            mark = round(cur_price * shares, 2) if cur_price else 0.0
            current_nav = cash + (shares * cur_price if cur_price else 0.0)
            initial_nav = float(data.get("initial_nav") or 0.0)
            account_pnl_pct = None
            if initial_nav > 0:
                account_pnl_pct = round(
                    (current_nav / initial_nav - 1.0) * 100.0, 2
                )
            body = {
                "ticker": ticker,
                "holding_state": "long" if shares > 0 else "flat",
                "shares": shares,
                "avg_entry_price": (round(float(avg_entry), 4) if avg_entry else None),
                "current_price": round(cur_price, 4) if cur_price else None,
                "mark_to_market": mark,
                "cash": round(cash, 2),
                "total_nav": round(current_nav, 2),
                "unrealised_pnl_pct": unrealised,
                "account_total_pnl_pct": account_pnl_pct,
            }
            block = (
                f"```json\n{_json.dumps(body, indent=2)}\n```\n"
                "This is your current book on this name — what you hold and "
                "the open P&L. Use it to inform the call."
            )
            return block
        except Exception:
            return ""

    def _finpanel_discrete_decide(
        self,
        state: AgentState,
        ticker: str,
        prior_direction: str | None,
        today_close: float | None = None,
    ) -> tuple[str, dict]:
        analyst_signals = state["data"].get("analyst_signals", {}) or {}
        round1: dict = {}
        for agent_name, sig in analyst_signals.items():
            if agent_name in {"risk_management_agent", "finpanel_decisions"}:
                continue
            if not isinstance(sig, dict):
                continue
            payload = sig.get(ticker)
            if not isinstance(payload, dict):
                continue
            d = payload.get("direction")
            if d is None:
                continue
            round1[agent_name] = {
                "direction": str(d).upper(),
                "reasoning": str(payload.get("reasoning", "") or ""),
            }

        emissions = round1
        n_buy, n_sell, n_hold = _tally_directions(emissions)

        memory_block = self._build_memory_block(state, ticker)
        _es = state.get("metadata", {}).get("experiment_settings") or {}
        records_block = (self._build_actor_records_block(state, ticker, emissions)
                         if _es.get("use_records", True) else "")

        account_state_block = self._build_manager_account_block(state, ticker)

        final, final_rationale = (
            self._decision_manager_synthesize_discrete(
                ticker=ticker, emissions=emissions,
                prior_direction=prior_direction,
                memory_block=memory_block,
                records_block=records_block,
                account_state_block=account_state_block,
            )
        )

        try:
            today_date = str(state["data"].get("end_date", ""))
            state["data"].setdefault("persistent_decision_history", []).append({
                "date": today_date,
                "direction": final,
                "rationale": final_rationale,
                "close": float(today_close) if today_close is not None else None,
            })
        except Exception:
            pass

        disputed = (n_buy > 0 and n_sell > 0)
        surface = {
            "decision_mode": "discrete",
            "n_agents": len(emissions),
            "n_buy": int(n_buy),
            "n_sell": int(n_sell),
            "n_hold": int(n_hold),
            "disputed": bool(disputed),
            "ballot_spread": ballot_spread(n_buy, n_sell, n_hold),
            "final_direction": final,
            "prior_direction": prior_direction,
        }
        print(f"   🧭 manager → final_direction={final}  "
              f"(BUY={n_buy} SELL={n_sell} HOLD={n_hold}, "
              f"disputed={'Y' if disputed else 'N'})")
        return final, surface

    def _build_actor_records_block(
        self, state: AgentState, ticker: str, emissions: dict,
    ) -> str:
        slugs = [s for s in self.panel.slugs if s in emissions]
        if not slugs:
            return ""
        today = str(state["data"].get("end_date") or "")
        if not today:
            return ""

        try:
            from data.reputation_memory import get_default_ledger, render_resume
            sections = get_default_ledger().resumes_for_manager(slugs, today)
        except Exception as exc:
            print(f"   ⚠️ résumé read skipped ({type(exc).__name__}: {exc})")
            return ""
        lines: list[str] = []
        for slug in slugs:
            text = render_resume(
                persona_text="",
                sections=sections.get(slug, {}) or {},
                actor_name=self._label_of(slug),
                focus_ticker=str(ticker),
            )
            if text.strip():
                lines.append(text.strip())

        if not lines:
            return ""
        header = (
            "\n  What each trader's judgment on this panel has been worth so far —\n"
            "  a record of what happened, not a ranking and not advice:\n"
        )
        body = "\n".join(f"  {ln}" for ln in "\n".join(lines).splitlines())
        return (
            header + body +
            "\n  Use it to decide whose read of TODAY deserves the most weight.\n"
        )

    def _reflect_record_node(self, state: AgentState) -> AgentState:
        try:
            import json as _json
            import os as _os
            data = state["data"]
            ticker = (data.get("tickers") or ["?"])[0]
            today = str(data.get("end_date"))
            run_id = (state.get("metadata", {}).get("run_id") or f"{ticker}_{today}")

            _day = data["_day"]
            raw = _day.get("proc", "actors", ticker=ticker) or {}
            signals = {
                name: float(v.get("raw_position_percentage", 0.0) or 0.0)
                for name, v in raw.items()
                if isinstance(v, dict) and name not in (
                    "risk_management_agent", "finpanel_decisions")
            }
            if not signals:
                return state

            ballots = []
            for name, v in raw.items():
                if not isinstance(v, dict):
                    continue
                if name in {"risk_management_agent", "finpanel_decisions"}:
                    continue
                ballots.append({
                    "agent_name": name,
                    "direction": v.get("direction"),
                    "reasoning_excerpt": str(v.get("reasoning", "") or ""),
                })

            manager_final = _day.get("proc", "manager", ticker=ticker)
            close_today = data.get("_today_close_price")

            record = {
                "date": today,
                "ticker": ticker,
                "signals": signals,
                "ballots": ballots,
                "manager_final": manager_final,
                "close_today": close_today,
            }
            spath = _reflection_store_path(
                state.get("metadata", {}).get("run_dir"), run_id)
            _os.makedirs(_os.path.dirname(spath), exist_ok=True)
            with open(spath, "a", encoding="utf-8") as fh:
                fh.write(_json.dumps(record) + "\n")
        except Exception as exc:
            print(f"   ⚠️ reflection record skipped ({type(exc).__name__}: {exc})")
        return state

    def _reflect_eval_node(self, state: AgentState) -> AgentState:
        if not (state.get("metadata", {}).get("experiment_settings") or {}).get("use_records", True):
            return state
        try:
            import json as _json
            import os as _os

            data = state["data"]
            ticker = (data.get("tickers") or ["?"])[0]
            today = str(data.get("end_date"))
            run_id = (state.get("metadata", {}).get("run_id")
                      or f"{ticker}_{today}")

            spath = _reflection_store_path(
                state.get("metadata", {}).get("run_dir"), run_id)
            _os.makedirs(_os.path.dirname(spath), exist_ok=True)

            past_rows = []
            seen = set()
            if _os.path.exists(spath):
                with open(spath, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            r = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        d = r.get("date")
                        if not d or d in seen or d == today:
                            continue
                        seen.add(d)
                        past_rows.append(r)
            past_rows.sort(key=lambda r: r["date"])
            if len(past_rows) < 2:
                return state

            reasoning_by_subject: dict = {}
            for _r in past_rows:
                for _b in (_r.get("ballots") or []):
                    _nm = _b.get("agent_name")
                    _rx = str(_b.get("reasoning_excerpt") or "").strip()
                    if _nm and _rx:
                        reasoning_by_subject.setdefault(_nm, []).append(_rx)

            if True:
                from data import api as _api

                px = _api.get_price_data(
                    ticker=ticker, start_date=past_rows[0]["date"], end_date=today,
                    data_cutoff_mode="t_minus_1", decision_time=today,
                )
                if px is not None and not px.empty:
                    pxc = px.copy()
                    pxc.columns = [str(c).lower() for c in pxc.columns]
                    close = {str(idx)[:10]: float(c)
                             for idx, c in pxc["close"].items() if c and c > 0}

                    dated_1td = []
                    for _i in range(len(past_rows) - 1):
                        _d0 = str(past_rows[_i].get("date", ""))[:10]
                        _d1 = str(past_rows[_i + 1].get("date", ""))[:10]
                        _c0, _c1 = close.get(_d0), close.get(_d1)
                        if _c0 and _c1 and _c0 > 0:
                            dated_1td.append((_d0, past_rows[_i].get("signals") or {}, _c1 / _c0 - 1.0))

                    if dated_1td:
                        try:
                            from data.reputation_memory import get_default_ledger
                            _pool_set = set(self.panel.slugs)
                            rm = get_default_ledger()
                            _obs_1td: dict = {}
                            for _do, _sigs, _r1 in dated_1td:
                                for _sn, _sv in (_sigs or {}).items():
                                    _obs_1td.setdefault(_sn, []).append((_do, float(_sv), _r1))
                            _bk_ok, _bk_fail = 0, 0

                            try:
                                from finpanel.distiller import compose_episode, track_record_line
                                from data.reputation_memory import (
                                    MemoryBucket as _MB, period_label as _plabel,
                                )
                                from datetime import datetime as _dt, timedelta as _tdl
                                _td = _dt.strptime(today[:10], "%Y-%m-%d").date()
                                _mp = _plabel(_td, "month")
                                _nm, _nr = 0, 0
                                for _subj, _samples in (reasoning_by_subject or {}).items():
                                    if _subj not in _pool_set:
                                        continue
                                    _subj_obs = _obs_1td.get(_subj) or []
                                    _op = track_record_line([(s, r) for _, s, r in _subj_obs])
                                    _content = compose_episode(_samples, _op, "")
                                    if not _content.strip():
                                        continue
                                    _first = ("month", _mp) not in rm._current_buckets(_subj, str(ticker), _td - _tdl(days=1))
                                    rm.append_bucket(_MB(_subj, str(ticker), "month", _mp,
                                                         _content, max(1, len(_samples)), today))
                                    _nm += 1
                                    if _first:
                                        def _agg(resolution, period, _o=_subj_obs):
                                            sel = [x for x in _o
                                                   if _plabel(_dt.strptime(x[0], "%Y-%m-%d").date(), resolution) == period]
                                            txt = track_record_line([(s, r) for _, s, r in sel])
                                            return f"[code-computed across {period}] {txt}" if txt else ""
                                        _ok, _fail = self._bucket_rollup(
                                            rm, _subj, str(ticker), today, agg_line=_agg)
                                        _bk_ok += _ok; _bk_fail += _fail
                                        _nr += 1
                                if _nm:
                                    print(f"   🪣 actor buckets: +{_nm} month, rollups={_nr}")
                            except Exception as exc:
                                print(f"   ⚠️ actor bucket write skipped ({type(exc).__name__}: {exc})")

                            data["distill_stats"] = {
                                "distilled": _bk_ok, "failed": _bk_fail,
                            }
                            if _bk_fail:
                                print(f"   ⚠️ bucket distills failed: {_bk_fail} (ok={_bk_ok})")

                        except Exception as exc:
                            print(f"   ⚠️ reputation memory write skipped "
                                  f"({type(exc).__name__}: {exc})")

        except Exception as exc:
            print(f"   ⚠️ reflection eval skipped ({type(exc).__name__}: {exc})")
        return state

    def _preprocess_node(self, state: AgentState) -> AgentState:
        tickers = state["data"].get("tickers", [])
        if not tickers:
            return state
        ticker = tickers[0]
        as_of = state["data"].get("end_date", "")

        news_text = _build_news_evidence(ticker, as_of)
        if state["data"].get("use_news_selection"):
            try:
                from data.news_retrieval import NewsSelectionStore
                from pathlib import Path as _P
                if not hasattr(self, "_news_sel_store"):
                    import os as _os
                    _sp = state["data"].get("news_selection_path") or _os.environ.get("NEWS_SELECTION_PATH") or str(_P(__file__).resolve().parents[2] / "data" / "news_selection.jsonl")
                    self._news_sel_store = NewsSelectionStore.load(_sp)
                sel = self._news_sel_store.get(ticker, as_of)
                if sel is not None:
                    news_text = "\n".join(sel)
                    print(f"   🔎 News selection: top-{len(sel)} (ada-002) for {ticker}")
            except Exception as _ex:
                print(f"   ⚠️ news selection load failed: {type(_ex).__name__}: {_ex}")
        if news_text:
            print(f"   📰 News: {len(news_text.splitlines())} articles for {ticker} (<= {as_of})")

        fundamentals_text = ""
        if state["data"].get("use_fundamentals"):
            try:
                if not hasattr(self, "_brief_store"):
                    from data.fundamentals_briefs import FundamentalsBriefStore
                    from pathlib import Path as _P
                    import os as _os
                    _bp = state["data"].get("fundamentals_briefs_path") or _os.environ.get("FUNDAMENTALS_BRIEFS_PATH") or str(_P(__file__).resolve().parents[2] / "data" / "fundamentals_briefs.jsonl")
                    self._brief_store = FundamentalsBriefStore.load(_bp)
                brief = self._brief_store.get_brief(ticker, as_of) if as_of else None
                fundamentals_text = brief.render() if brief else ""
                if brief:
                    print(f"   📑 Fundamentals: latest filing brief for {ticker} (<= {as_of})")
            except Exception as _ex:
                print(f"   ⚠️ fundamentals brief load failed: {type(_ex).__name__}: {_ex}")

        technical_text = ""
        if as_of:
            try:
                from data.technical_features import compute_technical_brief
                technical_text = compute_technical_brief(ticker, as_of) or ""
                if technical_text:
                    print(f"   📈 Technicals: MACD/RSI/Bollinger + 1w/1m for {ticker} (<= {as_of})")
            except Exception as _ex:
                print(f"   ⚠️ technical brief failed: {type(_ex).__name__}: {_ex}")

        _day = state["data"]["_day"]
        _day.put("pre", "news", news_text or "", ticker=ticker)
        _day.put("pre", "fundamentals", fundamentals_text, ticker=ticker)
        _day.put("pre", "technical", technical_text, ticker=ticker)
        return state

    def actor_agents_node(self, state: AgentState) -> AgentState:
        tickers = state["data"].get("tickers", [])
        if not tickers:
            return state

        ticker = tickers[0]
        analyst_signals = state["data"].setdefault("analyst_signals", {})

        _mn, _mp = self.model_name, self.model_provider
        actor_agent_map = [
            (slug,
             (lambda st, _s=slug, _m=_mn, _p=_mp:
                 run_llm_actor(st, _s, model_name=_m, model_provider=_p)),
             slug)
            for slug in self.panel.slugs
        ]
        if not actor_agent_map:
            print(f"⚠️ Panel {self.panel.name!r} is empty — skipping actor agents this day.")
            return state

        print(f"🎨 Running the full panel ({len(actor_agent_map)} actors, "
              f"{self.panel.name}) for {ticker}...")

        raw_outputs: dict = {}
        for agent_name, agent_func, _slug in actor_agent_map:
            try:
                result = agent_func(state)

                output = result.get("output")
                recommendations = []
                if output:
                    recommendations = getattr(output, "recommendations", []) or []

                if recommendations:
                    rec = recommendations[0]
                    reasoning = str(getattr(rec, "reasoning", "") or "")

                    raw_direction = getattr(rec, "direction", None)
                    _dir = str(raw_direction).upper().strip() if raw_direction is not None else "HOLD"
                    if _dir not in {"BUY", "SELL", "HOLD"}:
                        _dir = "HOLD"
                    position_pct = (1.0 if _dir == "BUY"
                                    else (-1.0 if _dir == "SELL" else 0.0))
                    original_pct = position_pct
                    raw_signal = _classify_signal(original_pct)
                    raw_outputs[agent_name] = {
                        "raw_position_percentage": original_pct,
                        "raw_signal": raw_signal,
                        "reasoning": reasoning,
                        "direction": _dir,
                    }
                    import os as _os
                    _p2b_path = _os.environ.get("FINPANEL_P2B_CAPTURE")
                    if _p2b_path:
                        try:
                            import json as _json
                            with open(_p2b_path, "a", encoding="utf-8") as _fh:
                                _fh.write(_json.dumps({
                                    "date": str(state["data"].get("end_date")),
                                    "agent_name": agent_name,
                                    "raw_position_percentage": original_pct,
                                }) + "\n")
                        except OSError:
                            pass

                    if position_pct > 0.05:
                        signal = "BULLISH"
                        confidence = min(1.0, abs(position_pct))
                    elif position_pct < -0.05:
                        signal = "BEARISH"
                        confidence = min(1.0, abs(position_pct))
                    else:
                        signal = "NEUTRAL"
                        confidence = 1.0 - min(1.0, abs(position_pct) / 0.05)

                    analyst_signals[agent_name] = {
                        ticker: {
                            "signal": signal,
                            "confidence": confidence,
                            "reasoning": reasoning,
                            "position_percentage": position_pct,
                            "direction": _dir,
                        }
                    }

                    if self.collect_reasoning and reasoning:
                        self._save_reasoning(
                            date=state["data"].get("end_date", "unknown"),
                            agent_name=agent_name,
                            signal=signal,
                            confidence=confidence,
                            position_pct=position_pct,
                            reasoning=reasoning,
                            ticker=ticker,
                        )

                    print(f"   💡 {agent_name}: {signal} (conf={confidence:.2f}, pos={position_pct:+.1%})")
                else:
                    analyst_signals[agent_name] = {
                        ticker: {"signal": "NEUTRAL", "confidence": 0.0, "reasoning": "No recommendation", "position_percentage": 0.0}
                    }

            except Exception as e:
                print(f"   ❌ {agent_name} failed: {e}")
                analyst_signals[agent_name] = {
                    ticker: {"signal": "NEUTRAL", "confidence": 0.0, "reasoning": f"Error: {e}", "position_percentage": 0.0}
                }

        state["data"]["_day"].put("proc", "actors", dict(raw_outputs), ticker=ticker)
        return state

    def _save_reasoning(
        self,
        date: str,
        agent_name: str,
        signal: str,
        confidence: float,
        position_pct: float,
        reasoning: str,
        ticker: str,
    ) -> None:
        import json
        import os
        from datetime import datetime

        if not self.reasoning_log_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            self.reasoning_log_path = f"results/reasoning_logs/reasoning_{ts}.jsonl"

        os.makedirs(os.path.dirname(self.reasoning_log_path), exist_ok=True)

        record = {
            "date": date,
            "ticker": ticker,
            "agent_name": agent_name,
            "signal": signal,
            "confidence": round(confidence, 4),
            "position_pct": round(position_pct, 4),
            "reasoning": reasoning,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.reasoning_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _apply_update(state: AgentState, update) -> AgentState:
        if not update:
            return state
        if update.get("messages"):
            state["messages"] = list(state["messages"]) + list(update["messages"])
        if "data" in update:
            state["data"] = {**state["data"], **update["data"]}
        if "metadata" in update:
            state["metadata"] = {**state["metadata"], **update["metadata"]}
        return state

    def _run_pipeline(self, state: AgentState) -> AgentState:
        state = self._apply_update(state, self._preprocess_node(state))
        state = self._apply_update(state, self._reflect_eval_node(state))
        state = self._apply_update(state, metrics_calculator_agent(state))
        state = self._apply_update(state, self.actor_agents_node(state))
        state = self._apply_update(state, risk_management_agent(state))
        state = self._apply_update(state, self.manager_agent_execute_trades(state))
        state = self._apply_update(state, self._reflect_record_node(state))
        return state

    def run(self, initial_state: AgentState) -> AgentState:
        tickers = initial_state["data"].get("tickers", [])
        ticker_name = tickers[0] if tickers else "UNKNOWN"
        print(f"📊 Single-stock mode ({ticker_name}): running clean directional workflow")

        initial_state["metadata"]["single_stock_mode"] = True
        initial_state["metadata"]["workflow_type"] = "single_stock"
        initial_state["metadata"]["model_name"] = self.model_name
        initial_state["metadata"]["model_provider"] = self.model_provider
        initial_state["metadata"]["original_tickers"] = list(tickers)

        if initial_state["metadata"].get("run_id") and not initial_state["metadata"].get("backtest_run_id"):
            initial_state["metadata"]["backtest_run_id"] = initial_state["metadata"]["run_id"]
        if initial_state["metadata"].get("backtest_run_id") and not initial_state["metadata"].get("run_id"):
            initial_state["metadata"]["run_id"] = initial_state["metadata"]["backtest_run_id"]

        initial_state["data"]["_day"] = DayState()
        result = self._run_pipeline(initial_state)
        print("📊 === SINGLE-STOCK WORKFLOW COMPLETED ===")
        return result
