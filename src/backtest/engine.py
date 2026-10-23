from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Protocol, Sequence

import pandas as pd

from data import api
from evaluation.perf_metrics import compute_perf_stats
from backtest.execution_core import execute_target_positions, finsaber_buy_size
from backtest.artifacts import (
    compute_mechanism_surface,
    compute_test_metrics,
    plot_cost_breakdown_from_csv,
    plot_equity_curve_from_csv,
    save_data_audit_json,
    save_equity_curve_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INITIAL_CASH = 100_000.0

EXEC_PROFILES = {
    "low": {"slippage_bps": 0.0, "short_borrow_apr": 0.02},
    "base": {"slippage_bps": 0.0, "short_borrow_apr": 0.03},
    "high": {"slippage_bps": 0.0, "short_borrow_apr": 0.06},
}


@dataclass
class DayContext:

    date: pd.Timestamp
    ticker: str
    close: float
    history: pd.Series
    position_shares: float
    cash: float
    nav_pre: float
    ohlc: Optional[pd.DataFrame] = None
    phase: str = "test"

    @property
    def current_weight(self) -> float:
        return (self.position_shares * self.close) / self.nav_pre if self.nav_pre > 0 else 0.0


class Strategy(Protocol):
    name: str

    def decide(self, ctx: DayContext) -> float:
        ...


def action_from_direction(direction: str, current_weight: float) -> float:
    d = (direction or "").upper()
    if d == "BUY":
        return 1.0
    if d == "SELL":
        return 0.0
    return current_weight


def _ohlc_series(ticker: str, fetch_start: str, end: str) -> pd.DataFrame:
    df = api.get_price_data(ticker, fetch_start, end)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    idx = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index)
    cols = {c: df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close", "volume") if c in df.columns}
    out = pd.DataFrame(cols, index=idx).sort_index()
    return out[~out.index.duplicated(keep="last")]


def _metrics(values: list[float]) -> dict:
    st = compute_perf_stats(values, initial_value=INITIAL_CASH)
    return {
        "net_return_pct": st["cr"] * 100.0 if st["cr"] is not None else None,
        "annualized_return_pct": st["ar"] * 100.0 if st["ar"] is not None else None,
        "annualized_volatility_pct": st["av"] * 100.0 if st["av"] is not None else None,
        "sharpe_ratio": st["sharpe"],
        "sortino_ratio": st["sortino"],
        "max_drawdown_pct": st["mdd"] * 100.0 if st["mdd"] is not None else None,
    }


def session_run_dir(run_id: str, session_label: Optional[str]) -> Path:
    base = PROJECT_ROOT / "results" / "sessions"
    code = (session_label or "").strip()
    if code and run_id.startswith(f"{code}_"):
        return base / code / run_id[len(code) + 1:]
    return base / run_id


def _run_backtest_iter(
    strategy: Strategy,
    ticker: str,
    start: str,
    end: str,
    eval_start: Optional[str] = None,
    profile: str = "base",
    *,
    warmup_days: int = 420,
    write_artifacts: bool = True,
    session: Optional[str] = None,
    session_label: Optional[str] = None,
    stop_date: Optional[str] = None,
):
    cfg = EXEC_PROFILES[profile]
    warmup_days = int(getattr(strategy, "warmup_days", warmup_days))
    fetch_start = (datetime.fromisoformat(start) - timedelta(days=warmup_days)).date().isoformat()
    ohlc = _ohlc_series(ticker, fetch_start, end)
    if ohlc.empty:
        return {"ticker": ticker, "strategy": strategy.name, "error": "no price data"}
    close, open_ = ohlc["close"], ohlc["open"]

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    window_dates = [d for d in close.index if start_ts <= d < end_ts]
    if not window_dates:
        return {"ticker": ticker, "strategy": strategy.name, "error": "no trading days in window"}

    if eval_start:
        try:
            strategy._warmup_start = pd.Timestamp(start)
        except Exception:
            pass
    if hasattr(strategy, "prepare"):
        strategy.prepare(close, ohlc)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    if session:
        run_id = f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', session)}_{ticker}"
    else:
        run_id = f"strategy_{strategy.name}_{ticker}_{ts}"
    print(f"🧾 Run ID: {run_id}")
    _run_dir = None
    if session and write_artifacts:
        _run_dir = session_run_dir(run_id, session_label or session)
        _run_dir.mkdir(parents=True, exist_ok=True)
    from llm.functions import set_sim_date
    set_sim_date(eval_start or start)
    if hasattr(strategy, "on_run_start"):
        strategy.on_run_start(run_id, _run_dir)
    _stop_dt = pd.Timestamp(stop_date).date() if stop_date else None

    cash = INITIAL_CASH
    positions: dict = {ticker: {"shares": 0, "cost_basis": 0.0, "margin_used": 0.0}}
    cum_cost = cum_turnover = 0.0
    cum_commission = cum_slippage = cum_borrow = 0.0
    rows: list[dict] = []
    eval_ts = pd.Timestamp(eval_start) if eval_start else None

    audit_log: list[dict] = []
    _es = getattr(strategy, "experiment_settings", {}) or {}

    def _flush_partial() -> None:
        if _run_dir is None or not rows:
            return
        try:
            eqp = str(_run_dir / "equity.csv")
            save_equity_curve_csv(rows, eqp, run_metadata={"run_id": run_id, "session": session_label or session, "partial": True}, evaluation_start_date=eval_start)
            _pr = ([r for r in rows if r["phase"] == "test"] if eval_start else rows) or rows
            pm = _metrics([r["total_value"] for r in _pr])
            pm["trading_days"] = len(_pr)
            sm = {
                "run_id": run_id, "session": session_label or session, "partial": True,
                "ablation_stage": getattr(strategy, "ablation_stage", None) or f"strategy:{strategy.name}",
                "experiment_settings": dict(getattr(strategy, "experiment_settings", {}) or {}),
                "model_name": getattr(strategy, "model_name", None) or "rule-based",
                "tickers": [ticker], "start_date": start, "end_date": end,
                "evaluation_start_date": eval_start, "metrics": pm,
                "metrics_test": compute_test_metrics(rows, eval_start, INITIAL_CASH),
                "artifacts": {"equity_curve_csv": eqp},
            }
            with open(_run_dir / "summary.json", "w", encoding="utf-8") as fh:
                json.dump(sm, fh, indent=2)
        except Exception:
            pass

    _ZERO = {"commission_paid": 0.0, "slippage_paid": 0.0, "borrow_fee_paid": 0.0, "total_cost_paid": 0.0, "turnover": 0.0}
    pending_shares = None
    _ledger_reset = False
    for d in window_dates:
        yield d
        if _stop_dt is not None and d.date() >= _stop_dt:
            break
        day_str = d.date().isoformat()
        set_sim_date(day_str)
        if eval_ts is not None and not _ledger_reset and d >= eval_ts:
            _ledger_reset = True
            cash = INITIAL_CASH
            positions = {ticker: {"shares": 0, "cost_basis": 0.0, "margin_used": 0.0}}
            pending_shares = None
            cum_cost = cum_turnover = 0.0
            cum_commission = cum_slippage = cum_borrow = 0.0
            hook = getattr(strategy, "on_evaluation_start", None)
            if callable(hook):
                try:
                    hook()
                except Exception:
                    pass
        phase = "test" if (eval_ts is None or d >= eval_ts) else "warmup"
        try:
            api.set_runtime_price_context(
                universe_mode=_es.get("universe_mode", "static"),
                data_cutoff_mode=_es.get("data_cutoff_mode", "close_of_day"),
                decision_time=day_str,
                disclosure_lag_days=0,
                audit_log=audit_log,
            )
        except Exception:
            pass
        open_px = float(open_.loc[d])
        close_px = float(close.loc[d])
        costs = _ZERO
        if pending_shares is not None and open_px > 0:
            cash, positions, costs, _ = execute_target_positions(
                [ticker],
                {ticker: open_px},
                positions,
                {ticker: pending_shares},
                0.0,
                cash,
                slippage_bps=cfg["slippage_bps"],
                short_borrow_apr=cfg["short_borrow_apr"],
            )
        sh_now = float(positions[ticker]["shares"])
        nav_close = cash + sh_now * close_px
        ctx = DayContext(date=d, ticker=ticker, close=close_px, history=close.loc[:d], position_shares=sh_now, cash=cash, nav_pre=nav_close, ohlc=ohlc.loc[:d], phase=phase)
        target_pct = max(0.0, min(1.0, float(strategy.decide(ctx))))
        if close_px <= 0:
            pending_shares = int(sh_now)
        elif target_pct > 0.5 and sh_now <= 0:
            pending_shares = finsaber_buy_size(target_pct * nav_close, close_px)
        elif target_pct <= 0.5 and sh_now > 0:
            pending_shares = 0
        else:
            pending_shares = int(sh_now)
        shares_post = float(positions[ticker]["shares"])
        pos_val = shares_post * close_px
        total_value = cash + pos_val
        cum_cost += costs["total_cost_paid"]
        cum_turnover += costs["turnover"]
        cum_commission += costs["commission_paid"]
        cum_slippage += costs["slippage_paid"]
        cum_borrow += costs["borrow_fee_paid"]
        rows.append(
            {
                "date": day_str,
                "phase": phase,
                "cash_balance": cash,
                "total_position_value": pos_val,
                "total_value": total_value,
                "gross_total_value": total_value + cum_cost,
                "active_positions": int(shares_post != 0),
                "commission_paid": costs["commission_paid"],
                "slippage_paid": costs["slippage_paid"],
                "borrow_fee_paid": costs["borrow_fee_paid"],
                "total_cost_paid": costs["total_cost_paid"],
                "cumulative_total_cost": cum_cost,
                "turnover": costs["turnover"],
                "cumulative_turnover": cum_turnover,
            }
        )
        if _run_dir is not None and len(rows) % 5 == 0:
            _flush_partial()

    try:
        api.set_runtime_price_context(audit_log=None)
    except Exception:
        pass

    extra = (strategy.on_run_end() if hasattr(strategy, "on_run_end") else {}) or {}
    ablation_stage = extra.get("ablation_stage") or f"strategy:{strategy.name}"
    profile_lbl = extra.get("execution_profile") or profile
    model_name = extra.get("model_name") or getattr(strategy, "model_name", None) or "rule-based"
    model_provider = extra.get("model_provider") or getattr(strategy, "model_provider", None) or "engine"
    experiment_settings = extra.get("experiment_settings") or {"experiment_id": f"strategy:{strategy.name}", "evaluation_start_date": eval_start}
    exec_protocol = extra.get("execution_protocol") or {}
    execution_template = exec_protocol.get("execution_template", "engine_strategy")
    protocol_version = exec_protocol.get("protocol_version", "engine_v1")

    scored_rows = [r for r in rows if r["phase"] == "test"] if eval_start else rows
    warmup_rows = [r for r in rows if r["phase"] == "warmup"] if eval_start else []
    if not scored_rows:
        scored_rows = rows
        warmup_rows = []
    values = [r["total_value"] for r in scored_rows]
    gross_ending = scored_rows[-1]["gross_total_value"]
    gross_return_pct = ((gross_ending / INITIAL_CASH - 1.0) * 100.0) if INITIAL_CASH > 0 else 0.0
    metrics = _metrics(values)
    metrics.update({
        "initial_total_value": INITIAL_CASH,
        "ending_total_value": values[-1],
        "gross_ending_total_value": gross_ending,
        "gross_return_pct": gross_return_pct,
        "cumulative_commission": cum_commission,
        "cumulative_slippage": cum_slippage,
        "cumulative_borrow_fee": cum_borrow,
        "cumulative_total_cost": cum_cost,
        "cumulative_turnover": cum_turnover,
        "trading_days": len(scored_rows),
        **compute_mechanism_surface(scored_rows),
        **(extra.get("metrics_extra") or {}),
    })
    if warmup_rows:
        metrics["warmup_trading_days"] = len(warmup_rows)
        metrics["warmup_ending_total_value"] = warmup_rows[-1]["total_value"]
        metrics["warmup_net_return_pct"] = (warmup_rows[-1]["total_value"] / INITIAL_CASH - 1.0) * 100.0
    metrics_test = compute_test_metrics(rows, eval_start, INITIAL_CASH)

    artifact_paths: dict = {}
    if write_artifacts:
        if session:
            run_dir = session_run_dir(run_id, session_label or session)
        else:
            date_dir = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
            run_dir = PROJECT_ROOT / "results" / date_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        eq_path = str(run_dir / "equity.csv")
        summary_path = str(run_dir / "summary.json")
        audit_path = str(run_dir / "audit.json")
        plot_path = str(run_dir / "equity.png")
        cost_plot_path = str(run_dir / "cost.png")
        run_meta = {
            "run_id": run_id,
            "session": session_label or session,
            "ablation_stage": ablation_stage,
            "execution_profile": profile_lbl,
            "execution_template": execution_template,
            "protocol_version": protocol_version,
        }
        save_equity_curve_csv(rows, eq_path, run_metadata=run_meta, evaluation_start_date=eval_start)
        try:
            plot_equity_curve_from_csv(eq_path, plot_path, model_name, model_provider, summary_metrics={
                "cash_balance": rows[-1]["cash_balance"],
                "total_position_value": rows[-1]["total_position_value"],
                "total_value": values[-1],
                "return_pct": metrics["net_return_pct"],
                "gross_return_pct": gross_return_pct,
                "sharpe_ratio": metrics["sharpe_ratio"],
                "sortino_ratio": metrics["sortino_ratio"],
                "max_drawdown": metrics["max_drawdown_pct"],
                "total_cost_paid": cum_cost,
                "cumulative_turnover": cum_turnover,
            })
            plot_cost_breakdown_from_csv(eq_path, cost_plot_path, model_name, model_provider)
        except Exception as exc:
            print(f"   ⚠️ plot skipped ({type(exc).__name__}: {exc})")
        audit_stats = save_data_audit_json(audit_path, audit_log)
        summary = {
            "run_id": run_id,
            "session": session_label or session,
            "stop_date": stop_date,
            "ablation_stage": ablation_stage,
            "execution_profile": profile_lbl,
            "execution_template": execution_template,
            "protocol_version": protocol_version,
            "experiment_settings": experiment_settings,
            "model_name": model_name,
            "model_provider": model_provider,
            "tickers": [ticker],
            "start_date": start,
            "end_date": end,
            "evaluation_start_date": eval_start,
            "metrics": metrics,
            "metrics_test": metrics_test,
            "artifacts": {
                "equity_curve_csv": eq_path,
                "equity_curve_plot": plot_path,
                "cost_breakdown_plot": cost_plot_path,
                "data_audit_json": audit_path,
            },
            "data_audit": {
                "records": audit_log,
                "record_count": audit_stats["record_count"],
                "violation_count": audit_stats["violation_count"],
                "violation_rate": audit_stats["violation_rate"],
                "records_by_source": audit_stats["records_by_source"],
            },
        }
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        artifact_paths = {
            "run_summary_json": summary_path,
            "equity_curve_csv": eq_path,
            "equity_curve_plot": plot_path,
            "cost_breakdown_plot": cost_plot_path,
            "data_audit_json": audit_path,
        }

    def _r(x, n=2):
        return round(x, n) if x is not None else None

    return {
        "ticker": ticker,
        "strategy": strategy.name,
        "CR%": _r(metrics["net_return_pct"]),
        "Sharpe": _r(metrics["sharpe_ratio"]),
        "Sortino": _r(metrics["sortino_ratio"]),
        "MDD%": _r(metrics["max_drawdown_pct"], 1),
        "AV%": _r(metrics["annualized_volatility_pct"]),
        "cum_cost": _r(cum_cost, 1),
        "days": len(rows),
        "run_id": run_id,
        "artifacts": artifact_paths,
    }


def run_backtest(*args, **kwargs) -> dict:
    gen = _run_backtest_iter(*args, **kwargs)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value or {}
