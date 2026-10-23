from __future__ import annotations

import contextlib
import csv
import fcntl
import json
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
SUMMARY_CSV = RESULTS / "summary.csv"
DAILY_CSV = RESULTS / "daily.csv"

SUMMARY_COLS = ["period", "start", "end", "label", "session", "ticker",
                "days", "cr", "sharpe", "mdd", "av", "sortino", "model", "run_id", "updated"]
DAILY_COLS = ["period", "label", "session", "model", "ticker", "date", "total_value"]

_MODEL_TAGS = {"gpt-4o-mini-2024-07-18": "4o-mini, 42", "gpt-oss-120b": "gpt-oss, 42",
               "Qwen3-30B-A3B": "Qwen3-30B, 42", "Llama-3.3-70B-Instruct": "Llama-3.3-70B, 42"}
_UNTAGGED_MODELS = {"", "rule-based"}


def _model_tag(model: str | None) -> str:
    m = model or ""
    return "" if m in _UNTAGGED_MODELS else _MODEL_TAGS.get(m, m)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            return [dict(r) for r in csv.DictReader(fh)]
    except Exception:
        return []


def _write_rows(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", restval="", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    os.replace(tmp, path)


STORE_LOCK = RESULTS / ".store.lock"


@contextlib.contextmanager
def _store_lock():
    STORE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_LOCK, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _num(x) -> float | None:
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _disp(label: str, session: str, model: str | None = None) -> str:
    out = f"{label} @{session}" if session else (label or "")
    tag = _model_tag(model)
    return f"{out} [{tag}]" if tag else out


def _campaign_session(session: str, ticker: str) -> str:
    s, t = session or "", ticker or ""
    return s[:-len(t)] if t and s.endswith(t) and len(s) > len(t) else s


def _row_key(r: dict) -> tuple:
    return (r.get("period"), r.get("label"), r.get("session") or "", r.get("model") or "", r.get("ticker"))


_METRICS = [("SPR", "sharpe", 2, True), ("SOR", "sortino", 2, True), ("CR", "cr", 2, True),
            ("MDD", "mdd", 1, False), ("AV", "av", 2, False)]

_STRATEGY_ORDER = [
    "buy_and_hold", "sma_cross", "wma_cross", "atr_band", "bollinger_bands",
    "turn_of_the_month", "trend_following",
    "arima", "xgboost",
    "a2c", "ddpg", "ppo", "sac", "td3",
]
_FG_ORDER = ["s0", "s1", "s2", "s0_k2", "s0_k6", "s0_k8"]
_LLM_ORDER = ["finmem", "finagent", "tradingagents"]
_TICKER_ORDER = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"]


def _strip_suffixes(label: str) -> str:
    return label.split(" @", 1)[0].split(" [", 1)[0]


def _order_key(label: str):
    stem = _strip_suffixes(label)
    base = stem.lower()
    if base.startswith("finpanel"):
        cfg = base.split(":", 1)[1] if ":" in base else "s0"
        sub = _FG_ORDER.index(cfg) if cfg in _FG_ORDER else len(_FG_ORDER)
        return (2, sub, stem, label)
    fam = base.split("@", 1)[0]
    if fam in _LLM_ORDER:
        return (1, _LLM_ORDER.index(fam), stem, label)
    if base in _STRATEGY_ORDER:
        return (0, _STRATEGY_ORDER.index(base), stem, label)
    return (3, 0, stem, label)


def _group_of(label: str) -> str:
    base = _strip_suffixes(label).lower()
    if base.startswith("finpanel"):
        return "FinPanel"
    if base.split("@", 1)[0] in _LLM_ORDER:
        return "LLM"
    if base in ("arima", "xgboost"):
        return "Predictor"
    if base in ("a2c", "ddpg", "ppo", "sac", "td3"):
        return "RL"
    return "Rule"


def _tkey(t: str):
    return (_TICKER_ORDER.index(t), t) if t in _TICKER_ORDER else (len(_TICKER_ORDER), t)


def _summary_sort_key(r: dict):
    return (r.get("period") or "", _order_key(_disp(r.get("label", ""), r.get("session", ""), r.get("model", ""))), _tkey(r.get("ticker") or ""))


def _daily_sort_key(r: dict):
    return (*_summary_sort_key(r), r.get("date") or "")


def _base_label(summary: dict) -> str:
    stage = str(summary.get("ablation_stage", ""))
    if stage.startswith("strategy:"):
        return stage[len("strategy:"):]
    cfg = (summary.get("experiment_settings", {}) or {}).get("config_id")
    return f"finpanel:{cfg}" if cfg else "finpanel"


def _row_from_summary(p: str) -> dict | None:
    try:
        s = json.load(open(p, encoding="utf-8"))
    except Exception:
        return None
    m = s.get("metrics", {}) or {}
    mt = s.get("metrics_test") or {}
    if mt.get("net_return_pct") is not None:
        m, days = mt, mt.get("test_trading_days")
    else:
        days = m.get("trading_days")
    ec = (s.get("artifacts", {}) or {}).get("equity_curve_csv")
    if not (ec and Path(ec).exists()):
        sib = Path(p).parent / "equity.csv"
        ec = str(sib) if sib.exists() else ec
    return {
        "label": _base_label(s),
        "session": s.get("session") or "",
        "ticker": (s.get("tickers") or ["?"])[0],
        "cr": m.get("net_return_pct"),
        "sharpe": m.get("sharpe_ratio"),
        "mdd": m.get("max_drawdown_pct"),
        "av": m.get("annualized_volatility_pct"),
        "sortino": m.get("sortino_ratio"),
        "days": days,
        "model": "" if (s.get("model_name") or "") in _UNTAGGED_MODELS else s.get("model_name"),
        "run_id": s.get("run_id") or "",
        "equity_csv": ec,
    }


def test_curve(equity_csv) -> list[list]:
    if not equity_csv:
        return []
    p = Path(equity_csv)
    if not p.exists():
        return []
    try:
        with open(p, encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("date") and r.get("total_value")]
    except Exception:
        return []
    first_test = next((i for i, r in enumerate(rows) if r.get("phase", "test") == "test"), None)
    if first_test is None:
        return []
    return [[r["date"], float(r["total_value"])] for r in rows[first_test:]]


def _fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def _period_key(period: str, start: str | None, end: str | None) -> str:
    p = (period or "custom").strip()
    return p if p != "custom" else f"custom_{start}_{end}"


def _load_db() -> dict:
    curves: dict = {}
    for d in _read_rows(DAILY_CSV):
        v = _num(d.get("total_value"))
        if not d.get("date") or v is None:
            continue
        k = (d.get("period"), _disp(d.get("label", ""), _campaign_session(d.get("session", ""), d.get("ticker", "")), d.get("model", "")), d.get("ticker"))
        curves.setdefault(k, []).append([d["date"], v])
    for c in curves.values():
        c.sort()
    out: dict = {}
    for r in _read_rows(SUMMARY_CSV):
        if not (r.get("label") or "").strip() or not (r.get("ticker") or "").strip():
            continue
        period = r.get("period") or "custom"
        ticker = r.get("ticker") or "?"
        disp = _disp(r.get("label", ""), _campaign_session(r.get("session", ""), ticker), r.get("model", ""))
        b = out.setdefault(period, {"meta": {"period": period}, "rows": {}})
        b["meta"]["start"] = r.get("start") or b["meta"].get("start")
        b["meta"]["end"] = r.get("end") or b["meta"].get("end")
        days = _num(r.get("days"))
        row = {"label": disp, "ticker": ticker,
               "base_label": r.get("label", ""), "session": r.get("session") or "",
               "model": r.get("model") or "",
               "cr": _num(r.get("cr")), "sharpe": _num(r.get("sharpe")),
               "sortino": _num(r.get("sortino")),
               "mdd": _num(r.get("mdd")), "av": _num(r.get("av")),
               "days": int(days) if days is not None else None}
        cur = curves.get((period, disp, ticker))
        if cur:
            row["curve"] = cur
        b["rows"][f"{disp}|{ticker}"] = row
        if r.get("updated"):
            b["updated"] = max(b.get("updated", ""), r["updated"])
    return out


def _has_signal(r: dict) -> bool:
    for k in ("cr", "sharpe", "mdd", "av"):
        v = r.get(k)
        v = _num(v) if isinstance(v, str) else v
        if isinstance(v, (int, float)) and v != 0:
            return True
    return False


def upsert_results(summary_paths: list[str], period: str, start: str | None, end: str | None) -> str:
    with _store_lock():
        return _upsert_results_unlocked(summary_paths, period, start, end)


def _upsert_results_unlocked(summary_paths: list[str], period: str, start: str | None, end: str | None) -> str:
    key = _period_key(period, start, end)
    srows = _read_rows(SUMMARY_CSV)
    drows = _read_rows(DAILY_CSV)
    by_key = {_row_key(r): r for r in srows}
    daily_keys = {_row_key(d) for d in drows}
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for p in summary_paths:
        r = _row_from_summary(p)
        if not r:
            continue
        rk = (key, r["label"], r["session"], r["model"], r["ticker"])
        curve = test_curve(r.pop("equity_csv", None))
        inc = by_key.get(rk)
        if inc is not None and rk in daily_keys and _has_signal(inc) and not (curve and _has_signal(r)):
            continue
        by_key[rk] = {**r, "period": key, "start": start or "", "end": end or "", "updated": now}
        drows = [d for d in drows if _row_key(d) != rk]
        drows += [{"period": key, "label": r["label"], "session": r["session"], "model": r["model"],
                   "ticker": r["ticker"], "date": dt, "total_value": v} for dt, v in curve]
        daily_keys.add(rk)
    _write_rows(SUMMARY_CSV, sorted(by_key.values(), key=_summary_sort_key), SUMMARY_COLS)
    _write_rows(DAILY_CSV, sorted(drows, key=_daily_sort_key), DAILY_COLS)
    return key


def print_console_table(summary_paths: list[str]) -> None:
    rows = [r for r in (_row_from_summary(p) for p in summary_paths) if r]
    if not rows:
        return
    try:
        from rich.console import Console
        from rich.table import Table
        t = Table(title="This run", title_style="bold cyan", header_style="bold")
        t.add_column("strategy"); t.add_column("ticker")
        for h, *_ in _METRICS:
            t.add_column(h, justify="right")
        for r in sorted(rows, key=lambda r: (_order_key(_disp(r["label"], r["session"], r["model"])), _tkey(r["ticker"]))):
            cells = [_disp(r["label"], r["session"], r["model"]), r["ticker"]]
            for _h, key, nd, color in _METRICS:
                v = r.get(key)
                if color and isinstance(v, (int, float)):
                    cells.append(f"[green]{v:+.{nd}f}[/green]" if v >= 0 else f"[red]{v:+.{nd}f}[/red]")
                else:
                    cells.append(_fmt(v, nd))
            t.add_row(*cells)
        Console().print(t)
    except Exception:
        for r in rows:
            print(f"  {_disp(r['label'], r['session']):<16} {r['ticker']:>6} CR={_fmt(r['cr'])}")
