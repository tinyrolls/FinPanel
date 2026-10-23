from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import os

from data.api import resolve_data_pkl

os.environ.setdefault("DATA_PKL", str(resolve_data_pkl()))

from backtest.engine import EXEC_PROFILES, run_backtest
from baselines.strategies import STRATEGIES
from baselines.rl import RL_NAMES, get_rl_strategy
from baselines.llm_baselines import LLM_NAMES, get_llm_strategy
from utils.configs import REGIMES, build_config

FINPANEL_ALIASES = {"finpanel", "fp"}


def resolve_period(name: str) -> tuple[str, str, str]:
    key = name.strip().lower()
    for r in REGIMES:
        if key in (r["label"].lower(), r["period_id"].lower()):
            return r["start_date"], r["end_date"], r["period_id"]
    options = ", ".join(f"{r['label']} ({r['start_date']}→{r['end_date']})" for r in REGIMES)
    raise SystemExit(f"Unknown --period '{name}'. Choose one of:\n  {options}\n(or pass --start-date/--end-date explicitly).")


def _session_manifest(args, session_dir: str, strategies: list[str], panels: list[str], period_id: str, start: str, end: str, inherits: list[str] | None = None) -> None:
    from datetime import datetime as _dt
    d = Path(session_dir)
    d.mkdir(parents=True, exist_ok=True)
    mf = d / "session.json"
    is_fp = any(s in FINPANEL_ALIASES for s in strategies)
    cur = {"session": args.session, "model": args.model_name, "strategies": strategies,
           "panels": panels if is_fp else None, "period": period_id, "start": start, "end": end}
    if inherits:
        cur["cache_inherits_from"] = list(inherits)
    if mf.exists():
        try:
            prev = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        if prev.get("model") and prev["model"] != cur["model"]:
            print(f"  ⚠️ session {args.session} was created with model={prev['model']} but now model={cur['model']} — the per-session cache is model-keyed; this MISSES and mixes runs.")
        if prev.get("period") and prev["period"] != period_id:
            print(f"  ⚠️ session {args.session} was created on period={prev['period']} but now {period_id}.")
        cur["created_at"] = prev.get("created_at")
    cur.setdefault("created_at", _dt.now().isoformat(timespec="seconds"))
    cur["last_run"] = _dt.now().isoformat(timespec="seconds")
    mf.write_text(json.dumps(cur, indent=2), encoding="utf-8")


def _run_finpanel(args, tickers: list[str], start: str, end: str) -> tuple[int, list[str]]:
    import os
    from data.reputation_memory import derive_experiment_id
    from finpanel.finpanel_strategy import FinPanelStrategy

    es = build_config(getattr(args, "overrides", None))
    if es.get("train_start"):
        es["train_start"] = str(es["train_start"])
    es["train_end"] = args.eval_start_date or (start if es.get("train_start") else "")
    panel = f"panel{es.get('panel_size', 6)}"
    code = getattr(args, "session", None)
    eid = (f"{args.rm_experiment_id}_{panel}" if args.rm_experiment_id
           else derive_experiment_id(panel, start, end, args.model_name))
    if code:
        eid = f"{eid}_{code}"
    os.environ["FINPANEL_EXPERIMENT_ID"] = eid
    exec_protocol = {"protocol_version": "engine_v1",
                     "execution_template": "engine_strategy", "execution_config": {}}
    print(f"→ FinPanel ({panel}) on engine: tickers={tickers} substrate={eid}\n")
    eng_session = f"{code}_{panel}" if code else None
    def _mk(t: str) -> FinPanelStrategy:
        return FinPanelStrategy(
            ticker=t, start_date=start, end_date=end,
            model_name=args.model_name, model_provider="OpenAI",
            experiment_settings=dict(es), execution_protocol=exec_protocol,
            offline_mode=bool(es.get("offline_mode", True)),
            initial_nav=100000.0, execution_profile=args.execution_profile,
        )

    _kw = {"session": eng_session, "session_label": code, "stop_date": args.stop_date}
    results = [run_backtest(_mk(t), t, start, end, args.eval_start_date,
                            args.execution_profile, **_kw) for t in tickers]

    rc, paths = 0, []
    for t, r in zip(tickers, results):
        if r.get("error"):
            print(f"  ⚠️ finpanel/{panel}/{t}: {r['error']}")
            rc = 1
        sp = (r.get("artifacts") or {}).get("run_summary_json")
        if sp:
            paths.append(sp)
    return rc, paths


def _run_classic(strategy_name: str, args, tickers: list[str], start: str, end: str) -> tuple[int, list[str]]:
    is_rl = strategy_name in RL_NAMES
    if strategy_name in STRATEGIES:
        strat_cls = STRATEGIES[strategy_name]
    elif is_rl:
        strat_cls = get_rl_strategy(strategy_name)
    else:
        strat_cls = get_llm_strategy(strategy_name)
    seed = getattr(args, "seed", None)
    code = getattr(args, "session", None)
    is_llm = strategy_name in LLM_NAMES
    if code and not is_llm:
        print(f"  ℹ️ --session ignored for non-LLM '{strategy_name}' (fixed id, overwrite).")
    eng_session = f"{code}_{strategy_name}" if (code and is_llm) else None
    label_session = code if (code and is_llm) else None
    rc, paths = 0, []
    for t in tickers:
        _prev_eid = os.environ.get("FINPANEL_EXPERIMENT_ID")
        if is_llm:
            os.environ["FINPANEL_EXPERIMENT_ID"] = f"{strategy_name}_{code or args.model_name}"
        strat = strat_cls(seed=seed) if (is_rl and seed is not None) else strat_cls()
        if is_llm and getattr(args, "model_name", None):
            strat.model_name = args.model_name
        r = run_backtest(strat, t, start, end, args.eval_start_date, args.execution_profile,
                         session=eng_session, session_label=label_session, stop_date=args.stop_date)
        if is_llm:
            if _prev_eid is None:
                os.environ.pop("FINPANEL_EXPERIMENT_ID", None)
            else:
                os.environ["FINPANEL_EXPERIMENT_ID"] = _prev_eid
        if r.get("error"):
            print(f"  ⚠️ {strategy_name}/{t}: {r['error']}")
            rc = 1
        sp = (r.get("artifacts") or {}).get("run_summary_json")
        if sp:
            paths.append(sp)
    return rc, paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    strat_choices = sorted(STRATEGIES) + list(RL_NAMES) + list(LLM_NAMES) + ["finpanel"]
    ap.add_argument("--strategy", required=True, metavar="LIST",
                    help="one or a COMMA-LIST of: " + ", ".join(strat_choices))
    ap.add_argument("--period", default=None,
                    help="named window (label or period_id): " + ", ".join(r["label"] for r in REGIMES))
    ap.add_argument("--start-date", default=None, help="override --period")
    ap.add_argument("--end-date", default=None, help="override --period")
    ap.add_argument("--tickers", default="TSLA", help="comma-separated (default TSLA)")
    ap.add_argument("--eval-start-date", default=None, help="optional warmup/test split date")
    ap.add_argument("--warmup-days", type=int, default=None, help="memory warm-up (LLM methods): run from `period start − N calendar days` to BUILD memory before the scored window (FinMem/FinAgent's train pass). Scoring starts at the period start (eval_start); the warm-up region is unscored, the scored region keeps strict look-ahead. The report still groups by the period window, not the warm-up-extended start.")
    ap.add_argument("--warmup-start", default=None, help="warm-up/training FIXED ANCHOR date (e.g. 2023-07-01): ALL methods (RL/ML train + LLM memory) start from this exact date and score from the period start — an identical pre-window / same starting line. Overrides --warmup-days. RL/ML train on [warmup-start, period-start); LLM builds memory over it.")
    ap.add_argument("--execution-profile", default="base", choices=list(EXEC_PROFILES))
    ap.add_argument("--seed", type=int, default=None, help="(RL only) override the fixed training seed; re-run with several seeds to union the non-degenerate cells (a dead/no-trade seed never clobbers a live one)")
    ap.add_argument("--set", dest="set_overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="(finpanel) override a configs/base.yaml key for this run (repeatable), e.g. --set panel_size=8")
    ap.add_argument("--model-name", default="gpt-4o-mini-2024-07-18", help="(finpanel) decision model")
    ap.add_argument("--rm-experiment-id", default=None, help="(finpanel) record-substrate id override")
    ap.add_argument("--session", nargs="?", const="__AUTO__", default="__AUTO__",
                    help="session (LLM methods only): a short session code. AUTO-generated by default (every LLM run is a promotable session); --session CODE resumes that thread; --no-session opts out. Runs land in results/sessions/ and re-running OVERWRITES in place (continuous curve). The report groups by --period; the session shows as `strategy @code`. Relies on FINPANEL_LLM_CACHE=1 to make the re-walk near-free.")
    ap.add_argument("--no-session", action="store_true", help="disable the default auto-session (plain one-off run → results/<date>/ only, not promotable). Non-LLM baselines ignore sessions regardless.")
    ap.add_argument("--cache-from", default=None, metavar="CODE[,CODE...]",
                    help="inherit frozen LLM responses from earlier session(s) — READ-ONLY, first code wins on a tie. A CODE resolves to results/sessions/<CODE>/llm_cache; a value containing a path separator is used verbatim. Inherited hits are COPIED into THIS session's cache, so it stays a self-contained, independently reproducible record. Entries are content-addressed (model+prompt+tags), so a changed prompt/model/brief simply misses — inheriting can never serve a wrong answer. Typical use: run the warm-up once, then start each arm with --cache-from <warmup-code> so every (actor, day) judgment is reused and only the manager re-runs.")
    ap.add_argument("--stop-date", default=None, help="end the run early once the sim reaches this date (the --period metadata stays the full window). Advance a session in chunks by bumping it forward.")
    args = ap.parse_args()

    strategies = [s.strip().lower() for s in args.strategy.split(",") if s.strip()]
    import yaml as _yaml
    args.overrides = {}
    for kv in args.set_overrides:
        if "=" not in kv:
            raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        args.overrides[k.strip()] = _yaml.safe_load(v)
    tc_label = f"panel{build_config(args.overrides).get('panel_size', 6)}"
    unknown = [s for s in strategies if s not in STRATEGIES and s not in RL_NAMES and s not in LLM_NAMES and s not in FINPANEL_ALIASES]
    if unknown:
        raise SystemExit(f"Unknown --strategy {unknown}. Choose from: {', '.join(strat_choices)}")

    if args.period:
        start, end, period_id = resolve_period(args.period)
    elif args.start_date and args.end_date:
        start, end, period_id = args.start_date, args.end_date, "custom"
    else:
        raise SystemExit("Pass --period NAME (or both --start-date and --end-date).")
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    run_start = start
    if args.warmup_start:
        run_start = args.warmup_start
        if not args.eval_start_date:
            args.eval_start_date = start
        print(f"🔥 warm-up: run from {run_start} (fixed anchor), score from {args.eval_start_date}")
    elif args.warmup_days and int(args.warmup_days) > 0:
        from datetime import date as _date, timedelta as _timedelta
        run_start = (_date.fromisoformat(start) - _timedelta(days=int(args.warmup_days))).isoformat()
        if not args.eval_start_date:
            args.eval_start_date = start
        print(f"🔥 warm-up: run from {run_start}, score from {args.eval_start_date} ({args.warmup_days}d pre-window)")

    from utils import report

    if args.no_session:
        args.session = None
    elif args.session == "__AUTO__":
        args.session = secrets.token_hex(2)
        print(f"🔖 new session code: {args.session}  (resume with --session {args.session})")
    elif args.session:
        print(f"🔖 resuming session: {args.session}")

    _inherit: list[str] = []
    if args.cache_from:
        for _code in [c.strip() for c in str(args.cache_from).split(",") if c.strip()]:
            _p = Path(_code) if os.sep in _code else Path("results/sessions") / _code / "llm_cache"
            if _code == args.session:
                print(f"  ⚠️ --cache-from {_code} is this run's own session — ignored (it already reads that dir).")
            elif not _p.exists():
                print(f"  ⚠️ --cache-from {_code}: nothing at {_p} — inheriting NOTHING from it (typo?).")
            else:
                _inherit.append(str(_p))
    if _inherit:
        os.environ.setdefault("FINPANEL_LLM_CACHE", "1")
        os.environ["FINPANEL_LLM_CACHE_READ_DIRS"] = os.pathsep.join(_inherit)
        print(f"  ♻️ inheriting LLM cache (read-only) from: {', '.join(_inherit)}")

    if args.session:
        os.environ.setdefault("FINPANEL_LLM_CACHE", "1")
        _sess_root = f"results/sessions/{args.session}"
        _cache_dir = f"{_sess_root}/llm_cache"
        os.environ["FINPANEL_LLM_CACHE_DIR"] = _cache_dir
        _session_manifest(args, _sess_root, strategies, [tc_label], period_id, start, end, inherits=_inherit)

    print(f"=== {','.join(strategies)} | {period_id} {start}→{end} | tickers={tickers} | profile={args.execution_profile} ===")

    rc = 0
    produced: list[str] = []
    for s in strategies:
        if s in FINPANEL_ALIASES:
            _rc, _paths = _run_finpanel(args, tickers, run_start, end)
            rc = _rc or rc
            produced.extend(_paths)
        else:
            _rc, _paths = _run_classic(s, args, tickers, run_start, end)
            rc = _rc or rc
            produced.extend(_paths)

    new = sorted(set(produced))
    if new:
        report.print_console_table(new)
        report.upsert_results(new, period_id, start, end)
    sys.exit(rc)


if __name__ == "__main__":
    main()
