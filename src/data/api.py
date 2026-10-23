import os
import pickle
import pandas as pd
from datetime import timedelta
from pathlib import Path

from data.models import (
    CompanyNews,
    DataAuditRecord,
    FinancialMetrics,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_SELECTED_PKL_CANDIDATES = [
    _PROJECT_ROOT / "data" / "stock_data_selected8_2020_2025_finsaberv2.pkl",
    _PROJECT_ROOT / "data" / "stock_data_selected4_2020_2025_finsaberv2.pkl",
    _PROJECT_ROOT / "data" / "stock_data_cherrypick_2000_2024_v2.pkl",
]


def resolve_data_pkl() -> Path:
    override = os.environ.get("DATA_PKL")
    if override:
        return Path(override)
    return next((p for p in _SELECTED_PKL_CANDIDATES if p.exists()), _SELECTED_PKL_CANDIDATES[0])


_FROZEN_SELECTED_PKL = resolve_data_pkl()
_FROZEN_SP500_PKL = _PROJECT_ROOT / "data" / "stock_data_sp500_2000_2024_v2.pkl"
_FROZEN_SP500_CSV = _PROJECT_ROOT / "all_sp500_prices_2000_2024_delisted_include.csv"

_FROZEN_SELECTED_DATA: dict | None = None
_FROZEN_SP500_DATA: dict | None = None
_FROZEN_SP500_PRICES_DF: pd.DataFrame | None = None
_RUNTIME_PRICE_CONTEXT = {
    "universe_mode": "static",
    "data_cutoff_mode": "close_of_day",
    "decision_time": None,
    "disclosure_lag_days": 0,
    "audit_log": None,
}


def set_runtime_price_context(
    universe_mode: str | None = None,
    data_cutoff_mode: str | None = None,
    decision_time: str | None = None,
    disclosure_lag_days: int | None = None,
    audit_log: list[dict] | None = None,
) -> None:

    if universe_mode is not None:
        _RUNTIME_PRICE_CONTEXT["universe_mode"] = universe_mode
    if data_cutoff_mode is not None:
        _RUNTIME_PRICE_CONTEXT["data_cutoff_mode"] = data_cutoff_mode
    _RUNTIME_PRICE_CONTEXT["decision_time"] = decision_time
    if disclosure_lag_days is not None:
        _RUNTIME_PRICE_CONTEXT["disclosure_lag_days"] = int(disclosure_lag_days)
    _RUNTIME_PRICE_CONTEXT["audit_log"] = audit_log


def _load_pickle_dataset(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj if isinstance(obj, dict) else {}


def _get_frozen_selected_data() -> dict:
    global _FROZEN_SELECTED_DATA
    if _FROZEN_SELECTED_DATA is None:
        _FROZEN_SELECTED_DATA = _load_pickle_dataset(_FROZEN_SELECTED_PKL)
    return _FROZEN_SELECTED_DATA


def _get_frozen_sp500_data() -> dict:
    global _FROZEN_SP500_DATA
    if _FROZEN_SP500_DATA is None:
        if _FROZEN_SP500_PKL.exists() and _FROZEN_SP500_PKL.stat().st_size > 5 * 1024 * 1024 * 1024:
            print(f"⚠️ Skipping {_FROZEN_SP500_PKL.name} ({_FROZEN_SP500_PKL.stat().st_size / 1024**3:.1f}GB) to avoid OOM. Using CSV fallback.")
            _FROZEN_SP500_DATA = {}
        else:
            _FROZEN_SP500_DATA = _load_pickle_dataset(_FROZEN_SP500_PKL)
    return _FROZEN_SP500_DATA


def _get_frozen_sp500_prices_df() -> pd.DataFrame:
    global _FROZEN_SP500_PRICES_DF
    if _FROZEN_SP500_PRICES_DF is None:
        if not _FROZEN_SP500_CSV.exists():
            _FROZEN_SP500_PRICES_DF = pd.DataFrame()
        else:
            df = pd.read_csv(_FROZEN_SP500_CSV)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].astype(str)
            _FROZEN_SP500_PRICES_DF = df.dropna(subset=["date"]).copy()
    return _FROZEN_SP500_PRICES_DF


def _normalize_date_str(date_str: str) -> str:
    return pd.to_datetime(date_str).strftime("%Y-%m-%d")


def _apply_cutoff_window(
    start_date: str,
    end_date: str,
    data_cutoff_mode: str = "close_of_day",
    decision_time: str | None = None,
    disclosure_lag_days: int = 0,
) -> tuple[str, str, str]:

    start_norm = _normalize_date_str(start_date)
    end_norm = _normalize_date_str(end_date)
    decision_norm = _normalize_date_str(decision_time) if decision_time else end_norm

    if data_cutoff_mode == "t_minus_1":
        allowed_cutoff_dt = pd.to_datetime(decision_norm) - timedelta(days=1)
    elif data_cutoff_mode == "disclosure_lag_n":
        lag_days = max(int(disclosure_lag_days or 0), 0)
        allowed_cutoff_dt = pd.to_datetime(decision_norm) - timedelta(days=lag_days)
    else:
        allowed_cutoff_dt = pd.to_datetime(end_norm)

    allowed_cutoff = allowed_cutoff_dt.strftime("%Y-%m-%d")
    effective_end = min(end_norm, allowed_cutoff)
    return start_norm, effective_end, allowed_cutoff


def _prices_from_pickle_dataset(dataset: dict, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    if not dataset:
        return pd.DataFrame()

    rows = []
    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()

    for d in sorted(dataset.keys()):
        if d < start_dt or d > end_dt:
            continue
        day_data = dataset.get(d, {}) or {}
        price_map = day_data.get("price", {}) or {}
        if ticker not in price_map:
            continue
        px = price_map[ticker] or {}
        close = px.get("adjusted_close", px.get("close"))
        if close is None:
            continue
        rows.append(
            {
                "Date": pd.to_datetime(d),
                "open": float(px.get("adjusted_open", px.get("open", close))),
                "high": float(px.get("adjusted_high", px.get("high", close))),
                "low": float(px.get("adjusted_low", px.get("low", close))),
                "close": float(close),
                "volume": int(px.get("volume", 0) or 0),
            }
        )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df


def _prices_from_dynamic_csv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = _get_frozen_sp500_prices_df()
    if df.empty:
        return pd.DataFrame()

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    filtered = df[
        (df["symbol"] == ticker)
        & (df["date"] >= start_dt)
        & (df["date"] <= end_dt)
    ].copy()

    if filtered.empty:
        return pd.DataFrame()

    filtered["Date"] = pd.to_datetime(filtered["date"])
    result = filtered.set_index("Date").sort_index()
    if "adjusted_close" in result.columns and "close" in result.columns:
        result["close"] = pd.to_numeric(result["close"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in result.columns:
            result[col] = 0.0 if col != "volume" else 0
    return result[["open", "high", "low", "close", "volume"]]


def _append_data_audit_record(
    audit_log: list[dict] | None,
    ticker: str,
    source: str,
    requested_start: str,
    requested_end: str,
    decision_time: str,
    allowed_cutoff: str,
    price_df: pd.DataFrame,
) -> None:
    if audit_log is None:
        return

    data_max_time = None
    violation = False
    if not price_df.empty:
        data_max_time = pd.to_datetime(price_df.index.max()).strftime("%Y-%m-%d")
        violation = data_max_time > allowed_cutoff

    record = DataAuditRecord(
        ticker=ticker,
        source=source,
        requested_start=requested_start,
        requested_end=requested_end,
        decision_time=decision_time,
        allowed_cutoff=allowed_cutoff,
        data_max_time=data_max_time,
        violation=violation,
    )
    audit_log.append(record.model_dump())


def _append_data_audit_record_with_time(
    audit_log: list[dict] | None,
    ticker: str,
    source: str,
    requested_start: str,
    requested_end: str,
    decision_time: str,
    allowed_cutoff: str,
    data_max_time: str | None,
) -> None:
    if audit_log is None:
        return

    violation = bool(data_max_time and data_max_time > allowed_cutoff)
    record = DataAuditRecord(
        ticker=ticker,
        source=source,
        requested_start=requested_start,
        requested_end=requested_end,
        decision_time=decision_time,
        allowed_cutoff=allowed_cutoff,
        data_max_time=data_max_time,
        violation=violation,
    )
    audit_log.append(record.model_dump())


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    data_cutoff_mode: str | None = None,
    decision_time: str | None = None,
    disclosure_lag_days: int = 0,
    audit_log: list[dict] | None = None,
) -> list[FinancialMetrics]:

    resolved_cutoff_mode = data_cutoff_mode or _RUNTIME_PRICE_CONTEXT.get("data_cutoff_mode", "close_of_day")
    resolved_decision_time = decision_time if decision_time is not None else _RUNTIME_PRICE_CONTEXT.get("decision_time")
    resolved_disclosure_lag_days = (
        disclosure_lag_days
        if disclosure_lag_days
        else int(_RUNTIME_PRICE_CONTEXT.get("disclosure_lag_days", 0) or 0)
    )
    resolved_audit_log = audit_log if audit_log is not None else _RUNTIME_PRICE_CONTEXT.get("audit_log")

    requested_end = _normalize_date_str(end_date)
    _, effective_end, allowed_cutoff = _apply_cutoff_window(
        start_date=end_date,
        end_date=end_date,
        data_cutoff_mode=resolved_cutoff_mode,
        decision_time=resolved_decision_time,
        disclosure_lag_days=resolved_disclosure_lag_days,
    )
    end_norm = effective_end

    candidates: list[tuple[str, dict]] = []
    source_name = "frozen_financial_metrics"
    for source_name, dataset in (
        ("frozen_selected_pkl_financial_metrics", _get_frozen_selected_data()),
        ("frozen_sp500_pkl_financial_metrics", _get_frozen_sp500_data()),
    ):
        if not dataset:
            continue
        for d in sorted(dataset.keys(), reverse=True):
            date_str = pd.to_datetime(d).strftime("%Y-%m-%d")
            if date_str > end_norm:
                continue
            day_data = dataset.get(d, {}) or {}
            price_map = day_data.get("price", {}) or {}
            if ticker in price_map:
                candidates.append((date_str, price_map[ticker] or {}))
                break
        if candidates:
            break

    if not candidates:
        _append_data_audit_record_with_time(
            audit_log=resolved_audit_log,
            ticker=ticker,
            source=source_name,
            requested_start=requested_end,
            requested_end=requested_end,
            decision_time=resolved_decision_time or requested_end,
            allowed_cutoff=allowed_cutoff,
            data_max_time=None,
        )
        return []

    report_period, px = candidates[0]
    close = float(px.get("close", 0.0) or 0.0)
    volume = float(px.get("volume", 0.0) or 0.0)
    market_cap = close * volume if close > 0 and volume > 0 else None

    payload = {name: None for name in FinancialMetrics.model_fields.keys()}
    payload.update(
        {
            "ticker": ticker,
            "report_period": report_period,
            "period": period,
            "currency": "USD",
            "market_cap": market_cap,
        }
    )
    _append_data_audit_record_with_time(
        audit_log=resolved_audit_log,
        ticker=ticker,
        source=source_name,
        requested_start=requested_end,
        requested_end=requested_end,
        decision_time=resolved_decision_time or requested_end,
        allowed_cutoff=allowed_cutoff,
        data_max_time=report_period,
    )
    return [FinancialMetrics(**payload)][:limit]


def get_company_news(
    ticker: str,
    start_date: str,
    end_date: str,
    limit: int = 1000,
    data_cutoff_mode: str | None = None,
    decision_time: str | None = None,
    disclosure_lag_days: int = 0,
    audit_log: list[dict] | None = None,
) -> list[CompanyNews]:

    resolved_cutoff_mode = data_cutoff_mode or _RUNTIME_PRICE_CONTEXT.get("data_cutoff_mode", "close_of_day")
    resolved_decision_time = decision_time if decision_time is not None else _RUNTIME_PRICE_CONTEXT.get("decision_time")
    resolved_disclosure_lag_days = (
        disclosure_lag_days
        if disclosure_lag_days
        else int(_RUNTIME_PRICE_CONTEXT.get("disclosure_lag_days", 0) or 0)
    )
    resolved_audit_log = audit_log if audit_log is not None else _RUNTIME_PRICE_CONTEXT.get("audit_log")

    start_norm, end_norm, allowed_cutoff = _apply_cutoff_window(
        start_date=start_date,
        end_date=end_date,
        data_cutoff_mode=resolved_cutoff_mode,
        decision_time=resolved_decision_time,
        disclosure_lag_days=resolved_disclosure_lag_days,
    )

    if pd.to_datetime(end_norm) < pd.to_datetime(start_norm):
        _append_data_audit_record_with_time(
            audit_log=resolved_audit_log,
            ticker=ticker,
            source="frozen_pickle_news",
            requested_start=_normalize_date_str(start_date),
            requested_end=_normalize_date_str(end_date),
            decision_time=resolved_decision_time or _normalize_date_str(end_date),
            allowed_cutoff=allowed_cutoff,
            data_max_time=None,
        )
        return []

    collected: list[CompanyNews] = []
    max_news_date: str | None = None

    for dataset in (_get_frozen_selected_data(), _get_frozen_sp500_data()):
        if not dataset:
            continue
        for d in sorted(dataset.keys()):
            day_str = pd.to_datetime(d).strftime("%Y-%m-%d")
            if day_str < start_norm or day_str > end_norm:
                continue
            day_data = dataset.get(d, {}) or {}
            news_map = day_data.get("news", {}) or {}
            if ticker not in news_map:
                continue
            items = news_map.get(ticker) or []
            if not isinstance(items, list):
                continue
            for idx, item in enumerate(items):
                title = str(item).strip()
                if not title:
                    continue
                collected.append(
                    CompanyNews(
                        ticker=ticker,
                        title=title,
                        author="unknown",
                        source="frozen_pickle",
                        date=day_str,
                        url=f"frozen://{ticker}/{day_str}/{idx}",
                        sentiment=None,
                    )
                )
                max_news_date = day_str
                if len(collected) >= limit:
                    _append_data_audit_record_with_time(
                        audit_log=resolved_audit_log,
                        ticker=ticker,
                        source="frozen_pickle_news",
                        requested_start=_normalize_date_str(start_date),
                        requested_end=_normalize_date_str(end_date),
                        decision_time=resolved_decision_time or _normalize_date_str(end_date),
                        allowed_cutoff=allowed_cutoff,
                        data_max_time=max_news_date,
                    )
                    return collected

    _append_data_audit_record_with_time(
        audit_log=resolved_audit_log,
        ticker=ticker,
        source="frozen_pickle_news",
        requested_start=_normalize_date_str(start_date),
        requested_end=_normalize_date_str(end_date),
        decision_time=resolved_decision_time or _normalize_date_str(end_date),
        allowed_cutoff=allowed_cutoff,
        data_max_time=max_news_date,
    )
    return collected

def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    universe_mode: str | None = None,
    data_cutoff_mode: str | None = None,
    decision_time: str | None = None,
    disclosure_lag_days: int = 0,
    audit_log: list[dict] | None = None,
) -> pd.DataFrame:

    resolved_universe_mode = universe_mode or _RUNTIME_PRICE_CONTEXT.get("universe_mode", "static")
    resolved_cutoff_mode = data_cutoff_mode or _RUNTIME_PRICE_CONTEXT.get("data_cutoff_mode", "close_of_day")
    resolved_decision_time = decision_time if decision_time is not None else _RUNTIME_PRICE_CONTEXT.get("decision_time")
    resolved_disclosure_lag_days = (
        disclosure_lag_days
        if disclosure_lag_days
        else int(_RUNTIME_PRICE_CONTEXT.get("disclosure_lag_days", 0) or 0)
    )
    resolved_audit_log = audit_log if audit_log is not None else _RUNTIME_PRICE_CONTEXT.get("audit_log")

    start_norm, end_effective, allowed_cutoff = _apply_cutoff_window(
        start_date=start_date,
        end_date=end_date,
        data_cutoff_mode=resolved_cutoff_mode,
        decision_time=resolved_decision_time,
        disclosure_lag_days=resolved_disclosure_lag_days,
    )

    if pd.to_datetime(end_effective) < pd.to_datetime(start_norm):
        _append_data_audit_record(
            audit_log=resolved_audit_log,
            ticker=ticker,
            source="frozen_none",
            requested_start=start_norm,
            requested_end=end_date,
            decision_time=resolved_decision_time or end_date,
            allowed_cutoff=allowed_cutoff,
            price_df=pd.DataFrame(),
        )
        return pd.DataFrame()

    if resolved_universe_mode == "dynamic":
        price_df = _prices_from_dynamic_csv(ticker, start_norm, end_effective)
        source_name = "frozen_sp500_csv_dynamic"
    else:
        price_df = _prices_from_pickle_dataset(_get_frozen_selected_data(), ticker, start_norm, end_effective)
        source_name = "frozen_selected_pkl_static"
        if price_df.empty:
            price_df = _prices_from_pickle_dataset(_get_frozen_sp500_data(), ticker, start_norm, end_effective)
            source_name = "frozen_sp500_pkl_static"

    _append_data_audit_record(
        audit_log=resolved_audit_log,
        ticker=ticker,
        source=source_name,
        requested_start=start_norm,
        requested_end=end_date,
        decision_time=resolved_decision_time or end_date,
        allowed_cutoff=allowed_cutoff,
        price_df=price_df,
    )
    return price_df

