from __future__ import annotations

import math
from typing import Optional, Sequence

RISK_FREE_RATE_ANNUAL = 0.03
TRADING_DAYS_PER_YEAR = 252.0


def compute_perf_stats(
    values: Sequence[float],
    rf_annual: float = RISK_FREE_RATE_ANNUAL,
    initial_value: Optional[float] = None,
) -> dict:
    vals = [float(v) for v in values]
    out: dict = {
        "cr": None, "ar": None, "av": None,
        "sharpe": None, "sortino": None, "mdd": None,
        "n_returns": 0,
    }
    if len(vals) < 2:
        return out

    base = float(initial_value) if initial_value else vals[0]
    if base > 0:
        out["cr"] = vals[-1] / base - 1.0

    peak = vals[0]
    mdd = 0.0
    for v in vals:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    out["mdd"] = mdd

    returns: list[float] = []
    for prev, curr in zip(vals, vals[1:]):
        if prev != 0:
            returns.append(curr / prev - 1.0)
    T = len(returns)
    out["n_returns"] = T
    if T < 1:
        return out

    if out["cr"] is not None:
        out["ar"] = (1.0 + out["cr"]) ** (TRADING_DAYS_PER_YEAR / T) - 1.0

    if T < 2:
        return out

    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    mean_ret = sum(returns) / T
    mean_excess = mean_ret - rf_daily
    variance = sum((r - mean_ret) ** 2 for r in returns) / (T - 1)
    std_ret = math.sqrt(max(variance, 0.0))
    out["av"] = std_ret * math.sqrt(TRADING_DAYS_PER_YEAR)
    if std_ret > 0:
        out["sharpe"] = (mean_excess / std_ret) * math.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        out["sharpe"] = 0.0
        out["sortino"] = 0.0
        return out

    downside = [r - rf_daily for r in returns if (r - rf_daily) < 0]
    if len(downside) > 1:
        d_mean = sum(downside) / len(downside)
        d_var = sum((x - d_mean) ** 2 for x in downside) / (len(downside) - 1)
        d_std = math.sqrt(max(d_var, 0.0))
        if d_std > 1e-12:
            out["sortino"] = (mean_excess / d_std) * math.sqrt(TRADING_DAYS_PER_YEAR)

    return out
