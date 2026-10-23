from __future__ import annotations

from typing import Dict, List, Tuple

COMMISSION_PER_SHARE = 0.0049
COMMISSION_MIN = 0.99
COMMISSION_MAX_RATE = 0.01


def moomoo_commission(shares: float, price: float) -> float:
    n = abs(shares) * price
    if n <= 0:
        return 0.0
    return min(max(COMMISSION_PER_SHARE * abs(shares), COMMISSION_MIN), COMMISSION_MAX_RATE * n)


def finsaber_buy_size(cash: float, price: float) -> int:
    if price <= 0 or cash <= 0:
        return 0
    size = int(cash / price)
    while size > 0:
        comm = moomoo_commission(size, price)
        if cash >= size * price + comm:
            return size
        size = int((cash - comm) / price)
    return 0


def execute_target_positions(
    tickers: List[str],
    price_map: Dict[str, float],
    positions: Dict[str, dict],
    target_shares_by_ticker: Dict[str, float],
    total_value: float,
    cash: float,
    slippage_bps: float = 5.0,
    short_borrow_apr: float = 0.03,
) -> Tuple[float, Dict[str, dict], Dict[str, float], Dict[str, dict]]:
    slippage_rate = slippage_bps / 10000.0

    total_traded_notional = 0.0
    total_commission = 0.0
    total_slippage = 0.0
    actions: Dict[str, dict] = {}

    for t in tickers:
        current_price = float(price_map.get(t, 0.0) or 0.0)
        pos = positions.get(t, {}) or {}
        current_shares = float(pos.get("shares", 0.0) or 0.0)

        trade_shares = 0
        target_shares = current_shares
        if current_price > 0:
            target_shares = float(int(target_shares_by_ticker.get(t, current_shares)))
            trade_shares = target_shares - current_shares

        trade_notional = abs(trade_shares) * current_price if current_price > 0 else 0.0
        commission_cost = min(max(0.0049 * abs(trade_shares), 0.99), 0.01 * trade_notional) if trade_notional > 0 else 0.0
        slippage_cost = trade_notional * slippage_rate if trade_notional > 0 else 0.0

        if t not in positions:
            positions[t] = {"shares": 0, "cost_basis": 0.0, "margin_used": 0.0}
        positions[t]["shares"] = target_shares

        if current_price > 0 and trade_shares != 0:
            if trade_shares > 0:
                cash -= trade_notional + commission_cost + slippage_cost
            else:
                cash += trade_notional - commission_cost - slippage_cost
            total_traded_notional += trade_notional
            total_commission += commission_cost
            total_slippage += slippage_cost

        if trade_shares > 0:
            action = "cover" if current_shares < 0 else "buy"
        elif trade_shares < 0:
            action = "short" if (target_shares < 0 and current_shares <= 0) else "sell"
        else:
            action = "hold"
        actions[t] = {"action": action, "quantity": abs(trade_shares), "commission": commission_cost, "slippage": slippage_cost}

    total_short_notional = 0.0
    for t in tickers:
        px = float(price_map.get(t, 0.0) or 0.0)
        sh = float((positions.get(t, {}) or {}).get("shares", 0.0) or 0.0)
        if px > 0 and sh < 0:
            total_short_notional += abs(sh) * px
    borrow_fee_paid = total_short_notional * short_borrow_apr / 252.0 if total_short_notional > 0 else 0.0
    if borrow_fee_paid > 0:
        cash -= borrow_fee_paid

    daily_turnover = float(total_traded_notional / total_value) if total_value > 0 else 0.0
    costs = {
        "commission_paid": float(total_commission),
        "slippage_paid": float(total_slippage),
        "borrow_fee_paid": float(borrow_fee_paid),
        "total_cost_paid": float(total_commission + total_slippage + borrow_fee_paid),
        "traded_notional": float(total_traded_notional),
        "turnover": daily_turnover,
    }
    return cash, positions, costs, actions
