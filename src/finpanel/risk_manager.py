import json
from datetime import datetime, timedelta

from data import api
from finpanel.state import AgentState, show_agent_reasoning
from utils.progress import progress


def risk_management_agent(state: AgentState) -> AgentState:
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    risk_limits = {}
    current_prices = {}

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Marking to market")

        prices_df = api.get_price_data(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
        )
        if prices_df.empty:
            try:
                end_dt = datetime.strptime(data["end_date"], "%Y-%m-%d")
                fallback_start = (end_dt - timedelta(days=7)).strftime("%Y-%m-%d")
                prices_df = api.get_price_data(
                    ticker=ticker,
                    start_date=fallback_start,
                    end_date=data["end_date"],
                )
            except Exception:
                pass
        if prices_df.empty:
            progress.update_status("risk_management_agent", ticker, "Failed: No price data")
            continue

        try:
            current_prices[ticker] = float(prices_df["close"].iloc[-1])
        except Exception:
            current_prices[ticker] = 0.0
        risk_limits[ticker] = {"ticker": ticker, "current_price": current_prices[ticker]}
        progress.update_status("risk_management_agent", ticker, "Done")

    total_portfolio_value = float(portfolio.get("cash", 0) or 0)
    for ticker, position in (portfolio.get("positions", {}) or {}).items():
        if ticker in current_prices:
            total_portfolio_value += float(position.get("shares", 0) or 0) * current_prices[ticker]
    total_portfolio_value = max(total_portfolio_value, 0.01)

    summary = {
        "portfolio_metrics": {
            "total_value": total_portfolio_value,
            "cash": portfolio.get("cash", 0),
        },
        "risk_limits_by_ticker": risk_limits,
    }

    message = {
        "role": "user",
        "content": json.dumps(summary, indent=2, default=str),
        "name": "risk_management_agent",
    }

    if state.get("metadata", {}).get("show_reasoning", False):
        show_agent_reasoning(summary, "Position Context (Rule-Based)")

    state["data"]["analyst_signals"]["risk_management_agent"] = summary

    return {
        "messages": list(state["messages"]) + [message],
        "data": data,
        "metadata": state.get("metadata", {}),
    }
