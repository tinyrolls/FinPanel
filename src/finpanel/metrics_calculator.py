from typing import Dict, Any
from datetime import datetime, timedelta
import pandas as pd

from finpanel.state import AgentState
from data import api

import os

USE_FUNDAMENTAL_DATA = os.environ.get("USE_FUNDAMENTAL_DATA", "true").lower() != "false"


def get_basic_metrics(prices_df: pd.DataFrame, fundamental_data: dict) -> Dict[str, Any]:
    if prices_df.empty:
        return {"error": "No price data available"}
    try:
        closes = prices_df["close"]

        def pct_n_days(n: int):
            if len(closes) <= n:
                return None
            base = closes.iloc[-(n + 1)]
            if base is None or float(base) == 0:
                return None
            return round((float(closes.iloc[-1]) / float(base) - 1.0) * 100.0, 2)

        returns = closes.pct_change().dropna()
        return {
            "current_price": closes.iloc[-1],
            "price_pct_1d": pct_n_days(1),
            "price_pct_5d": pct_n_days(5),
            "price_pct_10d": pct_n_days(10),
            "price_pct_30d": pct_n_days(30),
            "volatility_30d": returns.tail(30).std() * (252**0.5) if len(returns) > 0 else 0,
            "avg_volume_30d": prices_df["volume"].tail(30).mean() if "volume" in prices_df.columns else 0,
            "current_volume": prices_df["volume"].iloc[-1] if "volume" in prices_df.columns else 0,
            "market_cap": fundamental_data.get("market_cap", 0),
            "pe_ratio": fundamental_data.get("pe_ratio", 0),
            "data_quality": "basic",
        }
    except Exception as e:
        return {"error": f"Metrics calculation failed: {str(e)}"}


def metrics_calculator_agent(state: AgentState) -> Dict[str, Any]:
    print("--- Calculating Metrics (basic + fundamentals) ---")
    data = state["data"]
    tickers = data["tickers"]
    end_date = data["end_date"]
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")

    metrics_by_ticker = {}
    fundamental_by_ticker = data.get("fundamental_analysis", {}) or {}
    _fundamental_warning_shown = False

    for ticker in tickers:
        print(f"Calculating metrics for: {ticker}")
        try:
            prices_df = api.get_price_data(ticker, start_date, end_date)

            if prices_df.empty:
                print(f"Warning: No price data found for {ticker}.")
                metrics_by_ticker[ticker] = {"error": "No price data available"}
                continue

            fundamental_data = {}
            if USE_FUNDAMENTAL_DATA:
                try:
                    fundamentals = api.get_financial_metrics(ticker, end_date, period="ttm", limit=1)
                    if fundamentals:
                        fundamental_data = (
                            fundamentals[0].model_dump()
                            if hasattr(fundamentals[0], "model_dump")
                            else fundamentals[0]
                        )
                except Exception as e:
                    fundamental_data = fundamental_by_ticker.get(ticker, {}) or {}
                    if not _fundamental_warning_shown:
                        print(f"⚠️ Fundamental data API fallback: {e}")
                        _fundamental_warning_shown = True
            else:
                if not _fundamental_warning_shown:
                    print("ℹ️ Fundamental data disabled — pure technical mode")
                    _fundamental_warning_shown = True

            if not fundamental_data:
                fundamental_data = fundamental_by_ticker.get(ticker, {}) or {}
            fundamental_by_ticker[ticker] = fundamental_data

            metrics_by_ticker[ticker] = {
                "basic_metrics": get_basic_metrics(prices_df, fundamental_data),
                "fundamental_analysis": fundamental_data,
            }
            print(f"✅ Metrics for {ticker}")

        except Exception as e:
            print(f"Error calculating metrics for {ticker}: {e}")
            metrics_by_ticker[ticker] = {"error": str(e)}

    data["metrics_by_ticker"] = metrics_by_ticker
    data["fundamental_analysis"] = fundamental_by_ticker

    return {"data": data}
