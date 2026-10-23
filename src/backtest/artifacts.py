from __future__ import annotations

import csv
import os
from datetime import datetime

from evaluation.perf_metrics import compute_perf_stats
from utils.logger import save_json


def compute_performance_metrics(
    equity_curve: list[dict],
    initial_value: float,
) -> tuple[float, float | None, float | None, float | None, float | None, float | None]:
    if not equity_curve:
        return 0.0, None, None, None, None, None

    values = [float(item.get("total_value", 0.0) or 0.0) for item in equity_curve]
    ending_value = values[-1]
    total_return_pct = ((ending_value / initial_value - 1.0) * 100.0) if initial_value > 0 else 0.0

    stats = compute_perf_stats(values, initial_value=initial_value)
    sharpe_ratio = stats["sharpe"]
    sortino_ratio = stats["sortino"]
    annualized_return_pct = stats["ar"] * 100.0 if stats["ar"] is not None else None
    annualized_volatility_pct = stats["av"] * 100.0 if stats["av"] is not None else None
    max_drawdown_pct = stats["mdd"] * 100.0 if stats["mdd"] is not None else 0.0

    return total_return_pct, sharpe_ratio, sortino_ratio, max_drawdown_pct, annualized_return_pct, annualized_volatility_pct


def slice_equity_curve_for_evaluation(
    equity_curve: list[dict],
    evaluation_start_date: str | None,
) -> tuple[list[dict], float | None, int]:
    if not evaluation_start_date or not equity_curve:
        return list(equity_curve), None, 0
    eval_str = str(evaluation_start_date)[:10]
    first_test_idx = None
    for i, row in enumerate(equity_curve):
        if str(row.get("date", ""))[:10] >= eval_str:
            first_test_idx = i
            break
    if first_test_idx is None:
        return [], None, -1
    return list(equity_curve[first_test_idx:]), None, first_test_idx


def compute_test_metrics(
    daily_equity_curve: list[dict],
    evaluation_start_date: str | None,
    initial_total_value: float,
) -> dict | None:
    if not evaluation_start_date:
        return None
    test_curve, anchor_value, first_idx = slice_equity_curve_for_evaluation(daily_equity_curve, evaluation_start_date)
    if first_idx is None:
        return None
    if first_idx == -1 or len(test_curve) < 2:
        return {
            "evaluation_start_date": evaluation_start_date,
            "test_trading_days": 0,
            "note": "evaluation_start_date past last trading day; no test data",
        }
    baseline = anchor_value if anchor_value is not None else initial_total_value
    (ret_pct, sharpe, sortino, mdd, ann_ret, ann_vol) = compute_performance_metrics(test_curve, float(baseline))
    ending = float(test_curve[-1].get("total_value", 0.0) or 0.0)
    return {
        "evaluation_start_date": evaluation_start_date,
        "test_initial_value": float(baseline),
        "test_ending_value": ending,
        "net_return_pct": float(ret_pct),
        "annualized_return_pct": float(ann_ret) if ann_ret is not None else None,
        "annualized_volatility_pct": float(ann_vol) if ann_vol is not None else None,
        "sharpe_ratio": float(sharpe) if sharpe is not None else None,
        "sortino_ratio": float(sortino) if sortino is not None else None,
        "max_drawdown_pct": float(mdd) if mdd is not None else None,
        "test_trading_days": len(test_curve),
        "warmup_trading_days": max(0, len(daily_equity_curve) - len(test_curve)),
    }


def save_equity_curve_csv(
    equity_curve: list[dict],
    out_path: str,
    run_metadata: dict | None = None,
    evaluation_start_date: str | None = None,
) -> None:
    run_metadata = run_metadata or {}
    eval_str = str(evaluation_start_date)[:10] if evaluation_start_date else None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "date",
            "phase",
            "cash_balance",
            "total_position_value",
            "total_value",
            "gross_total_value",
            "active_positions",
            "commission_paid",
            "slippage_paid",
            "borrow_fee_paid",
            "total_cost_paid",
            "cumulative_total_cost",
            "turnover",
            "cumulative_turnover",
            "run_id",
            "ablation_stage",
            "execution_profile",
            "execution_template",
            "protocol_version",
        ])
        for row in equity_curve:
            row_date = str(row.get("date", ""))[:10]
            if eval_str is None:
                phase = "test"
            else:
                phase = "warmup" if row_date < eval_str else "test"
            writer.writerow([
                row.get("date", ""),
                phase,
                row.get("cash_balance", 0.0),
                row.get("total_position_value", 0.0),
                row.get("total_value", 0.0),
                row.get("gross_total_value", row.get("total_value", 0.0)),
                row.get("active_positions", 0),
                row.get("commission_paid", 0.0),
                row.get("slippage_paid", 0.0),
                row.get("borrow_fee_paid", 0.0),
                row.get("total_cost_paid", 0.0),
                row.get("cumulative_total_cost", 0.0),
                row.get("turnover", 0.0),
                row.get("cumulative_turnover", 0.0),
                run_metadata.get("run_id", ""),
                run_metadata.get("ablation_stage", ""),
                run_metadata.get("execution_profile", ""),
                run_metadata.get("execution_template", ""),
                run_metadata.get("protocol_version", ""),
            ])


def save_data_audit_json(audit_path: str, records: list[dict]) -> dict:
    violation_count = sum(1 for r in records if bool(r.get("violation")))
    by_source: dict = {}
    for record in records:
        source = str(record.get("source", "unknown"))
        by_source[source] = by_source.get(source, 0) + 1

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "record_count": len(records),
        "violation_count": violation_count,
        "violation_rate": (violation_count / len(records)) if records else 0.0,
        "records_by_source": by_source,
        "records": records,
    }
    save_json(audit_path, payload)
    return {
        "record_count": len(records),
        "violation_count": violation_count,
        "violation_rate": (violation_count / len(records)) if records else 0.0,
        "records_by_source": by_source,
    }


def plot_equity_curve_from_csv(
    csv_path: str,
    plot_path: str,
    model_name_value: str,
    model_provider_value: str,
    summary_metrics: dict,
) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    curve_df = pd.read_csv(csv_path)
    if curve_df.empty:
        return

    curve_df["date"] = pd.to_datetime(curve_df["date"], errors="coerce")
    curve_df = curve_df.dropna(subset=["date"]).sort_values("date")
    if curve_df.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(curve_df["date"], curve_df["total_value"], linewidth=2, label="Portfolio Total Value")
    if "gross_total_value" in curve_df.columns:
        plt.plot(
            curve_df["date"],
            curve_df["gross_total_value"],
            linewidth=1.5,
            linestyle="--",
            label="Gross Value (Before Costs)",
        )
    plt.title("Backtest Portfolio Value Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.3)

    latest_total = float(summary_metrics.get("total_value", 0.0) or 0.0)
    latest_cash = float(summary_metrics.get("cash_balance", 0.0) or 0.0)
    latest_position = float(summary_metrics.get("total_position_value", 0.0) or 0.0)
    latest_return = float(summary_metrics.get("return_pct", 0.0) or 0.0)
    latest_gross_return = float(summary_metrics.get("gross_return_pct", 0.0) or 0.0)
    latest_sharpe = summary_metrics.get("sharpe_ratio")
    latest_sortino = summary_metrics.get("sortino_ratio")
    latest_max_dd = summary_metrics.get("max_drawdown")
    total_cost_paid = float(summary_metrics.get("total_cost_paid", 0.0) or 0.0)
    cumulative_turnover_value = float(summary_metrics.get("cumulative_turnover", 0.0) or 0.0)

    metrics_text = (
        f"Model: {model_provider_value} / {model_name_value}\n"
        f"Cash: ${latest_cash:,.2f}\n"
        f"Position: ${latest_position:,.2f}\n"
        f"Total: ${latest_total:,.2f}\n"
        f"Net Return: {latest_return:+.2f}%\n"
        f"Gross Return: {latest_gross_return:+.2f}%\n"
        f"Cumulative Cost: ${total_cost_paid:,.2f}\n"
        f"Cumulative Turnover: {cumulative_turnover_value:.2%}"
    )
    metrics_text += (
        f"\nSharpe: {latest_sharpe:.2f}" if latest_sharpe is not None else "\nSharpe: N/A"
    )
    metrics_text += (
        f"\nSortino: {latest_sortino:.2f}" if latest_sortino is not None else "\nSortino: N/A"
    )
    metrics_text += (
        f"\nMax Drawdown: {latest_max_dd:.2f}%" if latest_max_dd is not None else "\nMax Drawdown: N/A"
    )

    plt.legend(loc="upper left")
    plt.text(
        0.02,
        0.98,
        metrics_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def plot_cost_breakdown_from_csv(
    csv_path: str,
    plot_path: str,
    model_name_value: str,
    model_provider_value: str,
) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    curve_df = pd.read_csv(csv_path)
    if curve_df.empty:
        return

    curve_df["date"] = pd.to_datetime(curve_df["date"], errors="coerce")
    curve_df = curve_df.dropna(subset=["date"]).sort_values("date")
    if curve_df.empty:
        return

    commission = curve_df.get("commission_paid", 0.0)
    slippage = curve_df.get("slippage_paid", 0.0)
    borrow = curve_df.get("borrow_fee_paid", 0.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(curve_df["date"], curve_df.get("cumulative_total_cost", 0.0), color="#c0392b", linewidth=2)
    ax1.set_title("Cumulative Trading Cost")
    ax1.set_ylabel("Cost ($)")
    ax1.grid(True, alpha=0.3)

    ax2.stackplot(
        curve_df["date"],
        commission,
        slippage,
        borrow,
        labels=["Commission", "Slippage", "Borrow Fee"],
        alpha=0.8,
    )
    ax2.set_title("Daily Cost Breakdown")
    ax2.set_ylabel("Daily Cost ($)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")
    ax2.set_xlabel("Date")

    fig.suptitle(f"Cost Analysis - {model_provider_value}/{model_name_value}")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def count_final_directions(coordination_log: list) -> dict:
    counts: dict = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for c in coordination_log:
        d = c.get("final_direction")
        if d in counts:
            counts[d] += 1
    return counts


def finpanel_mechanism_surface(coordination_log: list) -> dict:
    n = len(coordination_log)
    spreads = [c.get("ballot_spread") for c in coordination_log
               if isinstance(c.get("ballot_spread"), (int, float))]
    disputed = sum(1 for c in coordination_log if c.get("disputed"))
    return {
        "n_decisions": n,
        "disputed_days": disputed,
        "disputed_rate": round(disputed / n, 4) if n else 0.0,
        "mean_ballot_spread": round(sum(spreads) / len(spreads), 4) if spreads else 0.0,
        "unanimous_days": sum(1 for c in coordination_log
                              if c.get("ballot_spread") == 0.0),
        "manager_final_direction_counts": count_final_directions(coordination_log),
    }

def compute_mechanism_surface(daily_equity_curve: list[dict]) -> dict:
    n = len(daily_equity_curve)
    if n == 0:
        return {
            "trade_days": 0, "trading_days": 0, "trade_day_ratio": None,
            "position_exposure_min": None, "position_exposure_max": None,
            "position_exposure_mean": None, "position_exposure_end": None,
            "mechanism_active": False,
        }
    eps = 1e-9
    trade_days = sum(1 for r in daily_equity_curve if abs(float(r.get("turnover", 0.0))) > eps)
    expo = []
    for r in daily_equity_curve:
        tv = float(r.get("total_value", 0.0) or 0.0)
        if abs(tv) > eps:
            expo.append(float(r.get("total_position_value", 0.0) or 0.0) / tv)
    ratio = trade_days / n
    return {
        "trade_days": trade_days,
        "trading_days": n,
        "trade_day_ratio": round(ratio, 4),
        "position_exposure_min": round(min(expo), 4) if expo else None,
        "position_exposure_max": round(max(expo), 4) if expo else None,
        "position_exposure_mean": round(sum(expo) / len(expo), 4) if expo else None,
        "position_exposure_end": round(expo[-1], 4) if expo else None,
        "mechanism_active": ratio >= 0.10,
    }
