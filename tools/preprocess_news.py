from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from data.news_retrieval import ensure_embeddings, ensure_news_selection


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from data.api import resolve_data_pkl
    ap.add_argument("--data", default=str(resolve_data_pkl()))
    ap.add_argument("--emb", default=str(PROJECT_ROOT / "data" / "news_embeddings.parquet"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "news_selection.jsonl"))
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--start", default=None, help="Inclusive start date YYYY-MM-DD (e.g. the warm-up anchor 2023-01-01).")
    ap.add_argument("--end", default=None, help="Inclusive end date YYYY-MM-DD (e.g. 2025-12-31).")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lookback", type=int, default=7)
    ap.add_argument("--select-only", action="store_true", help="Skip embedding; re-rank from frozen vectors (free).")
    args = ap.parse_args()

    import datetime as dt
    start = dt.date.fromisoformat(args.start) if args.start else None
    end = dt.date.fromisoformat(args.end) if args.end else None
    emb_start = (start - dt.timedelta(days=args.lookback)) if start else None

    if not args.select_only:
        ensure_embeddings(args.data, args.emb, tickers=args.tickers, start=emb_start, end=end, verbose=True)
    ensure_news_selection(
        args.data, args.out, args.emb,
        tickers=args.tickers, start=start, end=end, k=args.k, lookback_days=args.lookback, verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
