from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from data.fundamentals_briefs import ensure_briefs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from data.api import resolve_data_pkl
    ap.add_argument("--data", default=str(resolve_data_pkl()))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "fundamentals_briefs.jsonl"))
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--provider", default="OpenAI")
    ap.add_argument("--tickers", nargs="+", default=None, help="Restrict to these tickers (default: all in data).")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of filings (smoke test).")
    args = ap.parse_args()

    ensure_briefs(
        args.data, args.out,
        model=args.model, provider=args.provider,
        tickers=args.tickers, limit=args.limit, verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
