from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HF_REPO_ID = "finsaber-team/FINSABER-V2-Data"
MODALITIES = ("price", "news", "filing_k", "filing_q")

DEFAULT_TICKERS = ("TSLA", "NFLX", "AMZN", "MSFT", "COIN")


def _maybe_download(root: Path, start: date, end: date | None) -> None:
    if (root / "price_daily").exists():
        print(f"✓ parquet root already present: {root}")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub needed for --download. Install: uv pip install huggingface_hub"
        ) from exc
    end_year = end.year if end is not None else 2025
    allow = [f"*/year={y}/*" for y in range(start.year, end_year + 1)]
    print(f"⬇️  downloading {HF_REPO_ID} years {start.year}–{end_year} → {root} …")
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset", local_dir=str(root), allow_patterns=allow)


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.date


def _read_hive(folder: Path, columns: list[str], tickers: list[str] | None, start: date, end: date | None) -> pd.DataFrame:
    dataset = ds.dataset(str(folder), format="parquet", partitioning="hive")
    filt = ds.field("year") >= start.year
    if end is not None:
        filt = filt & (ds.field("year") <= end.year)
    if tickers is not None:
        filt = filt & ds.field("symbol").isin(tickers)
    df = dataset.to_table(columns=columns, filter=filt).to_pandas()
    if df.empty:
        return df
    df["date"] = _to_date(df["date"])
    df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    return df


def _read_filings(folder: Path, tickers: list[str] | None, start: date, end: date | None) -> pd.DataFrame:
    frames = []
    for f in sorted(folder.glob("year=*/part-000.parquet")):
        year = int(f.parent.name.split("=")[-1])
        if year < start.year or (end is not None and year > end.year):
            continue
        df = pq.ParquetFile(f).read().to_pandas()
        if df.empty:
            continue
        df["date"] = _to_date(df["date"])
        df = df[df["date"] >= start]
        if end is not None:
            df = df[df["date"] <= end]
        if tickers is not None:
            df = df[df["symbol"].isin(tickers)]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol", "filing_idx"])


def build_dataset(root: Path, tickers: list[str] | None, start: date, end: date | None) -> dict:
    data: dict = defaultdict(lambda: defaultdict(dict))

    price_cols = ["date", "symbol", "cik", "open", "high", "low", "close", "adjusted_close", "volume"]
    price = _read_hive(root / "price_daily", price_cols, tickers, start, end)
    for r in price.itertuples(index=False):
        factor = (r.adjusted_close / r.close) if r.close else 1.0
        data[r.date]["price"][r.symbol] = {
            "cik": r.cik,
            "open": r.open, "high": r.high, "low": r.low, "close": r.close,
            "adjusted_open": r.open * factor, "adjusted_high": r.high * factor, "adjusted_low": r.low * factor,
            "adjusted_close": r.adjusted_close, "volume": r.volume,
        }

    if (root / "news_items").exists():
        news_cols = ["date", "symbol", "item_index", "news_text"]
        news = _read_hive(root / "news_items", news_cols, tickers, start, end)
        if not news.empty:
            news = news.sort_values(["date", "symbol", "item_index"])
            for r in news.itertuples(index=False):
                data[r.date]["news"].setdefault(r.symbol, []).append(r.news_text)

    for modality, folder in (("filing_k", "filingk"), ("filing_q", "filingq")):
        if not (root / folder).exists():
            continue
        filings = _read_filings(root / folder, tickers, start, end)
        for r in filings.itertuples(index=False):
            cur = data[r.date][modality].get(r.symbol)
            data[r.date][modality][r.symbol] = f"{cur}\n\n{r.filing_text}" if cur else r.filing_text

    return {d: dict(v) for d, v in data.items()}


def coverage_report(data: dict, tickers: list[str]) -> None:
    counts: dict[tuple[str, str], int] = {(m, t): 0 for m in MODALITIES for t in tickers}
    sample = None
    for day in data.values():
        for m in MODALITIES:
            for t, payload in day.get(m, {}).items():
                if (m, t) in counts and payload:
                    counts[(m, t)] += 1
                    if sample is None and m in ("filing_k", "filing_q"):
                        sample = (m, t, payload)
    print("\n📊 Coverage — non-empty days per (modality, ticker):")
    print("  modality   " + "".join(f"{t:>9}" for t in tickers))
    for m in MODALITIES:
        print(f"  {m:<10}" + "".join(f"{counts[(m, t)]:>9}" for t in tickers))
    if sample:
        m, t, payload = sample
        print(f"\n🔬 Filing sample — {m}/{t}: type={type(payload).__name__}, "
              f"len={len(payload) if isinstance(payload, str) else 'n/a'}")
        if isinstance(payload, str):
            print(f"   head: {payload[:300]!r}")
            print("   → RAW TEXT confirmed: add a one-time LLM-summary cache before the Fundamentals analyst.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(PROJECT_ROOT / "FINSABER-V2-Data"), help="FINSABER-V2 parquet root.")
    ap.add_argument("--download", action="store_true", help="Fetch the HF snapshot into --root if missing.")
    ap.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    ap.add_argument("--start", default="2020-01-01", help="Inclusive start date (YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="Inclusive end date (YYYY-MM-DD); default = all available.")
    ap.add_argument("--output", default=str(PROJECT_ROOT / "stock_data_selected5_2020_2024_finsaberv2.pkl"))
    args = ap.parse_args()

    root = Path(args.root)
    tickers = [t.upper() for t in args.tickers]
    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date() if args.end else None

    if args.download:
        _maybe_download(root, start, end)
    if not (root / "price_daily").exists():
        raise SystemExit(f"❌ parquet root not found at {root}. Pass --download or --root <path>.")

    print(f"📥 Reading {tickers} from {root}  (>= {start}{f', <= {end}' if end else ''}) …")
    data = build_dataset(root, tickers, start, end)
    if not data:
        raise SystemExit("❌ no rows matched — check ticker spellings and date range.")
    print(f"   assembled {len(data)} trading days")
    coverage_report(data, tickers)

    out = Path(args.output)
    with out.open("wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n💾 wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print("   set DATA_PKL=<this path> to use it (or drop it under data/ as the default).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
