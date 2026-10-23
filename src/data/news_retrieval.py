from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

EMBED_MODEL = "text-embedding-ada-002"
DEFAULT_K = 5
DEFAULT_LOOKBACK_DAYS = 7
QUERY_TEMPLATE = (
    "Material, market-moving news about {ticker}: earnings, revenue, guidance, "
    "products, regulation, partnerships, analyst ratings, litigation, leadership."
)


class _LiteLLMEmbedder:

    def embed_documents(self, texts) -> list:
        import litellm

        resp = litellm.embedding(model=EMBED_MODEL, input=list(texts))
        return [d["embedding"] for d in resp.data]

    def embed_query(self, text: str) -> list:
        import litellm

        resp = litellm.embedding(model=EMBED_MODEL, input=[text])
        return resp.data[0]["embedding"]


def _get_embedder():
    return _LiteLLMEmbedder()


def _news_by_date(data: dict, ticker: str) -> dict[dt.date, list[str]]:
    out: dict[dt.date, list[str]] = {}
    for day, day_data in data.items():
        d = day if isinstance(day, dt.date) else dt.date.fromisoformat(str(day))
        arts = (day_data.get("news", {}) or {}).get(ticker)
        if arts:
            out[d] = [str(a).strip() for a in arts if str(a).strip()]
    return out


def _price_days(data: dict, ticker: str) -> list[dt.date]:
    return sorted(
        (day if isinstance(day, dt.date) else dt.date.fromisoformat(str(day)))
        for day, dd in data.items() if ticker in (dd.get("price", {}) or {})
    )


def _window(by_date: dict[dt.date, list[str]], as_of: dt.date, lookback: int) -> list[str]:
    lo = as_of - dt.timedelta(days=lookback)
    seen: set[str] = set()
    out: list[str] = []
    for d in sorted((x for x in by_date if lo < x <= as_of), reverse=True):
        for h in by_date[d]:
            if h not in seen:
                seen.add(h)
                out.append(h)
    return out


def _cosine_topk(query_vec: np.ndarray, mat: np.ndarray, k: int) -> list[int]:
    if mat.shape[0] == 0:
        return []
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = mn @ qn
    return list(np.argsort(-sims)[: min(k, mat.shape[0])])


def _in_range(d: dt.date, start: dt.date | None, end: dt.date | None) -> bool:
    return (start is None or d >= start) and (end is None or d <= end)


class EmbeddingStore:

    def __init__(self, by_ticker: dict[str, dict[str, np.ndarray]]):
        self._by = by_ticker

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingStore":
        import pandas as pd
        by: dict[str, dict[str, np.ndarray]] = {}
        p = Path(path)
        if p.exists():
            df = pd.read_parquet(p)
            if "model" in df.columns:
                models = set(df["model"].unique())
                if models - {EMBED_MODEL}:
                    raise ValueError(
                        f"{p} holds vectors from {models}, but EMBED_MODEL={EMBED_MODEL!r}. "
                        f"Different embedding models are NOT comparable — re-embed into a fresh file."
                    )
            for ticker, headline, emb in zip(df["ticker"], df["headline"], df["embedding"]):
                by.setdefault(ticker, {})[headline] = np.asarray(emb, dtype=np.float32)
        return cls(by)

    def has(self, ticker: str, h: str) -> bool:
        return h in self._by.get(ticker, {})

    def get(self, ticker: str, h: str) -> np.ndarray | None:
        return self._by.get(ticker, {}).get(h)

    def add(self, ticker: str, new: dict[str, np.ndarray]) -> None:
        self._by.setdefault(ticker, {}).update(new)

    def save(self, path: str | Path) -> None:
        import pandas as pd
        rows = [(t, h, v.astype(np.float32), EMBED_MODEL) for t, hv in self._by.items() for h, v in hv.items()]
        df = pd.DataFrame(rows, columns=["ticker", "headline", "embedding", "model"])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression="zstd")


def ensure_embeddings(
    data: dict | str | Path,
    emb_path: str | Path,
    *,
    tickers: list[str] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    batch: int = 512,
    verbose: bool = True,
) -> int:
    if isinstance(data, (str, Path)):
        import pickle
        with Path(data).open("rb") as fh:
            data = pickle.load(fh)

    store = EmbeddingStore.load(emb_path)
    all_tickers = tickers or sorted({t for dd in data.values() for t in (dd.get("price", {}) or {})})
    embedder = None
    total = 0
    for ticker in all_tickers:
        by_date = _news_by_date(data, ticker)
        distinct = sorted({h for d, hs in by_date.items() if _in_range(d, start, end) for h in hs})
        missing = [h for h in distinct if not store.has(ticker, h)]
        if not missing:
            if verbose:
                print(f"✓ {ticker}: embeddings up to date ({len(distinct)} headlines in range)")
            continue
        if embedder is None:
            embedder = _get_embedder()
        if verbose:
            print(f"📐 {ticker}: embedding {len(missing)} new headlines ({len(distinct) - len(missing)} cached) …")
        new: dict[str, np.ndarray] = {}
        for i in range(0, len(missing), batch):
            chunk = missing[i : i + batch]
            vecs = embedder.embed_documents(chunk)
            for h, v in zip(chunk, vecs):
                new[h] = np.asarray(v, dtype=np.float32)
        store.add(ticker, new)
        store.save(emb_path)
        total += len(new)
        if verbose:
            print(f"   💾 {ticker}: froze {len(new)} new vectors → {emb_path}")
    if verbose:
        print(f"✅ embeddings done — {total} new vectors → {emb_path}")
    return total


def ensure_news_selection(
    data: dict | str | Path,
    sel_path: str | Path,
    emb_path: str | Path,
    *,
    tickers: list[str] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    k: int = DEFAULT_K,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    query_template: str = QUERY_TEMPLATE,
    verbose: bool = True,
) -> int:
    if isinstance(data, (str, Path)):
        import pickle
        with Path(data).open("rb") as fh:
            data = pickle.load(fh)

    sel_path = Path(sel_path)
    done: set[str] = set()
    if sel_path.exists():
        with sel_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done.add(f"{r['ticker']}|{r['date']}")

    store = EmbeddingStore.load(emb_path)
    all_tickers = tickers or sorted({t for dd in data.values() for t in (dd.get("price", {}) or {})})
    embedder = None
    n_written = 0
    with sel_path.open("a", encoding="utf-8") as fh:
        for ticker in all_tickers:
            by_date = _news_by_date(data, ticker)
            q_vec = None
            for as_of in _price_days(data, ticker):
                if not _in_range(as_of, start, end):
                    continue
                key = f"{ticker}|{as_of.isoformat()}"
                if key in done:
                    continue
                window = _window(by_date, as_of, lookback_days)
                vecs = [(h, store.get(ticker, h)) for h in window]
                vecs = [(h, v) for h, v in vecs if v is not None]
                if vecs:
                    if q_vec is None:
                        if embedder is None:
                            embedder = _get_embedder()
                        q_vec = np.asarray(embedder.embed_query(query_template.format(ticker=ticker)), np.float32)
                    mat = np.vstack([v for _, v in vecs])
                    selected = [vecs[i][0] for i in _cosine_topk(q_vec, mat, k)]
                else:
                    selected = []
                fh.write(json.dumps({"ticker": ticker, "date": as_of.isoformat(), "selected": selected}, ensure_ascii=False) + "\n")
                fh.flush()
                n_written += 1
    if verbose:
        print(f"💾 wrote {n_written} news selections → {sel_path} (k={k}, lookback={lookback_days}d)")
    return n_written


class NewsSelectionStore:

    def __init__(self, by_key: dict[str, list[str]]):
        self._by_key = by_key

    @classmethod
    def load(cls, path: str | Path) -> "NewsSelectionStore":
        by_key: dict[str, list[str]] = {}
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        by_key[f"{r['ticker']}|{r['date']}"] = r.get("selected", [])
        return cls(by_key)

    def get(self, ticker: str, date: str) -> list[str] | None:
        return self._by_key.get(f"{ticker}|{date}")
