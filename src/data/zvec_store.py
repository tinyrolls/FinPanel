from __future__ import annotations

import hashlib
import os
import struct
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

ADA_MODEL = "text-embedding-ada-002"
ADA_DIM = 1536
from data.reputation_memory import LOOK_AHEAD_GAP_DAYS, parse_date as _parse_date


ADA_MAX_TOKENS = 8_000
_ADA_ENC = None


def _ada_truncate(text: str) -> str:
    global _ADA_ENC
    if _ADA_ENC is None:
        import tiktoken
        _ADA_ENC = tiktoken.get_encoding("cl100k_base")
    ids = _ADA_ENC.encode(text, disallowed_special=())
    return text if len(ids) <= ADA_MAX_TOKENS else _ADA_ENC.decode(ids[:ADA_MAX_TOKENS])


def _ada_embed(text: str) -> List[float]:
    import litellm

    resp = litellm.embedding(model=ADA_MODEL, input=[_ada_truncate(text or " ")])
    return list(resp.data[0]["embedding"])


def _embed_cache_file(key: str) -> str:
    root = os.environ.get("FINPANEL_EMBED_CACHE_DIR", "data/embed_cache")
    return os.path.join(root, ADA_MODEL, key[:2], key + ".f32")


def _cached_ada_embed(text: str) -> List[float]:
    key = hashlib.sha256(f"{ADA_MODEL}|{text or ' '}".encode("utf-8")).hexdigest()
    fp = _embed_cache_file(key)
    try:
        with open(fp, "rb") as fh:
            b = fh.read()
        if len(b) == ADA_DIM * 4:
            return list(struct.unpack(f"<{ADA_DIM}f", b))
    except FileNotFoundError:
        pass
    vec = _ada_embed(text)
    try:
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        tmp = f"{fp}.tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            fh.write(struct.pack(f"<{len(vec)}f", *vec))
        os.replace(tmp, fp)
    except OSError:
        pass
    return vec


def _sql_quote(s: str) -> str:
    return str(s).replace("'", "''")


class ZvecStore:

    RESERVED = ("t_write", "text")

    def __init__(self, path: str | Path, scalar_fields: Optional[List[str]] = None, dim: int = ADA_DIM, embed_fn: Optional[Callable[[str], List[float]]] = None):
        import zvec
        from zvec import CollectionSchema, DataType, FieldSchema, FlatIndexParam, MetricType, VectorSchema

        self._zvec = zvec
        self.path = str(path)
        self.dim = dim
        self._embed = embed_fn or _cached_ada_embed
        fields = list(dict.fromkeys(list(scalar_fields or []) + list(self.RESERVED)))
        self.scalar_fields = fields
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        schema = CollectionSchema(
            name="mem",
            fields=[FieldSchema(f, DataType.STRING, nullable=True) for f in fields],
            vectors=VectorSchema("emb", DataType.VECTOR_FP32, dim, index_param=FlatIndexParam(metric_type=MetricType.COSINE)),
        )
        if Path(self.path).exists():
            self.col = zvec.open(path=self.path)
        else:
            self.col = zvec.create_and_open(path=self.path, schema=schema)

    def add(self, doc_id: str, text: str, fields: Dict[str, str]) -> None:
        if not (text or "").strip():
            return
        from zvec import Doc

        emb = self._embed(text)
        payload = {k: ("" if v is None else str(v)) for k, v in fields.items() if k in self.scalar_fields}
        payload["text"] = str(text)
        self.col.insert([Doc(id=str(doc_id), vectors={"emb": [float(x) for x in emb]}, fields=payload)])

    def query(self, query_text: str, t_now: str, k: int = 3, extra_filter: Optional[str] = None, output_fields: Optional[List[str]] = None) -> List[dict]:
        from zvec import Query

        now = _parse_date(t_now)
        if now is None:
            return []
        cutoff = (now - timedelta(days=LOOK_AHEAD_GAP_DAYS)).isoformat()
        clauses = [f"t_write <= '{_sql_quote(cutoff)}'"]
        if extra_filter:
            clauses.append(f"({extra_filter})")
        filt = " and ".join(clauses)
        of = list(dict.fromkeys((output_fields or []) + list(self.scalar_fields)))
        try:
            qv = self._embed(query_text)
            res = self.col.query(Query("emb", vector=[float(x) for x in qv]), topk=int(k), filter=filt, output_fields=of)
        except Exception:
            return []
        out: List[dict] = []
        for d in res:
            f = dict(d.fields or {})
            out.append({"id": d.id, "sim": 1.0 - float(d.score), "text": f.get("text", ""), "fields": f})
        return out
