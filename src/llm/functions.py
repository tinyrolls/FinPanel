import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeVar, Optional, Any, Callable, Dict, List
from pydantic import BaseModel
from utils.progress import progress

T = TypeVar('T', bound=BaseModel)


@dataclass
class LLMResponse:

    content: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", (model_name or "unknown").strip()) or "unknown"


class _LLMCache:

    def __init__(self):
        self.enabled = bool(os.environ.get("FINPANEL_LLM_CACHE"))
        self.root = os.environ.get("FINPANEL_LLM_CACHE_DIR", "runtime/llm_cache")
        self.read_roots: List[str] = [p for p in (os.environ.get("FINPANEL_LLM_CACHE_READ_DIRS") or "").split(os.pathsep) if p.strip()]
        self.mem: Dict[str, Dict[str, tuple]] = {}
        self.inherited: Dict[str, Dict[str, tuple]] = {}
        self._loaded: set = set()
        self.n_inherited = 0
        if self.enabled:
            os.makedirs(self.root, exist_ok=True)
            if self.read_roots:
                import atexit
                atexit.register(lambda: self.n_inherited and print(f"  ♻️ LLM cache: inherited {self.n_inherited} response(s) into {self.root}"))

    @staticmethod
    def _read_shard(path: str) -> Dict[str, tuple]:
        out: Dict[str, tuple] = {}
        if not os.path.exists(path):
            return out
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    out[r["key"]] = (r["content"], r["tags"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return out

    def _path(self, model_slug: str) -> str:
        return os.path.join(self.root, model_slug, "llm_cache.jsonl")

    def _ensure_loaded(self, model_slug: str) -> None:
        if model_slug in self._loaded:
            return
        self._loaded.add(model_slug)
        self.mem.setdefault(model_slug, {}).update(self._read_shard(self._path(model_slug)))
        if self.read_roots:
            inh = self.inherited.setdefault(model_slug, {})
            for r in reversed(self.read_roots):
                inh.update(self._read_shard(os.path.join(r, model_slug, "llm_cache.jsonl")))

    @staticmethod
    def key(model_name: str, prompt: Any, tags: Optional[List[str]]) -> str:
        ps = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False, sort_keys=True, default=str)
        raw = f"{model_name}|{ps}|{','.join(tags or [])}|{_SIM_DATE or ''}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, model_name: str, key: str):
        if not self.enabled:
            return None
        slug = _model_slug(model_name)
        self._ensure_loaded(slug)
        hit = self.mem.get(slug, {}).get(key)
        if hit is not None:
            return hit
        inh = self.inherited.get(slug, {}).get(key)
        if inh is not None:
            self.put(model_name, key, inh[0], inh[1])
            self.n_inherited += 1
            return inh
        return None

    def put(self, model_name: str, key: str, content: str, tags: Dict[str, str]) -> None:
        if not self.enabled:
            return
        slug = _model_slug(model_name)
        self._ensure_loaded(slug)
        self.mem.setdefault(slug, {})[key] = (content, tags)
        p = self._path(slug)
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "content": content, "tags": tags}, ensure_ascii=False) + "\n")
        except OSError:
            pass


_SIM_DATE: Optional[str] = None


def set_sim_date(date_str: Optional[str]) -> None:
    global _SIM_DATE
    _SIM_DATE = str(date_str)[:10] if date_str else None


def get_sim_date() -> Optional[str]:
    return _SIM_DATE


_LLM_CACHE: Optional["_LLMCache"] = None


def _llm_cache() -> "_LLMCache":
    global _LLM_CACHE
    if _LLM_CACHE is None:
        _LLM_CACHE = _LLMCache()
    return _LLM_CACHE


def _log_prompt_hash(agent_name: Optional[str], prompt: Any, model_name: str) -> None:
    try:
        prompt_str = json.dumps(prompt, ensure_ascii=False, sort_keys=True) if not isinstance(prompt, str) else prompt
        prompt_hash = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()[:12]
        from utils.run_context import get_run_name
        log_dir = "logs/prompt_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"prompt_hashes_{get_run_name()}.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            record = {
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name or "unknown",
                "model_name": model_name,
                "prompt_hash": prompt_hash,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _log_llm_call(agent_name: Optional[str], prompt: Any, model_name: str, attempt: int, content: str, tags: Dict[str, str], ok: bool) -> None:
    try:
        prompt_str = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False, default=str)
        from utils.run_context import get_run_name
        log_dir = "logs/llm_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"llm_calls_{get_run_name()}.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name or "unknown",
                "model_name": model_name,
                "attempt": attempt,
                "ok": ok,
                "prompt": prompt_str,
                "response": content,
                "tags": tags,
            }, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    tags: Optional[List[str]] = None,
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    validate: Optional[Callable[["LLMResponse"], bool]] = None,
) -> LLMResponse:
    _log_prompt_hash(agent_name, prompt, model_name)

    _cache = _llm_cache()
    _ck = _cache.key(model_name, prompt, tags) if _cache.enabled else None
    if _ck is not None:
        _hit = _cache.get(model_name, _ck)
        if _hit is not None:
            return LLMResponse(content=_hit[0], tags=dict(_hit[1] or {}))

    from llm.providers import get_model
    from llm.tag_parse import extract_tag

    llm = get_model(model_name, model_provider)

    last_resp = LLMResponse(content="", tags={})
    for attempt in range(max_retries):
        try:
            result = llm.invoke(prompt, sim_date=_SIM_DATE)
            content = result.content if hasattr(result, "content") else str(result)
            if content and str(content).strip():
                content = str(content)
                extracted = {name: extract_tag(content, name, default="") for name in tags} if tags else {}
                resp = LLMResponse(content=content, tags=extracted)
                last_resp = resp
                ok = validate is None or bool(validate(resp))
                _log_llm_call(agent_name, prompt, model_name, attempt + 1, content, extracted, ok)
                if ok:
                    if _ck is not None:
                        _cache.put(model_name, _ck, content, extracted)
                    return resp
                raise ValueError("response failed validation")
            raise ValueError("empty response content")
        except Exception as e:
            print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}: {str(e)[:50]}")
            if attempt == max_retries - 1:
                if not last_resp.content:
                    _log_llm_call(agent_name, prompt, model_name, attempt + 1, "", {}, False)
                print(f"ERROR: {agent_name} free-form call exhausted {max_retries} attempts. Last error: {e}")
                return last_resp

    return last_resp


