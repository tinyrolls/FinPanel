from __future__ import annotations

import os
import re
from datetime import datetime

_ENV_VAR = "FINPANEL_RUN_NAME"
_run_name: str | None = None


def _slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", str(s).strip()).strip("-")


def init_run_name(note: str = "", prefix: str = "") -> str:
    global _run_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    parts = [_slug(p) for p in (prefix, stamp, note) if p and _slug(p)]
    _run_name = "_".join(parts)
    os.environ[_ENV_VAR] = _run_name
    return _run_name


def get_run_name() -> str:
    global _run_name
    if _run_name is None:
        _run_name = os.environ.get(_ENV_VAR) or datetime.now().strftime("%Y%m%d_%H%M")
        os.environ[_ENV_VAR] = _run_name
    return _run_name
