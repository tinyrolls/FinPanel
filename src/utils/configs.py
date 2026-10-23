from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


REGIMES: List[Dict[str, str]] = [
    {"period_id": "OOK_2025_H1", "label": "2025h1", "display": "2025 H1", "start_date": "2025-01-02", "end_date": "2025-06-30"},
    {"period_id": "OOK_2025", "label": "2025", "display": "2025 (full year)", "start_date": "2025-01-02", "end_date": "2025-12-31"},
]


def _load_yaml(path: Path) -> Dict[str, any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_config(overrides: Dict[str, any] | None = None) -> Dict[str, any]:
    base = _load_yaml(_CONFIGS_DIR / "base.yaml")
    return {**base, **(overrides or {})}


