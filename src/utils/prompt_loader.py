from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml


_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_TEMPLATE_CACHE: Dict[str, str] = {}
_DATA_CACHE: Dict[str, dict] = {}


def _load_yaml(name: str) -> dict:
    if name in _DATA_CACHE:
        return _DATA_CACHE[name]
    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"prompt YAML not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    _DATA_CACHE[name] = data
    return data


def load_prompt(name: str) -> str:
    if name in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[name]
    data = _load_yaml(name)
    template = data.get("user")
    if not template or not isinstance(template, str):
        raise RuntimeError(
            f"{name}.yaml: required top-level 'user:' field missing or not a string"
        )
    _TEMPLATE_CACHE[name] = template
    return template


def render_messages(name: str, **kwargs) -> list:
    messages = []
    system = _load_yaml(name).get("system") or ""
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.format(**kwargs).strip()})
    messages.append({"role": "user", "content": load_prompt(name).format(**kwargs)})
    return messages


def render_prompt(name: str, **kwargs) -> str:
    return load_prompt(name).format(**kwargs)


