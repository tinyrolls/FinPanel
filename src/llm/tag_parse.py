from __future__ import annotations

import re
from typing import Sequence


def extract_tag(text: str, name: str, default: str = "") -> str:
    if not text:
        return default
    closed = re.search(
        rf"<{re.escape(name)}\s*>(.*?)</\s*{re.escape(name)}\s*>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if closed:
        return closed.group(1).strip()
    open_only = re.search(
        rf"<{re.escape(name)}\s*>(.*)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if open_only:
        return open_only.group(1).strip()
    return default


def strip_tags(text: str, names: Sequence[str]) -> str:
    out = text or ""
    for name in names:
        out = re.sub(rf"<{re.escape(name)}\s*>.*?</\s*{re.escape(name)}\s*>", "", out, flags=re.IGNORECASE | re.DOTALL)
        out = re.sub(rf"</?\s*{re.escape(name)}\s*>", "", out, flags=re.IGNORECASE)
    return out.strip()


def extract_choice(text: str, choices: Sequence[str], default: str, tag_names: Sequence[str] = ("direction", "action")) -> str:
    for name in tag_names:
        tagged = extract_tag(text, name, default="")
        if tagged:
            picked = normalize_choice(tagged, choices, "")
            if picked:
                return picked
    return normalize_choice(text or "", choices, default)


def choice_validator(tag: str, choices: Sequence[str]):
    wanted = set(choices)

    def _validate(resp) -> bool:
        return normalize_choice(resp.tags.get(tag, ""), list(choices), "") in wanted

    return _validate


def normalize_choice(value: str, choices: Sequence[str], default: str) -> str:
    if value:
        up = value.upper()
        best_pos = -1
        best = default
        for choice in choices:
            for m in re.finditer(rf"\b{re.escape(str(choice).upper())}\b", up):
                if m.start() >= best_pos:
                    best_pos = m.start()
                    best = choice
        if best_pos >= 0:
            return best
    return default
