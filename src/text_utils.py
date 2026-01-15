# -*- coding: utf-8 -*-

from __future__ import annotations

import re


_WS = re.compile(r"[ \t\f\v]+")
_MULTINEW = re.compile(r"\n{3,}")
_TRAILING_LINK_LINE = re.compile(r"(?im)^\s*(https?://\S+|t\.me/\S+)\s*$")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\uFEFF]")


def clean_text(text: str) -> str:
    """
    Минимальная чистка для коротких постов:
    - убираем zero-width мусор
    - нормализуем пробелы
    - режем "хвосты" из отдельных строк-ссылок
    - приводим переносы
    """
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _ZERO_WIDTH.sub("", t)

    lines = []
    for line in t.split("\n"):
        if _TRAILING_LINK_LINE.match(line.strip()):
            continue
        lines.append(line)

    t = "\n".join(lines).strip()
    t = _WS.sub(" ", t)
    t = _MULTINEW.sub("\n\n", t)

    t = "\n".join([ln.strip() for ln in t.split("\n")]).strip()

    return t


def snippet(text: str, max_len: int = 160) -> str:
    t = " ".join(text.split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"
