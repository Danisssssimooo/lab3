# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

import requests
from bs4 import BeautifulSoup


FetchMethod = Literal["scrape", "telethon"]


def fetch_last_posts(
    channel: str,
    limit: int,
    method: FetchMethod,
    session_path: Path,
) -> List[dict]:
    """
    Возвращает список постов в формате:
    {
      "id": int,
      "date": "2026-01-16T00:00:00+00:00",
      "url": "https://t.me/markettwits/123",
      "text": "..."
    }
    """
    channel = channel.lstrip("@").strip()

    if method == "telethon":
        return _fetch_telethon(channel, limit, session_path=session_path)

    return _fetch_scrape(channel, limit)


# ------------------------- scrape (public page) -------------------------

_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


def _fetch_scrape(channel: str, limit: int) -> List[dict]:

    url = f"https://t.me/s/{channel}"
    html = _get_with_retries(url, tries=4, timeout=15)

    soup = BeautifulSoup(html, "lxml")

    items = []
    for msg in soup.select("div.tgme_widget_message_wrap"):
        # ссылка на пост
        a = msg.select_one("a.tgme_widget_message_date")
        if not a or not a.get("href"):
            continue
        post_url = a["href"].strip()

        m = re.search(r"/(\d+)$", post_url)
        if not m:
            continue
        post_id = int(m.group(1))

        # дата
        time_tag = msg.select_one("time.time")
        dt_iso = ""
        if time_tag and time_tag.get("datetime"):
            # telegram отдаёт ISO-подобную дату
            dt_iso = time_tag["datetime"].strip()
        else:
            dt_iso = datetime.now(timezone.utc).isoformat()

        # текст
        text_div = msg.select_one("div.tgme_widget_message_text")
        text = ""
        if text_div:
            # get_text с separator сохраняет переносы
            text = text_div.get_text(separator="\n", strip=True)

        items.append({"id": post_id, "date": dt_iso, "url": post_url, "text": text})


    items.sort(key=lambda x: x["id"], reverse=True)
    return items[:limit]


def _get_with_retries(url: str, tries: int, timeout: int) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, tries + 1):
        try:
            headers = {"User-Agent": random.choice(_UA_POOL)}
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            # простой backoff
            sleep_s = min(2 ** attempt, 10) + random.random()
            time.sleep(sleep_s)

    raise RuntimeError(f"Не удалось скачать страницу после {tries} попыток: {url}") from last_err


# Telegram API

def _fetch_telethon(channel: str, limit: int, session_path: Path) -> List[dict]:
    """
    Требует переменные окружения:
      TG_API_ID  (int)
      TG_API_HASH (str)

    При первом запуске Telethon попросит телефон и код. Сессия сохранится в session_path.
    """
    api_id = os.getenv("TG_API_ID")
    api_hash = os.getenv("TG_API_HASH")

    if not api_id or not api_hash:
        raise RuntimeError(
            "Для telethon нужны TG_API_ID и TG_API_HASH в окружении. "
            "Смотри README.md (секреты не хардкодим)."
        )

    try:
        from telethon import TelegramClient  # type: ignore
    except Exception as e:
        raise RuntimeError("Telethon не установлен. Поставь telethon из requirements.txt.") from e

    session_path.parent.mkdir(parents=True, exist_ok=True)
    client = TelegramClient(str(session_path), int(api_id), api_hash)

    posts: List[dict] = []

    async def _run() -> None:
        await client.start()
        entity = await client.get_entity(channel)

        async for msg in client.iter_messages(entity, limit=limit):
            text = (msg.message or "").strip()
            if not text:
                continue

            dt = msg.date
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            posts.append(
                {
                    "id": int(msg.id),
                    "date": dt.isoformat(),
                    "url": f"https://t.me/{channel}/{msg.id}",
                    "text": text,
                }
            )

        await client.disconnect()

    client.loop.run_until_complete(_run())

    posts.sort(key=lambda x: x["id"], reverse=True)
    return posts[:limit]
