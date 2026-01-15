# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Faiss не импортируется. Проверь, что установлен faiss-cpu (или faiss-gpu)."
    ) from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError(
        "sentence-transformers не импортируется. Проверь requirements.txt."
    ) from e

from src.fetch import fetch_last_posts
from src.io_utils import ensure_dir, write_jsonl
from src.text_utils import clean_text


log = logging.getLogger("index")


@dataclass(frozen=True)
class Doc:
    doc_id: str          # telegram message id as string (stable)
    date_iso: str        # ISO string
    url: str
    text: str            # normalized text


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:

    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def build_docs(raw_posts: List[dict], min_len: int) -> List[Doc]:
    docs: List[Doc] = []
    seen_texts: set[str] = set()

    for p in raw_posts:
        text_raw = (p.get("text") or "").strip()
        text = clean_text(text_raw)

        if len(text) < min_len:
            continue

        # дедупликация на уровне документа (по нормализованному тексту)
        if text in seen_texts:
            continue
        seen_texts.add(text)

        doc_id = str(p.get("id"))
        date_iso = str(p.get("date") or "")
        url = str(p.get("url") or "")

        docs.append(Doc(doc_id=doc_id, date_iso=date_iso, url=url, text=text))

    return docs


def encode_docs(
    model_name: str,
    device: str,
    batch_size: int,
    texts: List[str],
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)

    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    emb = emb.astype(np.float32, copy=False)
    emb = l2_normalize(emb)
    return emb


def write_metadata(path: Path, docs: List[Doc]) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": len(docs),
        "docs": [
            {"doc_id": d.doc_id, "date": d.date_iso, "url": d.url, "text": d.text}
            for d in docs
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_faiss_index(emb: np.ndarray) -> "faiss.Index":
    if emb.ndim != 2:
        raise ValueError("embeddings должны иметь форму (n_docs, dim)")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  
    index.add(emb)
    return index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Индексация последних постов Telegram-канала в FAISS (cosine similarity).",
    )

    p.add_argument("--channel", default="markettwits", help="Telegram channel username (без @).")
    p.add_argument("--limit", type=int, default=30, help="Сколько последних постов брать.")
    p.add_argument(
        "--fetch-method",
        choices=["scrape", "telethon"],
        default="scrape",
        help="Способ загрузки: scrape (публичная страница) или telethon (API).",
    )
    p.add_argument(
        "--session",
        default="data/telethon.session",
        help="Путь к файлу сессии Telethon (используется только при --fetch-method telethon).",
    )

    p.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers модель.")
    p.add_argument("--device", default="cpu", help="cpu / cuda / cuda:0 и т.п.")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size для эмбеддингов.")

    p.add_argument("--data-dir", default="data", help="Каталог для данных.")
    p.add_argument("--raw-path", default="data/raw.jsonl", help="Куда писать сырые посты (jsonl).")
    p.add_argument("--docs-path", default="data/docs.jsonl", help="Куда писать документы (jsonl).")

    p.add_argument("--index-path", default="data/index.faiss", help="Куда сохранять FAISS индекс.")
    p.add_argument("--meta-path", default="data/meta.json", help="Куда сохранять метаданные.")
    p.add_argument("--emb-path", default="data/embeddings.npy", help="Куда сохранять матрицу эмбеддингов.")

    p.add_argument("--min-doc-len", type=int, default=20, help="Минимальная длина документа (после чистки).")
    p.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(name)s | %(message)s",
    )

    data_dir = Path(args.data_dir)
    ensure_dir(data_dir)

    raw_path = Path(args.raw_path)
    docs_path = Path(args.docs_path)
    index_path = Path(args.index_path)
    meta_path = Path(args.meta_path)
    emb_path = Path(args.emb_path)

    for pth in [raw_path, docs_path, index_path, meta_path, emb_path]:
        ensure_dir(pth.parent)

    log.info("Fetching posts: channel=%s limit=%d method=%s", args.channel, args.limit, args.fetch_method)
    raw_posts = fetch_last_posts(
        channel=args.channel,
        limit=args.limit,
        method=args.fetch_method,
        session_path=Path(args.session),
    )

    if not raw_posts:
        log.error("Не удалось получить посты (0 штук).")
        return 2

    fetched_at = datetime.now(timezone.utc).isoformat()
    raw_dump = []
    for p in raw_posts:
        raw_dump.append({
            "source": "telegram",
            "channel": args.channel,
            "fetched_at": fetched_at,
            **p,
        })

    write_jsonl(raw_path, raw_dump)
    log.info("Raw saved: %s (%d lines)", raw_path.as_posix(), len(raw_dump))

    docs = build_docs(raw_posts, min_len=args.min_doc_len)
    if not docs:
        log.error("После чистки/фильтрации не осталось документов. Попробуй уменьшить --min-doc-len.")
        return 3

    write_jsonl(
        docs_path,
        [
            {"doc_id": d.doc_id, "date": d.date_iso, "url": d.url, "text": d.text}
            for d in docs
        ],
    )
    log.info("Docs saved: %s (%d lines)", docs_path.as_posix(), len(docs))

    log.info("Encoding %d docs with model=%s device=%s batch=%d", len(docs), args.model, args.device, args.batch_size)
    emb = encode_docs(args.model, args.device, args.batch_size, [d.text for d in docs])
    np.save(emb_path, emb)
    log.info("Embeddings saved: %s shape=%s", emb_path.as_posix(), tuple(emb.shape))

    log.info("Building FAISS index (IndexFlatIP) ...")
    index = build_faiss_index(emb)
    faiss.write_index(index, str(index_path))
    log.info("Index saved: %s", index_path.as_posix())

    write_metadata(meta_path, docs)
    log.info("Metadata saved: %s", meta_path.as_posix())

    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
