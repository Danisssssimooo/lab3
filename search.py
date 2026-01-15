#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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

from src.text_utils import clean_text, snippet


log = logging.getLogger("search")


@dataclass(frozen=True)
class MetaDoc:
    doc_id: str
    date: str
    url: str
    text: str


def l2_normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def load_meta(meta_path: Path) -> List[MetaDoc]:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    docs = data.get("docs", [])
    out: List[MetaDoc] = []
    for d in docs:
        out.append(
            MetaDoc(
                doc_id=str(d.get("doc_id")),
                date=str(d.get("date") or ""),
                url=str(d.get("url") or ""),
                text=str(d.get("text") or ""),
            )
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Поиск по FAISS-индексу (semantic search по постам Telegram).",
    )
    p.add_argument("query", help="Текстовый запрос (в кавычках, если с пробелами).")

    p.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers модель.")
    p.add_argument("--device", default="cpu", help="cpu / cuda / cuda:0 и т.п.")

    p.add_argument("--index-path", default="data/index.faiss", help="Путь к файлу FAISS индекса.")
    p.add_argument("--meta-path", default="data/meta.json", help="Путь к метаданным (meta.json).")
    p.add_argument("--emb-path", default="data/embeddings.npy", help="Путь к embeddings.npy (для постфильтрации).")

    p.add_argument("--top-k", type=int, default=5, help="Сколько результатов вернуть (после постфильтрации).")
    p.add_argument("--candidates-mult", type=int, default=6, help="Во сколько раз больше кандидатов брать из индекса.")
    p.add_argument("--min-doc-len", type=int, default=20, help="Фильтр по длине документа (после чистки).")

    p.add_argument(
        "--dedupe-threshold",
        type=float,
        default=0.92,
        help="Порог near-duplicate по cosine между документами в выдаче (0..1).",
    )

    p.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    return p.parse_args()


def postfilter(
    cand_ids: List[int],
    cand_scores: List[float],
    docs: List[MetaDoc],
    doc_emb: np.ndarray,
    top_k: int,
    min_doc_len: int,
    dedupe_threshold: float,
) -> List[Tuple[int, float]]:
    kept: List[Tuple[int, float]] = []
    kept_ids: List[int] = []

    for idx, score in zip(cand_ids, cand_scores):
        if idx < 0 or idx >= len(docs):
            continue

        text = docs[idx].text
        if len(text) < min_doc_len:
            continue

        # near-duplicates: сравниваем вектор кандидата с уже оставленными
        v = doc_emb[idx]
        is_dup = False
        for kid in kept_ids:
            sim = float(np.dot(v, doc_emb[kid]))  
            if sim >= dedupe_threshold:
                is_dup = True
                break
        if is_dup:
            continue

        kept.append((idx, float(score)))
        kept_ids.append(idx)

        if len(kept) >= top_k:
            break

    return kept


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(name)s | %(message)s",
    )

    index_path = Path(args.index_path)
    meta_path = Path(args.meta_path)
    emb_path = Path(args.emb_path)

    for p in [index_path, meta_path, emb_path]:
        if not p.exists():
            log.error("Файл не найден: %s. Сначала запусти index.py", p.as_posix())
            return 2

    docs = load_meta(meta_path)
    if not docs:
        log.error("В meta.json нет документов.")
        return 3

    doc_emb = np.load(emb_path)
    if doc_emb.shape[0] != len(docs):
        log.error("embeddings.npy не совпадает с meta.json по числу документов.")
        return 4

    index = faiss.read_index(str(index_path))

    query = clean_text(args.query)
    if not query:
        log.error("Пустой запрос после чистки.")
        return 5

    model = SentenceTransformer(args.model, device=args.device)
    q = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    q[0] = l2_normalize_vec(q[0])
    q = q.reshape(1, -1)

    k = max(args.top_k * max(1, args.candidates_mult), args.top_k)
    scores, ids = index.search(q, k)

    cand_ids = [int(x) for x in ids[0].tolist() if int(x) >= 0]
    cand_scores = [float(x) for x in scores[0].tolist()][: len(cand_ids)]

    filtered = postfilter(
        cand_ids=cand_ids,
        cand_scores=cand_scores,
        docs=docs,
        doc_emb=doc_emb,
        top_k=args.top_k,
        min_doc_len=args.min_doc_len,
        dedupe_threshold=float(args.dedupe_threshold),
    )

    print()
    print(f'Query: "{query}"')
    print(f"Found: {len(filtered)} / requested top_k={args.top_k}")
    print("-" * 80)

    for rank, (idx, score) in enumerate(filtered, start=1):
        d = docs[idx]
        text_line = snippet(d.text, 180)
        print(f"{rank:>2}. score={score: .4f} | date={d.date}")
        print(f"    {text_line}")
        print(f"    {d.url}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
