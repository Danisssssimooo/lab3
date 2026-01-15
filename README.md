# lab3
Cемантическая поисковая система 
скачиваем последние посты из Telegram-канала, строим эмбеддинги, индексируем в FAISS и ищем по текстовому запросу.

## Структура

index.py - загрузка/чистка/эмбеддинги/построение и сохранение индекса
search.py - поиск по индексу + постфильтрация near-duplicates
srcfetch.py - получение постов (scrape или telethon)
srctext_utils.py - простая нормализация текста
data артефакты raw.jsonl, docs.jsonl, index.faiss, meta.json, embeddings.npy
