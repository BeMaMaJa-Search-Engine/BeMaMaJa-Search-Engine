from __future__ import annotations

import re
import time
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable

from src.utils import read_json, short_snippet, write_json

# Unicode-aware tokenizer: matches runs of letters/digits from any script (ä, ö, ü, ß)
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

# Precompiled patterns for Tübingen spelling normalization
TUEBINGEN_UMLAUT_RE = re.compile(r"tübingen", re.IGNORECASE)
TUEBINGEN_ASCII_RE = re.compile(r"\btuebingen\b", re.IGNORECASE)
TUEBINGEN_UBINGEN_RE = re.compile(r"\btubingen\b", re.IGNORECASE)

SUMMARY_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "preprocessor_summary.json"

# Below this many pages we just run sequentially.
MP_PAGE_THRESHOLD = 100

ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
}

GERMAN_STOPWORDS = {
    "aber", "alle", "als", "also", "am", "an", "auch", "auf", "aus", "bei",
    "bin", "bis", "bist", "da", "damit", "dann", "das", "dass", "dein", "deine",
    "dem", "den", "der", "des", "dich", "die", "dies", "diese", "dieser", "dieses",
    "doch", "dort", "du", "durch", "ein", "eine", "einen", "einer", "eines", "er",
    "es", "euch", "eure", "für", "hatte", "hatten", "hier", "ich", "ihm", "ihn",
}

FALLBACK_STOPWORDS = ENGLISH_STOPWORDS | GERMAN_STOPWORDS


def _load_stopwords() -> set[str]:
    # Try to load NLTK's English stopword list, falling back to a hardcoded set if unavailable.
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english")) | FALLBACK_STOPWORDS
    except Exception as exc:
        print(f"NLTK stopwords unavailable ({exc!r}); using small hardcoded list of {len(FALLBACK_STOPWORDS)} words")
        return FALLBACK_STOPWORDS


def _load_stemmer():
    # Try to load NLTK's Porter stemmer, returning None if it can't be loaded.
    try:
        from nltk.stem import PorterStemmer

        return PorterStemmer()
    except Exception as exc:
        print(f"NLTK PorterStemmer unavailable ({exc!r}); stemming will be skipped")
        return None


STOPWORDS = _load_stopwords()
STEMMER = _load_stemmer()


@lru_cache(maxsize=5000)
def _stem_cached(token: str) -> str:
    # Cache stemming results (bounded at 5000 entries) since the same tokens recur constantly across pages.
    return STEMMER.stem(token)


def normalize_tuebingen(text: str) -> str:
    # Fold all spelling variants of "Tübingen"/"Tuebingen" to one ASCII form to ease search later.
    text = TUEBINGEN_UMLAUT_RE.sub("tubingen", text)
    text = TUEBINGEN_ASCII_RE.sub("tubingen", text)
    text = TUEBINGEN_UBINGEN_RE.sub("tubingen", text)
    return text


def preprocess(text: str, use_stemming: bool = True) -> list[str]:
    # Lowercase, normalize, tokenize, strip stopwords/short tokens, and optionally stem the text.
    text = normalize_tuebingen(text or "").lower()
    tokens = TOKEN_PATTERN.findall(text)

    # Hoisted out of the loop: this condition is constant for the whole call.
    should_stem = use_stemming and STEMMER is not None

    cleaned: list[str] = []
    for token in tokens:
        if token in STOPWORDS or len(token) <= 1:
            continue
        if should_stem and token != "tubingen":
            token = _stem_cached(token)
        cleaned.append(token)
    return cleaned


def tokens_from_list(values: Iterable[str]) -> list[str]:
    # Join a list of strings into one text blob and preprocess it into tokens.
    return preprocess(" ".join(values))


def _process_page(page: dict) -> dict:
    # Preprocess a single page's text fields.
    title_tokens = preprocess(page.get("title", ""))
    heading_tokens = tokens_from_list(page.get("headings", []))
    body_tokens = preprocess(page.get("body", ""))
    return {
        "doc_id": page.get("doc_id"),
        "url": page.get("url", ""),
        "title": page.get("title", ""),
        "title_tokens": title_tokens,
        "heading_tokens": heading_tokens,
        "body_tokens_preview": body_tokens[:80],
        "body_length": len(body_tokens),
        "snippet": short_snippet(page.get("body", "")),
    }


def create_preprocessed_pages(
    raw_pages_path: str | Path,
    output_path: str | Path,
    use_multiprocessing: bool = True,
    processes: int | None = None,
) -> dict:
    # Load raw page data, preprocess each pages text fields, and write the results to JSON.
    # Pages are independent of one another, so for large inputs the per-page work is farmed
    # out to a process pool; small inputs run sequentially to avoid pool startup overhead.
    start_time = time.perf_counter()

    raw = read_json(raw_pages_path, {"pages": []})
    pages = raw.get("pages", [])

    parallel = use_multiprocessing and len(pages) >= MP_PAGE_THRESHOLD
    if parallel:
        num_workers = max(1, min(processes or cpu_count(), len(pages)))
        with Pool(processes=num_workers) as pool:
            documents = pool.map(_process_page, pages)
    else:
        num_workers = 1
        documents = [_process_page(page) for page in pages]

    total_body_tokens = sum(doc["body_length"] for doc in documents)

    output = {"documents": documents}
    write_json(output_path, output)

    elapsed_seconds = time.perf_counter() - start_time
    mode = f"multiprocessing ({num_workers} workers)" if parallel else "sequential"
    print(f"preprocess: processed {len(documents)} pages in {elapsed_seconds:.3f}s [{mode}]")

    summary = {
        "num_pages": len(documents),
        "total_body_tokens": total_body_tokens,
        "elapsed_seconds": elapsed_seconds,
        "mode": mode,
        "num_workers": num_workers,
    }
    if parallel:
        # Each worker process has its own lru_cache, so hit/miss counts cant be meaningfully aggregated across processes.
        summary["stemming_cache"] = {"maxsize": _stem_cached.cache_info().maxsize, "note": "per-worker caches, not aggregated"}
    else:
        cache_info = _stem_cached.cache_info()
        summary["stemming_cache"] = {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
        }

    SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(SUMMARY_OUTPUT_PATH, summary)

    return output
