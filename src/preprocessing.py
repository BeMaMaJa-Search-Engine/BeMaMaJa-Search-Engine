from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import wordpunct_tokenize
except ImportError:
    nltk_stopwords = None
    PorterStemmer = None
    wordpunct_tokenize = None


FALLBACK_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

GERMAN_CHARACTER_REPLACEMENTS = {
    "\u00e4": "ae",
    "\u00c4": "ae",
    "\u00f6": "oe",
    "\u00d6": "oe",
    "\u00fc": "ue",
    "\u00dc": "ue",
    "\u00df": "ss",
}

MOJIBAKE_REPLACEMENTS = {
    "\u00c3\u00a4": "ae",
    "\u00e3\u00a4": "ae",
    "\u00c3\u201e": "ae",
    "\u00e3\u201e": "ae",
    "\u00c3\u00b6": "oe",
    "\u00e3\u00b6": "oe",
    "\u00c3\u2013": "oe",
    "\u00e3\u2013": "oe",
    "\u00c3\u00bc": "ue",
    "\u00e3\u00bc": "ue",
    "\u00c3\u0153": "ue",
    "\u00e3\u0153": "ue",
    "\u00c3\u0178": "ss",
    "\u00e3\u0178": "ss",
    "\u00c3\u0192\u00c2\u00a4": "ae",
    "\u00e3\u0192\u00c2\u00a4": "ae",
    "\u00c3\u0192\u00c5\u201e": "ae",
    "\u00e3\u0192\u00c5\u201e": "ae",
    "\u00c3\u0192\u00c2\u00b6": "oe",
    "\u00e3\u0192\u00c2\u00b6": "oe",
    "\u00c3\u0192\u00c5\u2013": "oe",
    "\u00e3\u0192\u00c5\u2013": "oe",
    "\u00c3\u0192\u00c2\u00bc": "ue",
    "\u00e3\u0192\u00c2\u00bc": "ue",
    "\u00c3\u0192\u00c5\u201c": "ue",
    "\u00e3\u0192\u00c5\u201c": "ue",
}

TUEBINGEN_PATTERN = re.compile(r"\bt(?:ue|u)bingen\b", flags=re.IGNORECASE)
STOPWORDS_CACHE: tuple[set[str], str] | None = None
PORTER_STEMMER = PorterStemmer() if PorterStemmer is not None else None


def normalize_text_variants(text: str) -> str:
    """Normalize Tuebingen spellings, encoding artifacts, and German characters."""
    if not text:
        return ""

    normalized = str(text)

    for old, new in MOJIBAKE_REPLACEMENTS.items():
        normalized = normalized.replace(old, new)

    for old, new in GERMAN_CHARACTER_REPLACEMENTS.items():
        normalized = normalized.replace(old, new)

    return TUEBINGEN_PATTERN.sub("tubingen", normalized)


def get_english_stopwords() -> tuple[set[str], str]:
    """Return English stopwords from NLTK, or a small fallback set if loading fails."""
    global STOPWORDS_CACHE

    if STOPWORDS_CACHE is not None:
        return STOPWORDS_CACHE

    if nltk_stopwords is not None:
        try:
            STOPWORDS_CACHE = (set(nltk_stopwords.words("english")), "nltk")
            return STOPWORDS_CACHE
        except Exception:
            pass

    STOPWORDS_CACHE = (set(FALLBACK_STOPWORDS), "fallback")
    return STOPWORDS_CACHE


def _tokenize(text: str) -> list[str]:
    """Tokenize text with NLTK wordpunct_tokenize, using a simple fallback if NLTK is missing."""
    if wordpunct_tokenize is not None:
        return wordpunct_tokenize(text)
    return re.findall(r"\w+|[^\w\s]", text)


def preprocess(text: str, use_stemming: bool = True) -> list[str]:
    """Normalize, tokenize, remove stopwords, and optionally stem a text string."""
    if not text:
        return []

    normalized = normalize_text_variants(str(text).lower())
    stopword_set, _ = get_english_stopwords()

    tokens = [
        token
        for token in _tokenize(normalized)
        if token.isalnum() and token not in stopword_set
    ]

    if use_stemming and PORTER_STEMMER is None:
        raise RuntimeError("NLTK PorterStemmer is required for stemming. Install nltk first.")

    if use_stemming:
        # extra Check für den Type-Checker
        assert PORTER_STEMMER is not None
        tokens = [PORTER_STEMMER.stem(token) for token in tokens]

    return tokens


def preprocess_query(query: str, use_stemming: bool = True) -> list[str]:
    """Preprocess a query with the same function used for document text."""
    return preprocess(query, use_stemming=use_stemming)


def preprocess_page(page: dict, use_stemming: bool = True) -> dict:
    """Preprocess title, headings, and body text while preserving page metadata."""
    page = page or {}
    result = {}

    metadata_fields = (
        "doc_id",
        "url",
        "fetched_url",
        "canonical_url",
        "title",
        "language",
        "is_tuebingen_related",
        "outgoing_links",
        "crawl_time",
    )
    for field in metadata_fields:
        if field in page:
            result[field] = page[field]

    title = page.get("title") or ""
    headings = page.get("headings") or []
    body = page.get("body") or ""

    if isinstance(headings, list):
        headings_text = " ".join(str(heading) for heading in headings if heading)
    else:
        headings_text = str(headings)

    title_tokens = preprocess(str(title), use_stemming=use_stemming)
    heading_tokens = preprocess(headings_text, use_stemming=use_stemming)
    body_tokens = preprocess(str(body), use_stemming=use_stemming)

    result["title_tokens"] = title_tokens
    result["heading_tokens"] = heading_tokens
    result["body_tokens"] = body_tokens
    result["body_tokens_preview"] = body_tokens[:30]
    result["body_length"] = len(body_tokens)

    return result


def load_raw_pages(path: str) -> list[dict]:
    """Load raw pages from a JSON file using either {'pages': [...]} or a plain list format."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and isinstance(data.get("pages"), list):
        pages = data["pages"]
    elif isinstance(data, list):
        pages = data
    else:
        raise ValueError(
            f"Unsupported raw pages structure in {path}. Expected {{'pages': [...]}} or [...]."
        )

    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(
                f"Unsupported page entry at position {index} in {path}. Expected a JSON object."
            )

    return pages


def build_preprocessing_summary(
    documents: list[dict],
    input_path: str,
    output_path: str,
    summary_output_path: str,
    use_stemming: bool,
    stopword_source: str,
) -> dict:
    """Build compact statistics for a preprocessing run."""
    body_lengths = [int(document.get("body_length", 0)) for document in documents]
    vocabulary = set()
    for document in documents:
        vocabulary.update(document.get("title_tokens", []))
        vocabulary.update(document.get("heading_tokens", []))
        vocabulary.update(document.get("body_tokens", []))

    num_documents = len(documents)
    average_body_length = sum(body_lengths) / num_documents if num_documents else 0.0

    return {
        "step": "preprocessing",
        "input_path": input_path,
        "output_path": output_path,
        "summary_output_path": summary_output_path,
        "use_stemming": use_stemming,
        "stopword_source": stopword_source,
        "normalization": "tubingen_aliases+german_umlauts",
        "num_documents": num_documents,
        "average_body_length": average_body_length,
        "min_body_length": min(body_lengths) if body_lengths else 0,
        "max_body_length": max(body_lengths) if body_lengths else 0,
        "empty_body_documents": sum(1 for length in body_lengths if length == 0),
        "vocabulary_size": len(vocabulary),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def preprocess_raw_pages(
    input_path: str = "data/raw_pages.json",
    output_path: str = "data/preprocessed_pages.json",
    summary_output_path: str = "data/preprocessing_summary.json",
    use_stemming: bool = True,
) -> dict:
    """Preprocess all raw pages and write document tokens plus a separate summary file."""
    pages = load_raw_pages(input_path)
    documents = [preprocess_page(page, use_stemming=use_stemming) for page in pages]
    _, stopword_source = get_english_stopwords()

    summary = build_preprocessing_summary(
        documents=documents,
        input_path=input_path,
        output_path=output_path,
        summary_output_path=summary_output_path,
        use_stemming=use_stemming,
        stopword_source=stopword_source,
    )

    output_file = Path(output_path)
    summary_file = Path(summary_output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump({"documents": documents}, file, ensure_ascii=False, indent=2)

    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return {"documents": documents, "summary": summary}
