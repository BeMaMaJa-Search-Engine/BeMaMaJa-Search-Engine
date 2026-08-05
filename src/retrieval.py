import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
import nltk

try:
    from src.preprocessing import preprocess
except ImportError:
    def preprocess(text):
        # Einfacher Fallback
        return text.lower().split()

def compute_idf(N, df):
    """
    Berechnet den Okapi BM25 IDF-Wert.
    Das '+ 1.0' im Logarithmus verhindert negative IDF-Werte bei sehr häufigen Termen.
    """
    return math.log(1.0 + (N - df + 0.5) / (df + 0.5))

_CACHE = {}
TUEBINGEN_CONTEXT_TOKEN = "tubingen"
CONTROLLED_EXPANSION_WEIGHT = 0.10
CONTROLLED_QUERY_EXPANSIONS = {
    "food": {"cafe", "dine", "eat", "restaur"},
    "drink": {"bar", "beverag", "cafe", "pub"},
    "hous": {"accommod", "apart", "room"},
    "attract": {"castl", "museum", "sightse"},
}

def _get_prepared_index(index):
    """
    Lädt den Index und baut alle Lookups und die Rechtschreib-Buckets.
    Nutzt Caching für Speed-up.
    """
    cache_key = index if isinstance(index, str) else id(index)
    
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    # nur einmal json laden für speedup
    if isinstance(index, str):
        with open(index, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
    else:
        index_data = index

    documents = index_data.get("documents", [])
    frequencies = index_data.get("document_frequencies", {})
    
    # BM25 term frequencies contain title, heading and body tokens. Use the
    # matching total field length so those extra fields are normalized too.
    field_lengths = index_data.get("field_lengths", {})
    body_lengths = field_lengths.get("body", {})
    title_lengths = field_lengths.get("title", {})
    heading_lengths = field_lengths.get("heading", {})
    doc_lengths = {}
    for doc in documents:
        doc_id = doc["doc_id"]
        doc_id_key = str(doc_id)
        total_length = (
            body_lengths.get(doc_id_key, doc.get("doc_length", 0))
            + title_lengths.get(doc_id_key, 0)
            + heading_lengths.get(doc_id_key, 0)
        )
        doc_lengths[doc_id] = max(1, total_length)
    average_document_length = (
        sum(doc_lengths.values()) / len(doc_lengths)
        if doc_lengths
        else 1.0
    )
    doc_metadata = {doc["doc_id"]: doc for doc in documents}

    # Rechtschreib-Buckets vorbauen (Anfangsbuchstabe und Länge)
    spelling_buckets = defaultdict(list)
    for term, freq in frequencies.items():
        if term:
            spelling_buckets[(term[0], len(term))].append((term, freq))
            
    # Für schnelles Tie-Breaking direkt nach Häufigkeit absteigend sortieren
    for key in spelling_buckets:
        spelling_buckets[key].sort(key=lambda x: x[1], reverse=True)

    # Im Cache speichern
    prepared_data = (
        index_data,
        doc_lengths,
        doc_metadata,
        frequencies,
        spelling_buckets,
        average_document_length,
    )
    _CACHE[cache_key] = prepared_data
    _CACHE[id(index_data)] = prepared_data  # Verhindert Cache-Miss
    
    return prepared_data

def _bounded_edit_distance(source: str, target: str, max_distance: int) -> int:
    """Reject impossible candidates, otherwise use NLTK's exact distance."""
    if abs(len(source) - len(target)) > max_distance:
        return max_distance + 1

    character_difference = Counter(source)
    character_difference.subtract(target)
    lower_bound = (sum(abs(count) for count in character_difference.values()) + 1) // 2
    if lower_bound > max_distance:
        return max_distance + 1

    return nltk.edit_distance(source, target, transpositions=True)


def _correct_query_spelling_uncached(query_tokens: list[str], index_data: dict, max_distance: int = 2) -> list[str]:
    """
    Rechtschreibkorrektur mit O(1)-Bucket-Lookups und Frequenz-Tie-Breaking.
    """
    # Holt sich buckets aus dem Cache
    _, _, _, frequencies, spelling_buckets, _ = _get_prepared_index(index_data)
    
    corrected_tokens = []

    for token in query_tokens:
        # O(1) Hash-Lookup für existierende oder kurze Wörter
        if token in frequencies or token == "tubingen" or token.isdigit() or len(token) <= 3:
            corrected_tokens.append(token)
            continue

        allowed_distance = 1 if len(token) <= 5 else max_distance
        t_len = len(token)
        t_char = token[0]

        # Nur realistische Kandidaten
        candidates = []
        for length_diff in range(-allowed_distance, allowed_distance + 1):
            bucket = spelling_buckets.get((t_char, t_len + length_diff), [])
            candidates.extend(bucket)

        if not candidates:
            corrected_tokens.append(token)
            continue

        # Levenshtein auf reduzierte anzahl
        best_term = None
        min_dist = allowed_distance + 1
        best_freq = -1

        for term, freq in candidates:
            if min_dist == 1 and freq <= best_freq:
                continue

            distance = _bounded_edit_distance(token, term, allowed_distance)
            
            if distance < min_dist or (distance == min_dist and freq > best_freq):
                min_dist = distance
                best_term = term
                best_freq = freq

        if best_term and min_dist <= allowed_distance:
            # Canonicalize corrected terms with the same stemming pipeline used
            # for normal queries, so e.g. attractionu -> attraction -> attract.
            normalized_term = preprocess(best_term)
            if len(normalized_term) == 1 and normalized_term[0] in frequencies:
                best_term = normalized_term[0]
            corrected_tokens.append(best_term)
        else:
            corrected_tokens.append(token)

    return corrected_tokens


@lru_cache(maxsize=10_000)
def _correct_spelling_token(index_cache_key: int, token: str, max_distance: int) -> str:
    """Cache corrections across repeated queries and Streamlit reruns."""
    prepared_index = _CACHE.get(index_cache_key)
    if prepared_index is None:
        return token
    index_data = prepared_index[0]
    return _correct_query_spelling_uncached([token], index_data, max_distance)[0]


def correct_query_spelling(query_tokens: list[str], index_data: dict, max_distance: int = 2) -> list[str]:
    """Correct query tokens while reusing earlier token-level results."""
    _get_prepared_index(index_data)
    index_cache_key = id(index_data)
    return [
        _correct_spelling_token(index_cache_key, token, max_distance)
        for token in query_tokens
    ]


def add_tuebingen_context(query_tokens: list[str]) -> list[str]:
    """
    Add Tuebingen as an implicit local-search context token if it is not already present.
    """
    if TUEBINGEN_CONTEXT_TOKEN in set(query_tokens):
        return query_tokens
    return query_tokens + [TUEBINGEN_CONTEXT_TOKEN]


def get_controlled_expansion_terms(query_tokens: list[str]) -> list[str]:
    """Return a small transparent set of low-weight category expansion terms."""
    original_tokens = set(query_tokens)
    expansion_terms = set()
    for token in original_tokens:
        expansion_terms.update(CONTROLLED_QUERY_EXPANSIONS.get(token, set()))
    return sorted(expansion_terms - original_tokens)


def get_controlled_expansion_field_tokens(document: dict) -> set[str]:
    """Return prominent field tokens only for a Tuebingen-central document."""
    title_tokens = set(document.get("title_tokens", []))
    heading_tokens = set(document.get("heading_tokens", [])[:50])
    url_text = " ".join(
        [
            document.get("canonical_url", ""),
            document.get("fetched_url", ""),
            document.get("url", ""),
        ]
    ).lower()
    is_tuebingen_central = (
        TUEBINGEN_CONTEXT_TOKEN in title_tokens
        or "tuebingen" in url_text
        or "tubingen" in url_text
    )
    return title_tokens | heading_tokens if is_tuebingen_central else set()


def retrieve(query, index, top_k=1000, k1=1.2, b=0.75, assume_tuebingen_context=True):
    """
    First-Stage-Retrieval mit dem BM25-Algorithmus.
    
    :param query: Der Suchstring des Nutzers (str)
    :param index: Pfad zur 'data/index.json' (str) oder bereits geladenes Dictionary
    :param top_k: Anzahl der zurückzugebenden Dokumente (Standard: 100)
    :param k1: BM25 Term-Sättigungsparameter (Standard: 1.2)
    :param b: BM25 Längennormalisierungsparameter (Standard: 0.75)
    :return: Ein Dictionary mit Query-Infos und den Top-100-Kandidaten
    """
    # nutzt den cach für schnelleres laden
    (
        index_data,
        doc_lengths,
        doc_metadata,
        doc_frequencies,
        _,
        avgdl,
    ) = _get_prepared_index(index)

    documents = index_data.get("documents", [])
    inverted_index = index_data.get("inverted_index", {})
    
    # Gesamtanzahl der Dokumente (N)
    N = len(documents)
    if N == 0:
        return {
            "query": query,
            "query_tokens": [],
            "assume_tuebingen_context": assume_tuebingen_context,
            "candidates": [],
        }

    query_tokens = preprocess(query)
    query_tokens = correct_query_spelling(query_tokens, index_data)
    tuebingen_is_explicit = TUEBINGEN_CONTEXT_TOKEN in set(query_tokens)
    if assume_tuebingen_context:
        query_tokens = add_tuebingen_context(query_tokens)

    controlled_expansion_terms = get_controlled_expansion_terms(query_tokens)
    controlled_expansion_term_set = set(controlled_expansion_terms)
    weighted_query_terms = {token: 1.0 for token in query_tokens}
    for token in controlled_expansion_terms:
        weighted_query_terms[token] = CONTROLLED_EXPANSION_WEIGHT

    # BM25 Scoring (Term-at-a-Time Ansatz)
    scores = defaultdict(float)
    matched_terms_per_doc = defaultdict(list)
    matched_expansion_terms_per_doc = defaultdict(list)
    controlled_expansion_fields_by_doc = {}

    for token, query_weight in weighted_query_terms.items():
        if token not in inverted_index:
            continue
            
        # IDF für den aktuellen Query-Term berechnen
        df = doc_frequencies.get(token, len(inverted_index[token]))
        idf = compute_idf(N, df)
        # Explicit Tuebingen keeps the existing boost. Automatically added
        # Tuebingen remains a normal context term with factor 1.0.
        if token == TUEBINGEN_CONTEXT_TOKEN and tuebingen_is_explicit:
            idf *= 2.0
        
        # Posting-Liste ablaufen
        for posting in inverted_index[token]:
            doc_id = posting["doc_id"]
            if token in controlled_expansion_term_set:
                if doc_id not in controlled_expansion_fields_by_doc:
                    controlled_expansion_fields_by_doc[doc_id] = (
                        get_controlled_expansion_field_tokens(doc_metadata.get(doc_id, {}))
                    )
                if token not in controlled_expansion_fields_by_doc[doc_id]:
                    continue
            tf = posting["tf"]
            doc_len = doc_lengths.get(doc_id, avgdl)
            
            # BM25 Formel anwenden
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
            score_contribution = query_weight * idf * (numerator / denominator)

            scores[doc_id] += score_contribution
            if token in controlled_expansion_term_set:
                matched_expansion_terms_per_doc[doc_id].append(token)
            else:
                matched_terms_per_doc[doc_id].append(token)

    # Dokumente nach Score absteigend sortieren und Top-K auswählen
    ranked_doc_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)[:top_k]

    # Ergebnis im geforderten JSON-Format aufbereiten
    candidates = []
    for doc_id in ranked_doc_ids:
        meta = doc_metadata.get(doc_id, {})
        candidates.append({
            "doc_id": doc_id,
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "snippet": meta.get("snippet", ""),
            "bm25_score": round(scores[doc_id], 4),
            "matched_terms": matched_terms_per_doc[doc_id],
            "matched_expansion_terms": matched_expansion_terms_per_doc[doc_id],
            "controlled_expansion_terms": controlled_expansion_terms,
        })

    return {
        "query": query,
        "query_tokens": query_tokens,
        "controlled_expansion_terms": controlled_expansion_terms,
        "tuebingen_context_implicit": assume_tuebingen_context and not tuebingen_is_explicit,
        "assume_tuebingen_context": assume_tuebingen_context,
        "candidates": candidates
    }
