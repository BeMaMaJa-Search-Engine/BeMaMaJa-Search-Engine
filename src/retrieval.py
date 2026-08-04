import json
import math
from collections import defaultdict
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
    
    # Lookups BM25-Berechnung vorbauen
    doc_lengths = {doc["doc_id"]: doc.get("doc_length", 1) for doc in documents}
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
    prepared_data = (index_data, doc_lengths, doc_metadata, frequencies, spelling_buckets)
    _CACHE[cache_key] = prepared_data
    _CACHE[id(index_data)] = prepared_data  # Verhindert Cache-Miss
    
    return prepared_data

def correct_query_spelling(query_tokens: list[str], index_data: dict, max_distance: int = 2) -> list[str]:
    """
    Rechtschreibkorrektur mit O(1)-Bucket-Lookups und Frequenz-Tie-Breaking.
    """
    # Holt sich buckets aus dem Cache
    _, _, _, frequencies, spelling_buckets = _get_prepared_index(index_data)
    
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

            distance = nltk.edit_distance(token, term, transpositions=True)
            
            if distance < min_dist or (distance == min_dist and freq > best_freq):
                min_dist = distance
                best_term = term
                best_freq = freq

        if best_term and min_dist <= allowed_distance:
            corrected_tokens.append(best_term)
        else:
            corrected_tokens.append(token)

    return corrected_tokens


def add_tuebingen_context(query_tokens: list[str]) -> list[str]:
    """
    Add Tuebingen as an implicit local-search context token if it is not already present.
    """
    if TUEBINGEN_CONTEXT_TOKEN in set(query_tokens):
        return query_tokens
    return query_tokens + [TUEBINGEN_CONTEXT_TOKEN]


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
    index_data, doc_lengths, doc_metadata, doc_frequencies, _ = _get_prepared_index(index)

    documents = index_data.get("documents", [])
    inverted_index = index_data.get("inverted_index", {})
    avgdl = index_data.get("average_document_length", 1.0)
    
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
    if assume_tuebingen_context:
        query_tokens = add_tuebingen_context(query_tokens)

    # BM25 Scoring (Term-at-a-Time Ansatz)
    scores = defaultdict(float)
    matched_terms_per_doc = defaultdict(list)

    for token in set(query_tokens):
        if token not in inverted_index:
            continue
            
        # IDF für den aktuellen Query-Term berechnen
        df = doc_frequencies.get(token, len(inverted_index[token]))
        idf = compute_idf(N, df)
        # boosting tubingen since we have squed data
        if token == "tubingen":
            idf *= 2.0
        
        # Posting-Liste ablaufen
        for posting in inverted_index[token]:
            doc_id = posting["doc_id"]
            tf = posting["tf"]
            doc_len = doc_lengths.get(doc_id, avgdl)
            
            # BM25 Formel anwenden
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
            score_contribution = idf * (numerator / denominator)
            
            scores[doc_id] += score_contribution
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
            "matched_terms": matched_terms_per_doc[doc_id]
        })

    return {
        "query": query,
        "query_tokens": query_tokens,
        "assume_tuebingen_context": assume_tuebingen_context,
        "candidates": candidates
    }
