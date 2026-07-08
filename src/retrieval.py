import json
import math
from collections import defaultdict

#TODO einkomentieren nachdem src.preprocessing implementiert wurde
#try:
    #from src.preprocessing import preprocess
#except ImportError:
    #def preprocess(text):
        # Einfacher Fallback: Kleinbuchstaben und Whitespace-Split
        #return text.lower().split()

#TODO dummy entfernen wenn src.preprocessing implementiert wurde
def preprocess(text):
        return text.lower().split()


def compute_idf(N, df):
    """
    Berechnet den Okapi BM25 IDF-Wert (Inverse Document Frequency).
    Das '+ 1.0' im Logarithmus verhindert negative IDF-Werte bei sehr häufigen Termen.
    """
    return math.log(1.0 + (N - df + 0.5) / (df + 0.5))


def retrieve(query, index, top_k=100, k1=1.2, b=0.75):
    """
    Führt das First-Stage-Retrieval mit dem klassischen BM25-Algorithmus aus.
    
    :param query: Der Suchstring des Nutzers (str)
    :param index: Pfad zur 'data/index.json' (str) oder bereits geladenes Dictionary
    :param top_k: Anzahl der zurückzugebenden Dokumente (Standard: 100 laut Vorgabe)
    :param k1: BM25 Term-Sättigungsparameter (Standard: 1.2)
    :param b: BM25 Längennormalisierungsparameter (Standard: 0.75)
    :return: Ein Dictionary mit Query-Infos und den Top-100-Kandidaten
    """
    # Index laden
    if isinstance(index, str):
        with open(index, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
    else:
        index_data = index

    # Globale Index-Statistiken extrahieren
    documents = index_data.get("documents", [])
    inverted_index = index_data.get("inverted_index", {})
    doc_frequencies = index_data.get("document_frequencies", {})
    avgdl = index_data.get("average_document_length", 1.0)
    
    # Gesamtanzahl der Dokumente (N)
    N = len(documents)
    if N == 0:
        return {"query": query, "query_tokens": [], "candidates": []}

    # Lookup-Dictionary für Dokumentenlängen (|D|) und Metadaten
    doc_lengths = {}
    doc_metadata = {}
    for doc in documents:
        doc_id = doc["doc_id"]
        doc_lengths[doc_id] = doc.get("doc_length", 1)
        doc_metadata[doc_id] = doc

    # Query preprozessieren
    query_tokens = preprocess(query)

    # BM25 Scoring (Term-at-a-Time Ansatz)
    scores = defaultdict(float)
    matched_terms_per_doc = defaultdict(list)

    for token in set(query_tokens):
        if token not in inverted_index:
            continue
            
        # IDF für den aktuellen Query-Term berechnen
        df = doc_frequencies.get(token, len(inverted_index[token]))
        idf = compute_idf(N, df)
        
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
        "candidates": candidates
    }