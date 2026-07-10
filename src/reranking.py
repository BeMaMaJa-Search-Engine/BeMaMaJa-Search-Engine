import json

def compute_field_boost(query_tokens: list[str], target_tokens: list[str], weight: float) -> float:
    """
    Berechnet den Feld-Boost basierend auf bereits vorverarbeiteten Tokens.
    """
    if not target_tokens or not query_tokens:
        return 0.0
    
    # Zähle Matches zwischen einzigartigen Query-Tokens und den Feld-Tokens
    matches = sum(target_tokens.count(token) for token in set(query_tokens) if token in target_tokens)
    return float(matches * weight)


def rerank(retrieval_results, index, title_weight=2.0, heading_weight=1.0, bm25_importance=0.7, field_importance=0.3):
    """
    Finale Version des Field-Boostings mit Score-Normalisierung.
    
    :param retrieval_results: Ergebnisse aus src.retrieval.retrieve()
    :param index: Pfad zur 'data/index.json' oder geladenes Dictionary
    :param title_weight: Gewichtung für Treffer im Titel
    :param heading_weight: Gewichtung für Treffer in Überschriften (Headings)
    :param bm25_importance: Interpolationsgewicht für BM25 (0.0 - 1.0)
    :param field_importance: Interpolationsgewicht für den Field Boost (0.0 - 1.0)
    """
    # Index laden, um Zugriff auf alle preprocessed Felder zu haben
    if isinstance(index, str):
        with open(index, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
    else:
        index_data = index

    # schnelles Lookup für die Dokumente im Index
    doc_lookup = {str(d["doc_id"]): d for d in index_data.get("documents", [])}

    candidates = retrieval_results.get("candidates", [])
    query_tokens = retrieval_results.get("query_tokens", [])
    
    if not candidates:
        return {"query_id": retrieval_results.get("query_id", "1"), "query": retrieval_results.get("query", ""), "results": []}

    # Listen für die rohen Scores
    raw_bm25_scores = []
    raw_field_scores = []
    
    # Temporäre Liste zum Zwischenspeichern
    temp_candidates = []

    # Rohe Scores berechnen
    for candidate in candidates:
        doc_id = str(candidate["doc_id"])
        bm25_score = candidate.get("bm25_score", 0.0)
        
        # Tokens aus dem Index
        indexed_doc = doc_lookup.get(doc_id, {})
        title_tokens = indexed_doc.get("title_tokens", [])
        heading_tokens = indexed_doc.get("heading_tokens", [])

        # Wenn die Tokens nicht im Index stehen, nutzen wir das rohe Textfeld als Fallback
        if not title_tokens and "title" in candidate:
            from src.preprocessing import preprocess
            title_tokens = preprocess(candidate["title"])

        # Berechne individuelle Feld-Boosts
        t_boost = compute_field_boost(query_tokens, title_tokens, title_weight)
        h_boost = compute_field_boost(query_tokens, heading_tokens, heading_weight)
        total_field_score = t_boost + h_boost

        raw_bm25_scores.append(bm25_score)
        raw_field_scores.append(total_field_score)

        temp_candidates.append({
            "candidate": candidate,
            "raw_bm25": bm25_score,
            "raw_field": total_field_score
        })

    # Min-Max-Normalisierung vorbereiten
    min_bm25, max_bm25 = min(raw_bm25_scores), max(raw_bm25_scores)
    min_field, max_field = min(raw_field_scores), max(raw_field_scores)

    # Hilfsfunktion zur Normalisierung
    def normalize(value, min_v, max_v):
        if max_v == min_v:
            return 1.0 if max_v > 0 else 0.0
        return (value - min_v) / (max_v - min_v)

    reranked_candidates = []

    # Kombinierten und gewichteten Score berechnen
    for item in temp_candidates:
        norm_bm25 = normalize(item["raw_bm25"], min_bm25, max_bm25)
        norm_field = normalize(item["raw_field"], min_field, max_field)

        # Lineare Kombination
        final_score = (bm25_importance * norm_bm25) + (field_importance * norm_field)

        updated_candidate = item["candidate"].copy()
        updated_candidate["score"] = round(final_score, 4)
        
        # speichern für die UI
        updated_candidate["score_details"] = {
            "normalized_bm25": round(norm_bm25, 4),
            "bm25_component": round(norm_bm25, 4),
            "normalized_field_boost": round(norm_field, 4),
            "field_component": round(norm_field, 4)
        }
        
        reranked_candidates.append(updated_candidate)

    # sortieren
    reranked_candidates = sorted(reranked_candidates, key=lambda x: x["score"], reverse=True)
    for rank, candidate in enumerate(reranked_candidates, start=1):
        candidate["rank"] = rank

    return {
        "query_id": retrieval_results.get("query_id", "1"),
        "query": retrieval_results.get("query", ""),
        "results": reranked_candidates
    }