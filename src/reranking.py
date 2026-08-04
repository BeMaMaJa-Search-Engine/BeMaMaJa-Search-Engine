import json
import math
from collections import Counter

NON_TUEBINGEN_LOCATION_TERMS = {
    "berlin",
    "freiburg",
    "gunzburg",
    "heidelberg",
    "karlsruh",
    "konstanz",
    "ludwigsburg",
    "munich",
    "stuttgart",
    "ulm",
}


def compute_field_boost(
    query_tokens: list[str],
    title_tokens: list[str],
    heading_tokens: list[str],
    title_weight: float = 0.7,
    heading_weight: float = 0.3,
) -> float:
    """Compute a normalized field boost from unique query-token coverage."""
    unique_query_tokens = set(query_tokens)
    if not unique_query_tokens:
        return 0.0

    title_token_set = set(title_tokens)
    heading_token_set = set(heading_tokens)

    title_matches = unique_query_tokens & title_token_set
    heading_matches = unique_query_tokens & heading_token_set

    title_coverage = len(title_matches) / len(unique_query_tokens)
    heading_coverage = len(heading_matches) / len(unique_query_tokens)

    field_boost = (title_weight * title_coverage) + (heading_weight * heading_coverage)
    return max(0.0, min(1.0, field_boost))


def build_incoming_link_counts(link_graph: dict[str, list[int]]) -> dict[int, int]:
    """Count how many indexed documents link to each target document."""
    if not isinstance(link_graph, dict):
        return {}

    incoming_counts: dict[int, int] = {}

    for source_doc_id, target_doc_ids in link_graph.items():
        try:
            int(source_doc_id)
        except (TypeError, ValueError):
            continue

        if not isinstance(target_doc_ids, list):
            continue

        for target_doc_id in target_doc_ids:
            try:
                target_doc_id_int = int(target_doc_id)
            except (TypeError, ValueError):
                continue

            incoming_counts[target_doc_id_int] = incoming_counts.get(target_doc_id_int, 0) + 1

    return incoming_counts


def query_is_tuebingen_related(query_tokens: list[str]) -> bool:
    """Return True if the query explicitly contains the normalized Tuebingen token."""
    return "tubingen" in set(query_tokens)


def is_tuebingen_central_document(document: dict) -> bool:
    """Return True if title, headings, or URL metadata indicate a Tuebingen-central document."""
    title_heading_tokens = set(document.get("title_tokens", []) + document.get("heading_tokens", []))
    if "tubingen" in title_heading_tokens:
        return True

    url_text = " ".join(
        [
            document.get("url", ""),
            document.get("canonical_url", ""),
            document.get("fetched_url", ""),
        ]
    ).lower()

    return "tuebingen" in url_text or "tubingen" in url_text


def collect_prf_terms(
    candidates: list[dict],
    doc_lookup: dict[str, dict],
    query_tokens: list[str],
    document_frequencies: dict[str, int] | None = None,
    num_docs: int = 0,
    feedback_docs: int = 5,
    max_terms: int = 5,
    min_feedback_field_boost: float = 0.5,
    force_tuebingen_filter: bool = False,
) -> list[str]:
    """Collect IDF-weighted pseudo relevance feedback terms from top-ranked fields."""
    generic_tokens = {
        "thing",
        "page",
        "home",
        "contact",
        "legal",
        "compani",
        "menu",
        "search",
        "result",
    }
    query_token_set = set(query_tokens)
    require_tuebingen_centrality = force_tuebingen_filter or query_is_tuebingen_related(query_tokens)
    term_counts: Counter[str] = Counter()
    document_frequencies = document_frequencies or {}

    for candidate in candidates[:feedback_docs]:
        doc_id = candidate.get("doc_id")
        indexed_doc = doc_lookup.get(str(doc_id), {})
        if require_tuebingen_centrality and not is_tuebingen_central_document(indexed_doc):
            continue

        title_tokens = indexed_doc.get("title_tokens", [])
        heading_tokens = indexed_doc.get("heading_tokens", [])

        feedback_field_boost = compute_field_boost(
            query_tokens,
            title_tokens,
            heading_tokens,
        )
        if feedback_field_boost < min_feedback_field_boost:
            continue

        feedback_tokens = title_tokens + heading_tokens

        for token in feedback_tokens:
            if token in query_token_set:
                continue
            if len(token) < 3:
                continue
            if token in generic_tokens:
                continue
            if require_tuebingen_centrality and token in NON_TUEBINGEN_LOCATION_TERMS:
                continue
            term_counts[token] += 1

    if not term_counts:
        return []

    def prf_term_score(item: tuple[str, int]) -> tuple[float, int, str]:
        term, feedback_count = item
        df = document_frequencies.get(term, 0)
        idf = math.log((num_docs + 1) / (df + 1)) if num_docs > 0 else 1.0
        return (feedback_count * idf, feedback_count, term)

    ranked_terms = sorted(term_counts.items(), key=prf_term_score, reverse=True)
    return [term for term, _ in ranked_terms[:max_terms]]


def compute_prf_score(
    expansion_terms: list[str],
    title_tokens: list[str],
    heading_tokens: list[str],
) -> float:
    """Compute normalized coverage of PRF expansion terms in title and heading tokens."""
    unique_expansion_terms = set(expansion_terms)
    if not unique_expansion_terms:
        return 0.0

    document_tokens = set(title_tokens + heading_tokens)
    matched_terms = unique_expansion_terms & document_tokens
    prf_score = len(matched_terms) / len(unique_expansion_terms)
    return max(0.0, min(1.0, prf_score))


def rerank(
    retrieval_results,
    index,
    title_weight=0.7,
    heading_weight=0.3,
    bm25_importance=0.65,
    field_importance=0.20,
    link_importance=0.10,
    prf_importance=0.05,
    force_tuebingen_filter=False,
):
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
    incoming_link_counts = build_incoming_link_counts(index_data.get("link_graph", {}))
    document_frequencies = index_data.get("document_frequencies", {})
    num_docs = len(index_data.get("documents", []))

    candidates = retrieval_results.get("candidates", [])
    query_tokens = retrieval_results.get("query_tokens", [])
    
    if not candidates:
        return {"query_id": retrieval_results.get("query_id", "1"), "query": retrieval_results.get("query", ""), "results": []}

    # Listen für die rohen Scores
    raw_bm25_scores = []
    raw_link_scores = []
    
    # Temporäre Liste zum Zwischenspeichern
    temp_candidates = []
    expansion_terms = collect_prf_terms(
        candidates,
        doc_lookup,
        query_tokens,
        document_frequencies=document_frequencies,
        num_docs=num_docs,
        force_tuebingen_filter=force_tuebingen_filter,
    )

    # Rohe Scores berechnen
    for candidate in candidates:
        doc_id = candidate["doc_id"]
        doc_id_key = str(doc_id)
        bm25_score = candidate.get("bm25_score", 0.0)
        try:
            doc_id_int = int(doc_id)
        except (TypeError, ValueError):
            doc_id_int = None
        raw_link_score = incoming_link_counts.get(doc_id_int, 0) if doc_id_int is not None else 0
        
        # Tokens aus dem Index
        indexed_doc = doc_lookup.get(doc_id_key, {})
        title_tokens = indexed_doc.get("title_tokens", [])
        heading_tokens = indexed_doc.get("heading_tokens", [])

        # Wenn die Tokens nicht im Index stehen, nutzen wir das rohe Textfeld als Fallback
        if not title_tokens and "title" in candidate:
            from src.preprocessing import preprocess
            title_tokens = preprocess(candidate["title"])

        total_field_score = compute_field_boost(
            query_tokens,
            title_tokens,
            heading_tokens,
            title_weight=title_weight,
            heading_weight=heading_weight,
        )
        raw_prf_score = compute_prf_score(expansion_terms, title_tokens, heading_tokens)

        raw_bm25_scores.append(bm25_score)
        raw_link_scores.append(raw_link_score)

        temp_candidates.append({
            "candidate": candidate,
            "raw_bm25": bm25_score,
            "raw_field": total_field_score,
            "raw_link": raw_link_score,
            "raw_prf": raw_prf_score
        })

    # Min-Max-Normalisierung vorbereiten
    min_bm25, max_bm25 = min(raw_bm25_scores), max(raw_bm25_scores)
    min_link, max_link = min(raw_link_scores), max(raw_link_scores)

    # Hilfsfunktion zur Normalisierung
    def normalize(value, min_v, max_v):
        if max_v == min_v:
            return 1.0 if max_v > 0 else 0.0
        return (value - min_v) / (max_v - min_v)

    reranked_candidates = []

    for item in temp_candidates:
        # BM25 relativ normalisieren
        norm_bm25 = normalize(item["raw_bm25"], min_bm25, max_bm25)
        
        norm_field = item["raw_field"]

        # LinkScore relativ normalisieren
        norm_link = normalize(item["raw_link"], min_link, max_link)

        norm_prf = item["raw_prf"]

        # Lineare Kombination
        final_score = (
            (bm25_importance * norm_bm25)
            + (field_importance * norm_field)
            + (link_importance * norm_link)
            + (prf_importance * norm_prf)
        )

        updated_candidate = item["candidate"].copy()
        updated_candidate["score"] = round(final_score, 4)
        
        # speichern für die UI
        updated_candidate["score_details"] = {
            "normalized_bm25": round(norm_bm25, 4),
            "bm25_component": round(norm_bm25, 4),
            "normalized_field_boost": round(norm_field, 4),
            "field_component": round(norm_field, 4),
            "normalized_link": round(norm_link, 4),
            "link_component": round(norm_link, 4),
            "normalized_prf": round(norm_prf, 4),
            "prf_component": round(norm_prf, 4)
        }
        updated_candidate["expansion_terms"] = expansion_terms
        
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
