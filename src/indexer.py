import json
import time
from collections import Counter
from pathlib import Path


def load_preprocessed_documents(path: str = "data/preprocessed_pages.json") -> list[dict]:
    """Load preprocessed documents from a JSON file.

    The indexer expects the preprocessing output format:
    {"documents": [...]}.
    """
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported preprocessed data in {input_path}: top-level JSON must be an object.")

    if "documents" not in data:
        raise ValueError(f"Unsupported preprocessed data in {input_path}: missing 'documents' field.")

    documents = data["documents"]
    if not isinstance(documents, list):
        raise ValueError(f"Unsupported preprocessed data in {input_path}: 'documents' must be a list.")

    return documents


def build_document_metadata(documents: list[dict]) -> list[dict]:
    """Create document metadata entries for the index."""
    metadata: list[dict] = []

    for document in documents:
        doc_id = document.get("doc_id")
        if doc_id is None:
            raise ValueError("Cannot build document metadata: document is missing 'doc_id'.")

        body_tokens = document.get("body_tokens", [])
        doc_length = document.get("body_length", len(body_tokens))

        metadata.append(
            {
                "doc_id": doc_id,
                "url": document.get("url", ""),
                "fetched_url": document.get("fetched_url", ""),
                "canonical_url": document.get("canonical_url", ""),
                "title": document.get("title", ""),
                "snippet": document.get("snippet", ""),
                "title_tokens": document.get("title_tokens", []),
                "heading_tokens": document.get("heading_tokens", []),
                "doc_length": doc_length,
                "outgoing_links": document.get("outgoing_links", []),
                "crawl_time": document.get("crawl_time", ""),
            }
        )

    return metadata


def build_inverted_index(documents: list[dict]) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Build an inverted index and document frequencies from preprocessed documents."""
    postings_by_term: dict[str, list[dict]] = {}
    document_frequencies: dict[str, int] = {}

    for document in documents:
        doc_id = document.get("doc_id")
        if doc_id is None:
            continue

        tokens = (
            document.get("title_tokens", [])
            + document.get("heading_tokens", [])
            + document.get("body_tokens", [])
        )
        if not tokens:
            continue

        term_counts = Counter(tokens)
        for term, frequency in term_counts.items():
            postings_by_term.setdefault(term, []).append({"doc_id": doc_id, "tf": frequency})
            document_frequencies[term] = document_frequencies.get(term, 0) + 1

    inverted_index = {
        term: sorted(postings, key=lambda posting: posting["doc_id"])
        for term, postings in sorted(postings_by_term.items())
    }
    sorted_document_frequencies = dict(sorted(document_frequencies.items()))

    return inverted_index, sorted_document_frequencies


def build_field_lengths(documents: list[dict]) -> dict[str, dict[str, int]]:
    """Build per-field token length mappings keyed by document id."""
    field_lengths = {
        "body": {},
        "title": {},
        "heading": {},
    }

    for document in documents:
        doc_id = document.get("doc_id")
        if doc_id is None:
            continue

        body_tokens = document.get("body_tokens", [])
        doc_id_key = str(doc_id)

        field_lengths["body"][doc_id_key] = document.get("body_length", len(body_tokens))
        field_lengths["title"][doc_id_key] = len(document.get("title_tokens", []))
        field_lengths["heading"][doc_id_key] = len(document.get("heading_tokens", []))

    return field_lengths


def compute_average_document_length(documents: list[dict]) -> float:
    """Compute the average body length across all documents."""
    if not documents:
        return 0.0

    total_length = 0
    for document in documents:
        body_tokens = document.get("body_tokens", [])
        total_length += document.get("body_length", len(body_tokens))

    return total_length / len(documents)


def build_link_graph(documents: list[dict]) -> dict[str, list[int]]:
    """Build an internal link graph between indexed documents."""
    
    # Build a mapping from URLs to document IDs for quick lookup
    url_to_doc_id: dict[str, int] = {}
    for document in documents:
        doc_id = document.get("doc_id")
        if doc_id is None:
            continue

        url = document.get("url", "")
        fetched_url = document.get("fetched_url", "")
        canonical_url = document.get("canonical_url", "")
        if url:
            url_to_doc_id[url] = doc_id
        if fetched_url:
            url_to_doc_id[fetched_url] = doc_id
        if canonical_url:
            url_to_doc_id[canonical_url] = doc_id

    # Build the link graph by mapping each document to the outgoing links
    link_graph: dict[str, list[int]] = {}
    for document in documents:
        doc_id = document.get("doc_id")
        if doc_id is None:
            continue

        linked_doc_ids = set()
        for outgoing_link in document.get("outgoing_links", []):
            target_doc_id = url_to_doc_id.get(outgoing_link)
            if target_doc_id is None or target_doc_id == doc_id:
                continue
            linked_doc_ids.add(target_doc_id)

        link_graph[str(doc_id)] = sorted(linked_doc_ids)

    return link_graph


def build_index_summary(
    documents: list[dict],
    inverted_index: dict[str, list[dict]],
    average_document_length: float,
    link_graph: dict[str, list[int]],
) -> dict:
    """Build compact statistics for the indexing step."""
    documents_with_outgoing_links = sum(1 for document in documents if document.get("outgoing_links"))

    return {
        "step": "indexing",
        "num_docs": len(documents),
        "vocabulary_size": len(inverted_index),
        "average_document_length": round(average_document_length, 4),
        "documents_with_outgoing_links": documents_with_outgoing_links,
    }


def build_index(
    input_path: str = "data/preprocessed_pages.json",
    output_path: str = "data/index.json",
    summary_output_path: str = "data/index_summary.json",
) -> dict:
    """Build the complete search index and write index and summary JSON files."""
    start_time = time.perf_counter()

    documents = load_preprocessed_documents(input_path)
    document_metadata = build_document_metadata(documents)
    inverted_index, document_frequencies = build_inverted_index(documents)
    field_lengths = build_field_lengths(documents)
    average_document_length = compute_average_document_length(documents)
    link_graph = build_link_graph(documents)
    summary = build_index_summary(documents, inverted_index, average_document_length, link_graph)

    index = {
        "documents": document_metadata,
        "inverted_index": inverted_index,
        "document_frequencies": document_frequencies,
        "field_lengths": field_lengths,
        "average_document_length": average_document_length,
        "link_graph": link_graph,
    }

    index_output_path = Path(output_path)
    index_output_path.parent.mkdir(parents=True, exist_ok=True)
    with index_output_path.open("w", encoding="utf-8") as file:
        json.dump(index, file, indent=2, ensure_ascii=False)

    summary["index_file_size_mb"] = round(index_output_path.stat().st_size / (1024 * 1024), 2)
    summary["elapsed_seconds"] = round(time.perf_counter() - start_time, 4)

    summary_path = Path(summary_output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return {
        "index": index,
        "summary": summary,
    }
