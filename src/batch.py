import csv
import json
import time
from pathlib import Path

from src.reranking import rerank
from src.retrieval import retrieve
from src.utils import write_json


def load_index(path: str = "data/index.json") -> dict:
    """Load the JSON search index used for batch retrieval."""
    index_path = Path(path)
    with index_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_batch_queries(path: str = "queries.tsv") -> list[dict]:
    """Load batch queries from tab-separated text file.

    Each line is expected to contain:
    query_id <TAB> query text.
    """
    input_path = Path(path)
    queries: list[dict] = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if not row:
                continue

            query_id, query = row
            queries.append({"query_id": query_id.strip(), "query": query.strip()})

    return queries


def get_result_url(result: dict) -> str:
    """Return the best available URL for an output result row."""
    return (
        result.get("url")
        or result.get("fetched_url")
        or result.get("canonical_url")
        or ""
    )


def get_result_score(result: dict) -> float:
    """Return the final score for an output result row."""
    return float(result.get("score", result.get("bm25_score", 0.0)))


def format_result_rows(
    query_id: str,
    results: list[dict],
    top_k: int = 100,
) -> list[list[str]]:
    """Format ranked search results as tabular output rows."""
    rows: list[list[str]] = []

    for rank, result in enumerate(results[:top_k], start=1):
        document_url = get_result_url(result)
        if not document_url:
            continue

        rows.append([query_id, str(rank), document_url, f"{get_result_score(result):.6f}"])

    return rows


def write_result_rows(path: str, rows: list[list[str]]) -> None:
    """Write batch result rows as TSV without a header."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def write_summary(path: str, summary: dict) -> None:
    """Write the internal batch summary as UTF-8 JSON."""
    write_json(path, summary)


def run_single_batch_query(
    query_id: str,
    query: str,
    index: dict,
    top_k: int = 100,
    use_reranking: bool = True,
    assume_tuebingen_context: bool = False,
) -> dict:
    """Run retrieval for a single batch query and return ranked results."""
    start_time = time.perf_counter()

    retrieval_result = retrieve(
        query,
        index,
        top_k=top_k,
        assume_tuebingen_context=assume_tuebingen_context,
    )
    retrieval_result["query_id"] = query_id

    if use_reranking:
        reranked_result = rerank(retrieval_result, index)
        results = reranked_result["results"]
    else:
        results = retrieval_result["candidates"]

    runtime_seconds = round(time.perf_counter() - start_time, 4)

    return {
        "query_id": query_id,
        "query": query,
        "assume_tuebingen_context": assume_tuebingen_context,
        "results": results,
        "num_results": len(results),
        "runtime_seconds": runtime_seconds,
    }


def run_batch(
    input_path: str = "queries.tsv",
    index_path: str = "data/index.json",
    output_path: str = "data/results.tsv",
    summary_output_path: str = "data/batch_summary.json",
    top_k: int = 100,
    use_reranking: bool = True,
    assume_tuebingen_context: bool = False,
) -> dict:
    """Run batch retrieval and write official TSV results plus a JSON summary."""
    start_time = time.perf_counter()

    queries = load_batch_queries(input_path)

    index = load_index(index_path)

    query_results: list[dict] = []
    result_rows: list[list[str]] = []

    for query_entry in queries:
        query_result = run_single_batch_query(
            query_id=query_entry["query_id"],
            query=query_entry["query"],
            index=index,
            top_k=top_k,
            use_reranking=use_reranking,
            assume_tuebingen_context=assume_tuebingen_context,
        )
        query_results.append(query_result)
        result_rows.extend(
            format_result_rows(
                query_id=query_result["query_id"],
                results=query_result["results"],
                top_k=top_k,
            )
        )

    write_result_rows(output_path, result_rows)

    elapsed_seconds = round(time.perf_counter() - start_time, 4)
    query_count = len(query_results)
    total_runtime = sum(query_result["runtime_seconds"] for query_result in query_results)
    average_runtime_seconds = round(total_runtime / query_count, 4) if query_count else 0.0

    summary = {
        "step": "batch_retrieval",
        "input_path": input_path,
        "index_path": index_path,
        "output_path": output_path,
        "summary_output_path": summary_output_path,
        "query_count": query_count,
        "total_results": len(result_rows),
        "top_k": top_k,
        "use_reranking": use_reranking,
        "assume_tuebingen_context": assume_tuebingen_context,
        "elapsed_seconds": elapsed_seconds,
        "average_runtime_seconds": average_runtime_seconds,
    }

    write_summary(summary_output_path, summary)

    return {
        "queries": query_results,
        "summary": summary,
    }
