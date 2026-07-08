from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess
from src.retrieval import retrieve
from src.utils import project_path, read_json


def _highlight(text: str, terms: list[str]) -> str:
    escaped = html.escape(text or "")
    for term in sorted(set(terms), key=len, reverse=True):
        if len(term) > 2:
            escaped = escaped.replace(term, f"<mark>{term}</mark>")
            escaped = escaped.replace(term.capitalize(), f"<mark>{term.capitalize()}</mark>")
    return escaped


def _score_details(details: dict) -> str:
    if not details:
        return "No score details available."
    parts = []
    for key, value in details.items():
        if isinstance(value, (int, float)):
            parts.append(f"{key}: {value:.3f}")
    return " | ".join(parts)


def main() -> None:
    st.set_page_config(page_title="Tuebingen Search Engine", layout="wide")
    st.title("Tuebingen Search Engine")

    index = read_json(project_path("data", "index.json"), {})
    documents = index.get("documents", [])
    categories = ["All"] + sorted({doc.get("category", "General") for doc in documents})

    with st.sidebar:
        st.header("Filters")
        selected_category = st.selectbox("Category", categories)
        top_k = st.slider("Results", 5, 50, 10)
        st.caption(f"Indexed pages: {len(documents)}")

    query = st.text_input("Search", value="tuebingen attractions", placeholder="Try: food and drinks")

    if not query:
        st.info("Enter a query to search the local JSON index.")
        return

    start = time.perf_counter()
    results = retrieve(query, top_k=top_k)
    runtime = time.perf_counter() - start
    if selected_category != "All":
        results = [result for result in results if result.get("category") == selected_category]

    query_terms = preprocess(query, use_stemming=False)
    expansion_terms = results[0].get("expansion_terms", []) if results else []

    col1, col2, col3 = st.columns(3)
    col1.metric("Search time", f"{runtime:.3f}s")
    col2.metric("Indexed pages", len(documents))
    col3.metric("Shown results", len(results))
    if expansion_terms:
        st.caption("PRF expansion terms: " + ", ".join(expansion_terms))

    if not documents:
        st.warning("No index found yet. Run `python scripts/build_index.py` first.")
        return

    if not results:
        st.warning("No results found for this query/filter.")
        return

    for result in results:
        with st.container(border=True):
            st.subheader(f"{result['rank']}. {result.get('title') or result.get('url')}")
            st.markdown(f"[{result.get('url')}]({result.get('url')})")
            st.markdown(_highlight(result.get("snippet", ""), query_terms), unsafe_allow_html=True)
            st.write(f"Final score: `{result.get('score', 0.0):.4f}`")
            st.caption("Matched terms: " + (", ".join(result.get("matched_terms", [])) or "none"))
            st.caption("Category: " + result.get("category", "General"))
            with st.expander("Score breakdown and why this result?"):
                st.write(_score_details(result.get("score_details", {})))
                st.write(result.get("why_this_result", "BM25 found textual overlap with the query."))


if __name__ == "__main__":
    main()
