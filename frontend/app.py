from __future__ import annotations

import html
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess
from src.retrieval import retrieve
from src.utils import project_path, read_json


CATEGORY_RULES = {
    "Museums": ["museum", "castle", "exhibition", "collection", "collections", "heritage", "unimuseum"],
    "Attractions": ["attraction", "sightseeing", "visit", "tourism", "old town", "neckar"],
    "Food and drinks": ["restaurant", "cafe", "bar", "food", "drink", "dining", "menu", "wine"],
    "University": ["university", "study", "research", "campus", "faculty", "student", "library"],
    "Transport": ["train", "bus", "parking", "transport", "station"],
    "Events": ["event", "festival", "concert", "calendar"],
}

SOURCE_RULES = [
    ("unimuseum.uni-tuebingen.de", "Museum"),
    ("uni-tuebingen.de", "University"),
    ("tuebingen-info.de", "Tourism"),
    ("tuebingen.de", "Official City"),
    ("wikipedia.org", "Reference"),
    ("wikivoyage.org", "Reference"),
]

SCORE_LABELS = [
    ("BM25", "normalized_bm25"),
    ("Field Boost", "normalized_field_boost"),
    ("PRF", "normalized_prf"),
    ("LinkScore", "normalized_link"),
    ("LSA", "normalized_lsa"),
]


def add_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #0f172a;
            color: #e5e7eb;
        }
        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #f8fafc;
        }
        .hero {
            padding: 1.25rem 1.35rem;
            border: 1px solid #334155;
            border-radius: 14px;
            background: linear-gradient(135deg, #111827, #1e293b);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            color: #f8fafc;
            margin: 0;
        }
        .hero-subtitle {
            color: #cbd5e1;
            margin-top: .3rem;
        }
        .metric-card {
            border: 1px solid #334155;
            border-radius: 12px;
            background: #111827;
            padding: .85rem .95rem;
        }
        .metric-label {
            color: #94a3b8;
            font-size: .75rem;
            text-transform: uppercase;
            letter-spacing: .04em;
        }
        .metric-value {
            color: #f8fafc;
            font-size: 1.35rem;
            font-weight: 800;
        }
        div[data-testid="stTextInput"] div[data-baseweb="input"] {
            position: relative;
        }
        div[data-testid="stTextInput"] input {
            padding-right: 2.6rem;
        }
        div[data-testid="stTextInput"] div[data-baseweb="input"]::after {
            content: "";
            position: absolute;
            right: .95rem;
            top: 48%;
            width: .68rem;
            height: .68rem;
            border: 2px solid #93c5fd;
            border-radius: 999px;
            transform: translateY(-50%);
            pointer-events: none;
            opacity: .9;
        }
        div[data-testid="stTextInput"] div[data-baseweb="input"]::before {
            content: "";
            position: absolute;
            right: .78rem;
            top: 58%;
            width: .45rem;
            height: 2px;
            background: #93c5fd;
            border-radius: 999px;
            transform: rotate(45deg);
            pointer-events: none;
            opacity: .9;
            z-index: 1;
        }
        .result-card-marker {
            display: none;
        }
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .result-card-marker):has(> div[data-testid="stLayoutWrapper"] div[data-testid="stExpander"]) {
            border: 1px solid #334155;
            border-radius: 14px;
            background: #111827;
            padding: 1rem 1.05rem;
            margin: .8rem 0 1.15rem 0;
            box-shadow: 0 12px 30px rgba(0,0,0,.25);
        }
        .result-content {
            margin: 0;
        }
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .result-card-marker) div[data-testid="stButton"] button {
            border: 1px solid #60a5fa;
            border-radius: 999px;
            background: #172554;
            color: #dbeafe;
            font-size: .8rem;
            font-weight: 750;
            min-height: 2.15rem;
        }
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .result-card-marker) div[data-testid="stButton"] button::before {
            content: "✦";
            margin-right: .35rem;
            color: #fbbf24;
        }
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .result-card-marker) div[data-testid="stButton"] button:hover {
            border-color: #93c5fd;
            background: #1d4ed8;
            color: #f8fafc;
        }
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .result-card-marker) div[data-testid="stExpander"] {
            border: 1px solid #334155;
            border-radius: 12px;
            background: #0b1220;
            margin: .15rem 0 0 0;
        }
        .result-title {
            color: #f8fafc;
            font-size: 1.12rem;
            font-weight: 800;
            margin-bottom: .25rem;
        }
        .result-url {
            color: #93c5fd;
            font-size: .84rem;
            overflow-wrap: anywhere;
        }
        .summary {
            color: #dbeafe;
            background: #172554;
            border-left: 4px solid #60a5fa;
            border-radius: 10px;
            padding: .65rem .75rem;
            margin-top: .6rem;
        }
        .snippet {
            color: #cbd5e1;
            margin-top: .55rem;
            line-height: 1.48;
        }
        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: .16rem .5rem;
            margin: .25rem .28rem .15rem 0;
            font-size: .73rem;
            font-weight: 750;
            border: 1px solid #475569;
            color: #e5e7eb;
            background: #1e293b;
        }
        .score-badge {
            background: #064e3b;
            border-color: #10b981;
            color: #d1fae5;
        }
        .rank-badge {
            background: #1d4ed8;
            border-color: #60a5fa;
            color: #dbeafe;
        }
        mark {
            background: #fde68a;
            color: #78350f;
            border-radius: 4px;
            padding: 0 .12rem;
        }
        section[data-testid="stSidebar"] {
            background: #020617;
            border-right: 1px solid #1e293b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def esc(text: object) -> str:
    return html.escape(str(text or ""))


def doc_lookup(index: dict) -> dict[int, dict]:
    return {int(doc.get("doc_id", -1)): doc for doc in index.get("documents", [])}


def file_mtime(*parts: str) -> float:
    path = project_path(*parts)
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_index(index_mtime: float) -> dict:
    _ = index_mtime
    return read_json(project_path("data", "index.json"), {})


@st.cache_data(show_spinner=False)
def cached_retrieve(query: str, top_k: int, index_mtime: float) -> list[dict]:
    _ = index_mtime
    return retrieve(query, top_k=top_k)


@st.cache_data(show_spinner=False)
def cached_smart_summary(
    doc_id: int,
    query_text: str,
    title: str,
    snippet: str,
    body: str,
    prf_terms: tuple[str, ...],
    category: str,
) -> str:
    result = {"doc_id": doc_id, "title": title, "snippet": snippet}
    query_terms = preprocess(query_text, use_stemming=False)
    return smart_summary(result, {}, body, query_terms, list(prf_terms), category)


@st.cache_data(show_spinner=False)
def raw_body_lookup() -> dict[int, str]:
    raw_pages = read_json(project_path("data", "raw_pages.json"), {"pages": []})
    return {int(page.get("doc_id", -1)): page.get("body", "") for page in raw_pages.get("pages", [])}


def classify_category(result: dict, doc: dict, body: str = "") -> str:
    title = result.get("title") or doc.get("title", "")
    url = result.get("url") or doc.get("url", "")
    snippet = result.get("snippet") or doc.get("snippet", "")
    strong_text = f"{title} {url}".lower()
    full_text = f"{strong_text} {snippet} {body[:1000]}".lower()

    for category in ["Food and drinks", "Museums", "Events", "Transport"]:
        if any(term in strong_text for term in CATEGORY_RULES[category]):
            return category

    best_category = "General"
    best_score = 0
    for category, terms in CATEGORY_RULES.items():
        score = sum(3 for term in terms if term in strong_text)
        score += sum(1 for term in terms if term in full_text)
        if score > best_score:
            best_category = category
            best_score = score
    return best_category


def source_badge(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    for needle, label in SOURCE_RULES:
        if needle in host:
            return label
    return "External"


def split_sentences(text: str) -> list[str]:
    text = " ".join((text or "").split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 35 <= len(s.strip()) <= 340]


def smart_summary(result: dict, doc: dict, body: str, query_terms: list[str], prf_terms: list[str], category: str) -> str:
    body_text = (body or "")[:12000]
    sentences = split_sentences(body_text) or split_sentences(result.get("snippet", ""))
    if not sentences:
        return result.get("snippet", "")

    query_set = set(query_terms)
    prf_set = set(prf_terms)
    title_set = set(preprocess(result.get("title", ""), use_stemming=False))
    category_set = set(preprocess(category, use_stemming=False))

    scored = []
    for pos, sentence in enumerate(sentences[:80]):
        terms = set(preprocess(sentence, use_stemming=False))
        score = 3 * len(terms & query_set)
        score += 1.5 * len(terms & prf_set)
        score += 1.0 * len(terms & title_set)
        score += 0.8 * len(terms & category_set)
        if score:
            scored.append((score, -pos, sentence))

    if not scored:
        return result.get("snippet", "")
    best = sorted(scored, reverse=True)[:2]
    return " ".join(sentence for _score, _pos, sentence in sorted(best, key=lambda item: -item[1]))


def highlight(text: str, terms: list[str]) -> str:
    escaped = esc(text)
    for term in sorted({term for term in terms if len(term) > 2}, key=len, reverse=True):
        escaped = re.sub(re.escape(term), lambda m: f"<mark>{m.group(0)}</mark>", escaped, flags=re.IGNORECASE)
    return escaped


def why_reasons(result: dict, doc: dict, query_tokens: list[str], prf_terms: list[str]) -> list[str]:
    query_set = set(query_tokens)
    score_details = result.get("score_details", {})
    reasons = []

    title_hits = query_set & set(doc.get("title_tokens", []))
    heading_hits = query_set & set(doc.get("heading_tokens", []))
    body_hits = query_set & set(doc.get("body_tokens", []))
    url_hits = query_set & set(doc.get("url_tokens", []))

    if title_hits:
        reasons.append("Matched query term in title: " + ", ".join(sorted(title_hits)))
    if heading_hits:
        reasons.append("Matched query term in headings: " + ", ".join(sorted(heading_hits)))
    if url_hits:
        reasons.append("Matched query term in URL")
    if body_hits or result.get("matched_terms"):
        terms = sorted(body_hits or set(result.get("matched_terms", [])))
        reasons.append("Matched terms in body: " + ", ".join(terms[:6]))
    if prf_terms and score_details.get("normalized_prf", 0) > 0:
        reasons.append("Boosted by PRF terms: " + ", ".join(prf_terms[:5]))
    if score_details.get("normalized_link", 0) > 0:
        reasons.append("Internal LinkScore contributed")
    if score_details.get("normalized_lsa", 0) > 0:
        reasons.append("Semantic LSA similarity contributed")

    return reasons or ["BM25 found textual overlap with the query."]


def score_value(score_details: dict, key: str) -> float:
    return max(0.0, min(1.0, float(score_details.get(key, 0.0) or 0.0)))


def show_score_bars(score_details: dict) -> None:
    for label, key in SCORE_LABELS:
        value = score_value(score_details, key)
        cols = st.columns([1.1, 4, 0.7])
        cols[0].caption(label)
        cols[1].progress(value)
        cols[2].caption(f"{value:.2f}")


def format_runtime(seconds: float) -> str:
    if seconds <= 0:
        return "<0.001s"
    if seconds < 0.001:
        return "<0.001s"
    return f"{seconds:.3f}s"


def metric_cards(runtime: float, indexed: int, shown: int) -> None:
    cols = st.columns(3)
    values = [("Search time", format_runtime(runtime)), ("Indexed pages", indexed), ("Shown results", shown)]
    for col, (label, value) in zip(cols, values):
        col.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{esc(label)}</div>
              <div class="metric-value">{esc(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_card(result: dict, doc: dict, body: str, query_terms: list[str], query_tokens: list[str]) -> None:
    prf_terms = result.get("expansion_terms", [])
    category = classify_category(result, doc, body)
    source = source_badge(result.get("url", ""))
    terms_to_mark = query_terms + prf_terms + result.get("matched_terms", [])
    summary_key = f"summary_{int(result.get('doc_id', -1))}_{int(result.get('rank', 0))}"
    if summary_key not in st.session_state:
        st.session_state[summary_key] = False

    with st.container(border=True):
        st.markdown('<div class="result-card-marker"></div>', unsafe_allow_html=True)

        left, right = st.columns([5.5, 1.25])
        with left:
            st.markdown(
                f"""
                <div class="result-content">
                  <div class="result-title">
                    <span class="badge rank-badge">#{int(result.get("rank", 0))}</span>
                    {esc(result.get("title") or result.get("url"))}
                  </div>
                  <a class="result-url" href="{esc(result.get("url", ""))}" target="_blank">{esc(result.get("url", ""))}</a><br>
                  <span class="badge">Category: {esc(category)}</span>
                  <span class="badge">Source: {esc(source)}</span>
                  <span class="badge score-badge">Score {float(result.get("score", 0.0) or 0.0):.3f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            if st.button("AI Summary", key=f"button_{summary_key}", help="Show or hide the local extractive summary", use_container_width=True):
                st.session_state[summary_key] = not st.session_state[summary_key]

        if st.session_state[summary_key]:
            summary = cached_smart_summary(
                int(result.get("doc_id", -1)),
                " ".join(query_terms),
                result.get("title") or "",
                result.get("snippet", ""),
                body,
                tuple(prf_terms),
                category,
            )
            st.markdown(
                f"""
                <div class="summary"><strong>Smart Summary:</strong><br>{highlight(summary, terms_to_mark)}</div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="snippet">{highlight(result.get("snippet", ""), terms_to_mark)}</div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Why this result? Score details and ranking signals"):
            st.write("Why this result?")
            for reason in why_reasons(result, doc, query_tokens, prf_terms):
                st.markdown(f"- {reason}")
            st.write("Score breakdown")
            show_score_bars(result.get("score_details", {}))
            if prf_terms:
                st.caption("PRF terms: " + ", ".join(prf_terms))


def main() -> None:
    st.set_page_config(page_title="Tuebingen Search Engine", layout="wide")
    add_css()

    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">Tuebingen Search Engine</div>
          <div class="hero-subtitle">Explainable local search over English Tuebingen-related pages</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    index_mtime = file_mtime("data", "index.json")
    index = load_index(index_mtime)
    documents = index.get("documents", [])
    docs = doc_lookup(index)
    bodies = raw_body_lookup()

    categories = ["All"] + sorted(
        {classify_category({"title": doc.get("title"), "url": doc.get("url"), "snippet": doc.get("snippet")}, doc) for doc in documents}
    )

    with st.sidebar:
        st.header("Search")
        selected_category = st.selectbox("Category", categories)
        top_k = st.slider("Results", 5, 50, 10)

    query = st.text_input("Search", value="tuebingen attractions", placeholder="Try: food and drinks")
    if not documents:
        st.warning("No index found yet. Run `python scripts/build_index.py` first.")
        return
    if not query:
        st.info("Enter a query to search the local JSON index.")
        return

    start = time.perf_counter()
    results = cached_retrieve(query, top_k, index_mtime)
    runtime = time.perf_counter() - start

    query_terms = preprocess(query, use_stemming=False)
    query_tokens = preprocess(query)
    prf_terms = results[0].get("expansion_terms", []) if results else []

    filtered = []
    for result in results:
        doc = docs.get(int(result.get("doc_id", -1)), {})
        body = bodies.get(int(result.get("doc_id", -1)), "")
        category = classify_category(result, doc, body)
        if selected_category == "All" or category == selected_category:
            filtered.append((result, doc, body))

    with st.sidebar:
        st.divider()
        st.caption(f"Original query: {query}")
        st.caption("PRF expansion terms")
        st.write(", ".join(prf_terms) if prf_terms else "None")
        st.caption(f"Indexed pages: {len(documents)}")
        st.caption(f"Search time: {format_runtime(runtime)}")
        st.caption(f"Shown results: {len(filtered)}")
        st.caption(f"Active filter: {selected_category}")

    metric_cards(runtime, len(documents), len(filtered))

    if not filtered:
        st.warning("No results found for this query/filter.")
        return

    for result, doc, body in filtered:
        render_card(result, doc, body, query_terms, query_tokens)


if __name__ == "__main__":
    main()
