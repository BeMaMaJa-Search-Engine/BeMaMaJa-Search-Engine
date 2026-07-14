from __future__ import annotations

import csv
import hashlib
import html
import io
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.batch import format_result_rows, run_single_batch_query
from src.preprocessing import preprocess
from src.reranking import rerank
from src.retrieval import retrieve
from src.utils import project_path
from llm_summary import generate_llm_summary, gemini_is_configured
from loading_animation import water_cooling_loader
from storage_data import INDEX_OBJECT, RAW_PAGES_OBJECT, StorageDataError, load_json_source

SCORE_LABELS = [
    ("BM25", "normalized_bm25"),
    ("Field Boost", "normalized_field_boost"),
    ("PRF", "normalized_prf"),
    ("LinkScore", "normalized_link"),
    ("LSA", "normalized_lsa"),
]

AI_MODE_OPTIONS = {
    "Summary + relevance": "relevance",
    "Summary only": "summary",
    "Custom focus": "custom",
}


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
            position: relative;
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
            width: fit-content;
        }
        .metric-source {
            color: #64748b;
            font-size: .6rem;
            position: absolute;
            right: .55rem;
            bottom: .65rem;
        }
        div[data-testid="stButton"] button p {
            white-space: nowrap;
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
        .badge-row {
            display: flex;
            align-items: center;
            flex-wrap: nowrap;
            gap: .28rem;
        }
        .badge-row .badge {
            margin: .25rem 0 .15rem 0;
            padding-left: .4rem;
            padding-right: .4rem;
            white-space: nowrap;
        }
        @media (max-width: 700px) {
            .badge-row {
                flex-wrap: wrap;
            }
        }
        .score-badge {
            background: #064e3b;
            border-color: #10b981;
            color: #d1fae5;
        }
        .term-found {
            background: #064e3b;
            border-color: #10b981;
            color: #d1fae5;
        }
        .term-missing {
            background: #4c0519;
            border-color: #fb7185;
            color: #ffe4e6;
        }
        .search-correction {
            margin: .35rem 0 .7rem 0;
            padding: .48rem .7rem;
            border: 1px solid #3b82f6;
            border-radius: 8px;
            background: #172554;
            color: #dbeafe;
            font-size: .85rem;
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
        .sidebar-section {
            border-top: 1px solid #1e293b;
            margin-top: .9rem;
            padding-top: .85rem;
        }
        .sidebar-section-title {
            color: #cbd5e1;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .05em;
            text-transform: uppercase;
            margin-bottom: .55rem;
        }
        .sidebar-item {
            margin-bottom: .65rem;
        }
        .sidebar-label {
            color: #94a3b8;
            font-size: .72rem;
        }
        .sidebar-value {
            color: #f8fafc;
            font-size: .84rem;
            font-weight: 650;
            margin-top: .08rem;
            overflow-wrap: anywhere;
        }
        .sidebar-status-ready {
            color: #d1fae5;
        }
        .sidebar-status-missing {
            color: #fda4af;
        }
        .sidebar-note {
            color: #94a3b8;
            font-size: .7rem;
            line-height: 1.4;
            margin-top: .18rem;
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


@st.cache_resource(show_spinner=False)
def load_index(index_mtime: float) -> tuple[dict, str, str]:
    _ = index_mtime
    try:
        return load_json_source(project_path("data", "index.json"), INDEX_OBJECT, st.secrets)
    except StorageDataError as exc:
        return {}, "unavailable", str(exc)


@st.cache_data(show_spinner=False)
def cached_retrieve(query: str, top_k: int, index_mtime: float) -> dict:
    start = time.perf_counter()
    index = load_index(index_mtime)[0]
    first_stage = retrieve(query, index, top_k=top_k)
    reranked = rerank(first_stage, index)
    return {
        "results": normalize_results(reranked),
        "query_tokens": first_stage.get("query_tokens", []),
        "runtime": time.perf_counter() - start,
    }


def normalize_results(response: object) -> list[dict]:
    """Adapt the team's retrieval output to the UI result-card format."""
    if isinstance(response, list):
        candidates = response
    elif isinstance(response, dict):
        candidates = response.get("results") or response.get("candidates") or []
    else:
        candidates = []

    results: list[dict] = []
    for rank, item in enumerate(candidates, start=1):
        score = float(item.get("score", item.get("bm25_score", 0.0)) or 0.0)
        score_details = dict(item.get("score_details", {}))

        result = dict(item)
        result["rank"] = int(item.get("rank", rank) or rank)
        result["score"] = score
        result["score_details"] = score_details
        result["matched_terms"] = list(dict.fromkeys(item.get("matched_terms", [])))
        result.setdefault("expansion_terms", [])
        results.append(result)

    return results


def parse_batch_queries(file_bytes: bytes) -> list[tuple[str, str]]:
    """Read query_id and query text from an uploaded UTF-8 TSV file."""
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("The query file must use UTF-8 encoding.") from error

    queries = []
    for line_number, row in enumerate(csv.reader(io.StringIO(text), delimiter="\t"), start=1):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != 2:
            raise ValueError(f"Line {line_number} must contain query_id, TAB, and query text.")

        query_id, query = (value.strip() for value in row)
        if not query_id or not query:
            raise ValueError(f"Line {line_number} contains an empty query ID or query.")
        queries.append((query_id, query))

    if not queries:
        raise ValueError("The query file does not contain any queries.")
    return queries


def build_batch_output(file_bytes: bytes, index: dict) -> dict:
    """Run the team batch pipeline and return the official TSV in memory."""
    queries = parse_batch_queries(file_bytes)
    rows = []
    start = time.perf_counter()

    for query_id, query in queries:
        query_result = run_single_batch_query(query_id, query, index, top_k=100)
        rows.extend(format_result_rows(query_id, query_result["results"], top_k=100))

    output = io.StringIO()
    csv.writer(output, delimiter="\t", lineterminator="\n").writerows(rows)
    return {
        "data": output.getvalue(),
        "query_count": len(queries),
        "result_count": len(rows),
        "runtime": time.perf_counter() - start,
    }


@st.cache_data(show_spinner=False)
def cached_smart_summary(
    doc_id: int,
    query_text: str,
    title: str,
    snippet: str,
    body: str,
    prf_terms: tuple[str, ...],
) -> str:
    result = {"doc_id": doc_id, "title": title, "snippet": snippet}
    query_terms = preprocess(query_text, use_stemming=False)
    return smart_summary(result, body, query_terms, list(prf_terms))


@st.cache_resource(show_spinner=False)
def raw_body_lookup(raw_pages_mtime: float) -> tuple[dict[int, str], str, str]:
    _ = raw_pages_mtime
    local_path = project_path("data", "raw_pages.json")
    storage_secrets = {} if local_path.exists() else st.secrets
    try:
        raw_pages, source, warning = load_json_source(
            local_path,
            RAW_PAGES_OBJECT,
            storage_secrets,
        )
    except StorageDataError as exc:
        return {}, "unavailable", str(exc)
    bodies = {int(page.get("doc_id", -1)): page.get("body", "") for page in raw_pages.get("pages", [])}
    return bodies, source, warning


def split_sentences(text: str) -> list[str]:
    text = " ".join((text or "").split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 35 <= len(s.strip()) <= 340]


def smart_summary(result: dict, body: str, query_terms: list[str], prf_terms: list[str]) -> str:
    body_text = (body or "")[:12000]
    sentences = split_sentences(body_text) or split_sentences(result.get("snippet", ""))
    if not sentences:
        return result.get("snippet", "")

    query_set = set(query_terms)
    prf_set = set(prf_terms)
    title_set = set(preprocess(result.get("title", ""), use_stemming=False))

    scored = []
    for pos, sentence in enumerate(sentences[:80]):
        terms = set(preprocess(sentence, use_stemming=False))
        score = 3 * len(terms & query_set)
        score += 1.5 * len(terms & prf_set)
        score += 1.0 * len(terms & title_set)
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


def direct_query_term_matches(
    result: dict,
    doc: dict,
    body: str,
    query_terms: list[str],
) -> tuple[list[str], list[str]]:
    page_text = " ".join(
        [
            result.get("title") or doc.get("title", ""),
            unquote(result.get("url") or doc.get("url", "")),
            result.get("snippet") or doc.get("snippet", ""),
            body,
        ]
    ).lower()
    page_text = page_text.replace("t\u00fcbingen", "tubingen").replace("tuebingen", "tubingen")

    found = []
    not_found = []
    for term in dict.fromkeys(query_terms):
        pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
        (found if re.search(pattern, page_text) else not_found).append(term)
    return found, not_found


def spelling_corrections(
    query_terms: list[str],
    query_tokens: list[str],
    corrected_query_tokens: list[str],
) -> list[tuple[str, str]]:
    corrections = []
    for index, (original_token, corrected_token) in enumerate(zip(query_tokens, corrected_query_tokens)):
        if original_token == corrected_token:
            continue
        entered_term = query_terms[index] if index < len(query_terms) else original_token
        corrections.append((entered_term, corrected_token))
    return corrections


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
        reasons.append("Matched indexed terms: " + ", ".join(terms[:6]))
    if score_details.get("normalized_field_boost", 0) > 0:
        reasons.append("Title or heading Field Boost contributed")
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
    shown_components = 0
    for label, key in SCORE_LABELS:
        if key not in score_details:
            continue
        value = score_value(score_details, key)
        cols = st.columns([1.1, 4, 0.7])
        cols[0].caption(label)
        cols[1].progress(value)
        cols[2].caption(f"{value:.2f}")
        shown_components += 1

    if not shown_components:
        st.caption("No ranking component details were returned by the backend.")

def format_runtime(seconds: float) -> str:
    if seconds <= 0:
        return "<0.001s"
    if seconds < 0.001:
        return "<0.001s"
    return f"{seconds:.3f}s"


def metric_cards(runtime: float, indexed: int, shown: int, index_source: str) -> None:
    cols = st.columns(3)
    source_label = "Supabase" if index_source == "Supabase Storage" else "Local"
    values = [
        ("Search time", format_runtime(runtime), ""),
        ("Indexed pages", indexed, source_label),
        ("Shown results", shown, ""),
    ]
    for col, (label, value, source) in zip(cols, values):
        col.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{esc(label)}</div>
              <div class="metric-value">{esc(value)}</div>
              {f'<div class="metric-source">{esc(source)}</div>' if source else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_card(
    result: dict,
    doc: dict,
    body: str,
    raw_pages_mtime: float,
    query_text: str,
    query_terms: list[str],
    corrected_query_tokens: list[str],
    ai_mode: str,
    custom_instruction: str,
) -> None:
    prf_terms = result.get("expansion_terms", [])
    found_terms, not_found_terms = direct_query_term_matches(result, doc, body, query_terms)
    terms_to_mark = query_terms + prf_terms + result.get("matched_terms", [])
    query_hash = hashlib.sha1(query_text.encode("utf-8")).hexdigest()[:10]
    summary_key = f"summary_{int(result.get('doc_id', -1))}_{query_hash}"
    active_settings_key = f"active_{summary_key}"
    settings_hash = hashlib.sha1(f"{ai_mode}:{custom_instruction}".encode("utf-8")).hexdigest()[:10]
    requested_llm_key = f"llm_{summary_key}_{settings_hash}"
    storage_warning_key = f"storage_warning_{summary_key}_{settings_hash}"
    active_settings_hash = st.session_state.get(active_settings_key)
    settings_changed = bool(active_settings_hash and active_settings_hash != settings_hash)
    needs_summary = not active_settings_hash or settings_changed
    custom_instruction_missing = ai_mode == "custom" and not custom_instruction.strip()

    if summary_key not in st.session_state:
        st.session_state[summary_key] = False

    if settings_changed:
        button_label = "Update Summary"
        button_icon = ":material/refresh:"
        button_help = "Generate or load a summary with the current AI settings"
    elif st.session_state[summary_key]:
        button_label = "Hide Summary"
        button_icon = ":material/auto_awesome:"
        button_help = "Hide the current AI summary"
    else:
        button_label = "AI Summary"
        button_icon = ":material/auto_awesome:"
        button_help = "Generate or show a Gemini summary"

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
                  <div class="badge-row">
                    <span class="badge term-found">Found: {esc(", ".join(found_terms) or "none")}</span>
                    <span class="badge term-missing">Not found: {esc(", ".join(not_found_terms) or "none")}</span>
                    <span class="badge score-badge">Score {float(result.get("score", 0.0) or 0.0):.3f}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            summary_clicked = st.button(
                button_label,
                key=f"button_{summary_key}",
                help=button_help,
                icon=button_icon,
                disabled=needs_summary and custom_instruction_missing,
                use_container_width=True,
            )
            cooling_placeholder = st.empty()

        if summary_clicked:
            if needs_summary:
                if requested_llm_key not in st.session_state:
                    summary_body = body
                    if not summary_body:
                        raw_bodies, _raw_source, raw_warning = raw_body_lookup(raw_pages_mtime)
                        summary_body = raw_bodies.get(int(result.get("doc_id", -1)), "")
                        if raw_warning:
                            st.session_state[storage_warning_key] = raw_warning
                    with water_cooling_loader(cooling_placeholder):
                        st.session_state[requested_llm_key] = generate_llm_summary(
                            result=result,
                            doc=doc,
                            body=summary_body,
                            query=query_text,
                            secrets=st.secrets,
                            mode=ai_mode,
                            custom_instruction=custom_instruction,
                        )
                st.session_state[active_settings_key] = settings_hash
                st.session_state[summary_key] = True
                active_settings_hash = settings_hash
            else:
                st.session_state[summary_key] = not st.session_state[summary_key]
            st.rerun()

        if st.session_state[summary_key] and active_settings_hash:
            active_llm_key = f"llm_{summary_key}_{active_settings_hash}"
            llm_summary = st.session_state[active_llm_key]
            summary = llm_summary.text
            summary_label = "AI Summary"
            if not summary:
                fallback_body = body
                if not fallback_body:
                    fallback_body = raw_body_lookup(raw_pages_mtime)[0].get(
                        int(result.get("doc_id", -1)),
                        "",
                    )
                summary = cached_smart_summary(
                    int(result.get("doc_id", -1)),
                    query_text,
                    result.get("title") or "",
                    result.get("snippet", ""),
                    fallback_body,
                    tuple(prf_terms),
                )
                summary_label = "Local Smart Summary"

            st.markdown(
                f"""
                <div class="summary"><strong>{esc(summary_label)}:</strong><br>{highlight(summary, terms_to_mark)}</div>
                """,
                unsafe_allow_html=True,
            )
            if llm_summary.error:
                st.caption(f"{llm_summary.error} Showing local fallback summary.")
            elif llm_summary.source:
                st.caption(f"Summary source: {llm_summary.source}")
            if st.session_state.get(storage_warning_key):
                st.caption(st.session_state[storage_warning_key])

        st.markdown(
            f"""
            <div class="snippet">{highlight(result.get("snippet", ""), terms_to_mark)}</div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Why this result? Score details and ranking signals"):
            st.write("Why this result?")
            for reason in why_reasons(result, doc, corrected_query_tokens, prf_terms):
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
    raw_pages_mtime = file_mtime("data", "raw_pages.json")
    index, index_source, index_warning = load_index(index_mtime)
    documents = index.get("documents", [])
    docs = doc_lookup(index)

    if index_warning:
        st.warning(index_warning)

    with st.sidebar:
        st.header("Search")
        top_k = st.slider("Results", 5, 50, 10)
        with st.expander("AI Settings"):
            ai_mode_label = st.selectbox("AI response", list(AI_MODE_OPTIONS))
            ai_mode = AI_MODE_OPTIONS[ai_mode_label]
            custom_instruction = ""
            if ai_mode == "custom":
                custom_instruction = st.text_area(
                    "What should the summary focus on?",
                    placeholder="Example: Focus on opening hours and visitor information.",
                    max_chars=300,
                )
                if not custom_instruction.strip():
                    st.caption("Add a focus instruction. Until then, the standard relevance mode is used.")

        with st.expander("Batch Search"):
            st.caption("Upload UTF-8 TSV: query_id, TAB, query text.")
            batch_file = st.file_uploader(
                "Upload queries.tsv",
                type=["tsv"],
                label_visibility="collapsed",
            )
            if batch_file is not None:
                batch_bytes = batch_file.getvalue()
                batch_hash = hashlib.sha1(batch_bytes).hexdigest()

                if st.button("Run Batch Search", type="primary", use_container_width=True):
                    st.session_state.pop("batch_output", None)
                    try:
                        with st.spinner("Running batch queries..."):
                            batch_output = build_batch_output(batch_bytes, index)
                        batch_output["source_hash"] = batch_hash
                        st.session_state["batch_output"] = batch_output
                    except ValueError as error:
                        st.error(str(error))
                    except Exception as error:
                        st.error(f"Batch search failed ({type(error).__name__}).")

                batch_output = st.session_state.get("batch_output", {})
                if batch_output.get("source_hash") == batch_hash:
                    st.caption(
                        f"{batch_output['query_count']} queries | "
                        f"{batch_output['result_count']} results | "
                        f"{batch_output['runtime']:.2f}s"
                    )
                    st.download_button(
                        "Download results.tsv",
                        data=batch_output["data"],
                        file_name="results.tsv",
                        mime="text/tab-separated-values",
                        use_container_width=True,
                    )

    query = st.text_input("Search", value="tuebingen attractions", placeholder="Try: food and drinks")
    if not documents:
        st.warning("No index found yet. Run `python scripts/build_index.py` first.")
        return
    if not query:
        st.info("Enter a query to search the local JSON index.")
        return

    retrieval_output = cached_retrieve(query, top_k, index_mtime)
    runtime = float(retrieval_output.get("runtime", 0.0))

    results = retrieval_output.get("results", [])
    query_terms = preprocess(query, use_stemming=False)
    query_tokens = preprocess(query)
    corrected_query_tokens = retrieval_output.get("query_tokens", query_tokens)
    corrections = spelling_corrections(query_terms, query_tokens, corrected_query_tokens)
    prf_terms = results[0].get("expansion_terms", []) if results else []

    if corrections:
        correction_text = ", ".join(
            f"{esc(original)} &rarr; {esc(corrected)}" for original, corrected in corrections
        )
        st.markdown(
            f'<div class="search-correction"><strong>Spelling corrected:</strong> '
            f'Searching for {correction_text}</div>',
            unsafe_allow_html=True,
        )

    raw_bodies, _raw_source, raw_warning = raw_body_lookup(raw_pages_mtime)
    if raw_warning:
        st.warning(f"{raw_warning} Found-term badges are using title, URL, and snippet only.")

    prepared_results = []
    for result in results:
        doc = docs.get(int(result.get("doc_id", -1)), {})
        body = raw_bodies.get(int(result.get("doc_id", -1)), "")
        prepared_results.append((result, doc, body))

    with st.sidebar:
        gemini_ready = gemini_is_configured(st.secrets)
        gemini_status = "Ready" if gemini_ready else "Not configured"
        gemini_status_class = "sidebar-status-ready" if gemini_ready else "sidebar-status-missing"
        active_score_keys = {
            key
            for result in results
            for key in result.get("score_details", {})
        }
        active_ranking_labels = [
            label for label, key in SCORE_LABELS if key in active_score_keys
        ]
        active_ranking_text = " + ".join(active_ranking_labels) or "No ranking details"
        st.markdown(
            f"""
            <div class="sidebar-section">
              <div class="sidebar-section-title">Search overview</div>
              <div class="sidebar-item">
                <div class="sidebar-label">Original query</div>
                <div class="sidebar-value">{esc(query)}</div>
              </div>
              <div class="sidebar-item">
                <div class="sidebar-label">Indexed pages</div>
                <div class="sidebar-value">{len(documents)}</div>
              </div>
              <div class="sidebar-item">
                <div class="sidebar-label">Search time</div>
                <div class="sidebar-value">{esc(format_runtime(runtime))}</div>
              </div>
              <div class="sidebar-item">
                <div class="sidebar-label">Shown results</div>
                <div class="sidebar-value">{len(prepared_results)}</div>
              </div>
            </div>
            <div class="sidebar-section">
              <div class="sidebar-section-title">AI configuration</div>
              <div class="sidebar-item">
                <div class="sidebar-label">Gemini summary</div>
                <div class="sidebar-value {gemini_status_class}">{esc(gemini_status)}</div>
              </div>
              <div class="sidebar-item">
                <div class="sidebar-label">Response mode</div>
                <div class="sidebar-value">{esc(ai_mode_label)}</div>
              </div>
            </div>
            <div class="sidebar-section">
              <div class="sidebar-section-title">Ranking</div>
              <div class="sidebar-item">
                <div class="sidebar-label">Active ranking</div>
                <div class="sidebar-value">{esc(active_ranking_text)}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_cards(runtime, len(documents), len(prepared_results), index_source)

    if not prepared_results:
        st.warning("No results found for this query.")
        return

    for result, doc, body in prepared_results:
        render_card(
            result,
            doc,
            body,
            raw_pages_mtime,
            query,
            query_terms,
            corrected_query_tokens,
            ai_mode,
            custom_instruction,
        )


if __name__ == "__main__":
    main()
