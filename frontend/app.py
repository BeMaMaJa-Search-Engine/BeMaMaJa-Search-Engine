from __future__ import annotations

import hashlib
import html
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess
from src.reranking import rerank
from src.retrieval import retrieve
from src.utils import project_path, read_json
from llm_summary import generate_llm_summary, gemini_is_configured

SCORE_LABELS = [
    ("BM25", "normalized_bm25"),
    ("Field Boost", "normalized_field_boost"),
    ("PRF", "normalized_prf"),
    ("LinkScore", "normalized_link"),
    ("LSA", "normalized_lsa"),
]

# TODO(team): Remove entries as soon as the backend returns these score components.
PENDING_RANKING_SIGNALS = {
    "normalized_prf": "PRF",
    "normalized_link": "LinkScore",
    "normalized_lsa": "LSA",
}

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
def cached_retrieve(query: str, top_k: int, index_mtime: float) -> dict:
    _ = index_mtime
    index = read_json(project_path("data", "index.json"), {})
    first_stage = retrieve(query, index, top_k=top_k)
    reranked = rerank(first_stage, index)
    return {
        "results": normalize_results(reranked),
        "query_tokens": first_stage.get("query_tokens", []),
    }


def normalize_results(response: object) -> list[dict]:
    """Adapt the team's retrieval output to the UI result-card format."""
    if isinstance(response, list):
        candidates = response
    elif isinstance(response, dict):
        candidates = response.get("results") or response.get("candidates") or []
    else:
        candidates = []

    raw_scores = [float(item.get("score", item.get("bm25_score", 0.0)) or 0.0) for item in candidates]
    max_score = max(raw_scores, default=0.0)

    results: list[dict] = []
    for rank, item in enumerate(candidates, start=1):
        score = float(item.get("score", item.get("bm25_score", 0.0)) or 0.0)
        score_details = dict(item.get("score_details", {}))
        score_details.setdefault("normalized_bm25", score / max_score if max_score > 0 else 0.0)

        result = dict(item)
        result["rank"] = int(item.get("rank", rank) or rank)
        result["score"] = score
        result["score_details"] = score_details
        result["matched_terms"] = list(dict.fromkeys(item.get("matched_terms", [])))
        result.setdefault("expansion_terms", [])
        results.append(result)

    return results


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


@st.cache_data(show_spinner=False)
def raw_body_lookup() -> dict[int, str]:
    raw_pages = read_json(project_path("data", "raw_pages.json"), {"pages": []})
    return {int(page.get("doc_id", -1)): page.get("body", "") for page in raw_pages.get("pages", [])}


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

    pending = [label for key, label in PENDING_RANKING_SIGNALS.items() if key not in score_details]
    if pending:
        st.caption("Pending backend signals: " + ", ".join(pending))


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


def render_card(
    result: dict,
    doc: dict,
    body: str,
    query_text: str,
    query_terms: list[str],
    query_tokens: list[str],
    ai_mode: str,
    custom_instruction: str,
) -> None:
    prf_terms = result.get("expansion_terms", [])
    found_terms, not_found_terms = direct_query_term_matches(result, doc, body, query_terms)
    terms_to_mark = query_terms + prf_terms + result.get("matched_terms", [])
    summary_key = f"summary_{int(result.get('doc_id', -1))}_{int(result.get('rank', 0))}"
    active_settings_key = f"active_{summary_key}"
    settings_hash = hashlib.sha1(f"{ai_mode}:{custom_instruction}".encode("utf-8")).hexdigest()[:10]
    requested_llm_key = f"llm_{summary_key}_{settings_hash}"
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
                  <span class="badge term-found">Found: {esc(", ".join(found_terms) or "none")}</span>
                  <span class="badge term-missing">Not found: {esc(", ".join(not_found_terms) or "none")}</span>
                  <span class="badge score-badge">Score {float(result.get("score", 0.0) or 0.0):.3f}</span>
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

        if summary_clicked:
            if needs_summary:
                if requested_llm_key not in st.session_state:
                    with st.spinner("Generating AI summary..."):
                        st.session_state[requested_llm_key] = generate_llm_summary(
                            result=result,
                            doc=doc,
                            body=body,
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

        if st.session_state[summary_key] and active_settings_hash:
            active_llm_key = f"llm_{summary_key}_{active_settings_hash}"
            llm_summary = st.session_state[active_llm_key]
            summary = llm_summary.text
            summary_label = "AI Summary"
            if not summary:
                summary = cached_smart_summary(
                    int(result.get("doc_id", -1)),
                    query_text,
                    result.get("title") or "",
                    result.get("snippet", ""),
                    body,
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

    query = st.text_input("Search", value="tuebingen attractions", placeholder="Try: food and drinks")
    if not documents:
        st.warning("No index found yet. Run `python scripts/build_index.py` first.")
        return
    if not query:
        st.info("Enter a query to search the local JSON index.")
        return

    start = time.perf_counter()
    retrieval_output = cached_retrieve(query, top_k, index_mtime)
    runtime = time.perf_counter() - start

    results = retrieval_output.get("results", [])
    query_terms = preprocess(query, use_stemming=False)
    query_tokens = preprocess(query)
    corrected_query_tokens = retrieval_output.get("query_tokens", query_tokens)
    prf_terms = results[0].get("expansion_terms", []) if results else []

    prepared_results = []
    for result in results:
        doc = docs.get(int(result.get("doc_id", -1)), {})
        body = bodies.get(int(result.get("doc_id", -1)), "")
        prepared_results.append((result, doc, body))

    with st.sidebar:
        st.divider()
        st.caption(f"Original query: {query}")
        st.caption("Gemini AI summary")
        st.write("Configured" if gemini_is_configured(st.secrets) else "Not configured")
        st.caption(f"AI mode: {ai_mode_label}")
        st.caption("Active ranking")
        st.write("BM25 + Field Boost")
        if corrected_query_tokens != query_tokens:
            st.caption("Corrected search tokens")
            st.write(" ".join(corrected_query_tokens))
        if prf_terms:
            st.caption("PRF expansion terms")
            st.write(", ".join(prf_terms))
        else:
            st.caption("Future backend signals")
            st.write("PRF, LinkScore, and LSA are pending.")
        st.caption(f"Indexed pages: {len(documents)}")
        st.caption(f"Search time: {format_runtime(runtime)}")
        st.caption(f"Shown results: {len(prepared_results)}")

    metric_cards(runtime, len(documents), len(prepared_results))

    if not prepared_results:
        st.warning("No results found for this query.")
        return

    for result, doc, body in prepared_results:
        render_card(
            result,
            doc,
            body,
            query,
            query_terms,
            query_tokens,
            ai_mode,
            custom_instruction,
        )


if __name__ == "__main__":
    main()
