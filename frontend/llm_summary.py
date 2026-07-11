from __future__ import annotations

import os
import re
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup


DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
PLACEHOLDER_KEY = "paste_your_gemini_api_key_here"
MAX_CONTEXT_CHARS = 24000


@dataclass
class LlmSummary:
    text: str
    source: str
    error: str = ""


def read_gemini_settings(secrets: object | None = None) -> tuple[str, str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()

    if secrets is not None:
        try:
            api_key = api_key or str(secrets.get("GEMINI_API_KEY", "")).strip()
            model = model or str(secrets.get("GEMINI_MODEL", "")).strip()
        except Exception:
            pass

    if api_key == PLACEHOLDER_KEY:
        api_key = ""
    return api_key, model or DEFAULT_GEMINI_MODEL


def gemini_is_configured(secrets: object | None = None) -> bool:
    api_key, _model = read_gemini_settings(secrets)
    return bool(api_key)


def fetch_page_text(url: str, timeout: float = 8.0) -> str:
    if not url.startswith(("http://", "https://")):
        return ""

    headers = {"User-Agent": "MSE-Tuebingen-Search-UI/1.0"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()
    return clean_text(soup.get_text(" "))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def sentence_split(text: str) -> list[str]:
    text = clean_text(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]


def compact_context(text: str, query: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text

    query_terms = {term.lower() for term in re.findall(r"[A-Za-zÄÖÜäöüß]+", query) if len(term) > 2}
    keep_start = text[: max_chars // 2]
    remaining_budget = max_chars - len(keep_start) - 80
    selected = []

    for sentence in sentence_split(text[max_chars // 2 :]):
        lowered = sentence.lower()
        if any(term in lowered for term in query_terms):
            selected.append(sentence)
            if sum(len(s) + 1 for s in selected) >= remaining_budget:
                break

    suffix = " ".join(selected)
    if not suffix:
        suffix = text[-remaining_budget:]
    return clean_text(keep_start + " ... " + suffix[:remaining_budget])


def build_page_context(result: dict, doc: dict, body: str, query: str) -> tuple[str, str]:
    url = result.get("url") or doc.get("url", "")
    title = result.get("title") or doc.get("title", "")
    snippet = result.get("snippet") or doc.get("snippet", "")

    candidates = [
        body,
        doc.get("body", ""),
        result.get("body", ""),
        doc.get("text", ""),
        result.get("text", ""),
        snippet,
    ]
    page_text = clean_text(next((candidate for candidate in candidates if clean_text(candidate)), ""))
    source = "crawled page text" if page_text and page_text != clean_text(snippet) else "snippet"

    if len(page_text) < 600 and url:
        try:
            fetched = fetch_page_text(url)
            if len(fetched) > len(page_text):
                page_text = fetched
                source = "live fetched page text"
        except Exception:
            pass

    context = f"Title: {title}\nURL: {url}\nSnippet: {snippet}\nPage text: {compact_context(page_text, query)}"
    return context, source


def build_prompt(query: str, context: str) -> str:
    return f"""
You are writing a short explanation for a local Tuebingen search engine.

User query: {query}

Use only the provided page text. Do not invent facts.
Write 2-3 concise sentences in English.
Focus on why this page is relevant to the query and what useful information it contains.
If the page text is weak or unrelated, say that it only has limited relevance.

Provided page text:
{context}
""".strip()


def build_summary_only_prompt(context: str) -> str:
    return f"""
You are writing a short factual summary of a web page.

Use only the provided page text. Do not invent facts.
Write 2-3 concise sentences in English.
Summarize the page's main content without judging its relevance to a search query.

Provided page text:
{context}
""".strip()


def build_custom_prompt(context: str, custom_instruction: str) -> str:
    return f"""
You are writing a short factual summary of a web page.

User's requested focus: {custom_instruction}

Use only the provided page text. Do not invent facts.
Write 2-3 concise sentences in English.
Focus on the user's requested aspect. If the page does not contain that information, say so briefly.

Provided page text:
{context}
""".strip()


def route_prompt(
    mode: str,
    query: str,
    context: str,
    custom_instruction: str = "",
) -> str:
    if mode == "summary":
        return build_summary_only_prompt(context)
    if mode == "custom" and custom_instruction.strip():
        return build_custom_prompt(context, custom_instruction.strip())
    return build_prompt(query, context)


def call_gemini(prompt: str, api_key: str, model: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 140,
        },
    }
    response = requests.post(url, headers=headers, json=payload, timeout=20)
    response.raise_for_status()
    data = response.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = " ".join(part.get("text", "") for part in parts).strip()
    return clean_text(text)


def generate_llm_summary(
    result: dict,
    doc: dict,
    body: str,
    query: str,
    secrets: object | None = None,
    mode: str = "relevance",
    custom_instruction: str = "",
) -> LlmSummary:
    api_key, model = read_gemini_settings(secrets)
    if not api_key:
        return LlmSummary("", "not configured", "Gemini API key is missing.")

    context, source = build_page_context(result, doc, body, query)
    if not context.strip():
        return LlmSummary("", source, "No page text available.")

    try:
        prompt = route_prompt(mode, query, context, custom_instruction)
        summary = call_gemini(prompt, api_key, model)
        if not summary:
            return LlmSummary("", source, "Gemini returned an empty response.")
        return LlmSummary(summary, source)
    except Exception as exc:
        return LlmSummary("", source, f"Gemini summary failed: {type(exc).__name__}.")
