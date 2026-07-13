# Tuebingen Search Engine

Initial project structure for a student-style search engine project in the course **Modern Search Engines**.

The project will crawl English web pages related to Tuebingen, store them locally, build an index, retrieve ranked results for queries, export batch results, and provide a small Streamlit interface.

At this initial stage, the files are placeholders. Most internal pipeline files are JSON, while the official batch query and result files use the tab-separated text format required by the project instructions.

## Project Structure

```text
mse-tuebingen-search/
|-- README.md
|-- requirements.txt
|-- seeds.json
|-- queries.tsv
|-- data/
|   |-- raw_pages.json
|   |-- index.json
|   |-- frontier.json
|   |-- visited.json
|   |-- crawl_summary.json
|   |-- preprocessed_pages.json
|   |-- index_summary.json
|   |-- results.tsv
|   `-- batch_summary.json
|-- src/
|   |-- crawler.py
|   |-- preprocessing.py
|   |-- text_representations.py
|   |-- indexer.py
|   |-- retrieval.py
|   |-- reranking.py
|   |-- batch.py
|   `-- utils.py
|-- frontend/
|   `-- app.py
`-- scripts/
    |-- crawl.py
    |-- preprocess.py
    |-- build_index.py
    |-- run_batch.py
    `-- smoke_test.py
```

## File Responsibilities

- `seeds.json`: start URLs for the crawler.
- `queries.tsv`: official batch queries, one query per line as `query_id<TAB>query text`.
- `data/raw_pages.json`: locally stored crawled pages.
- `data/frontier.json`: simple JSON list of URLs still waiting to be crawled.
- `data/visited.json`: URLs already visited.
- `data/index.json`: readable index representation for the initial project stage.
- `data/crawl_summary.json`: crawl statistics.
- `data/preprocessed_pages.json`: tokenized document fields after preprocessing.
- `data/index_summary.json`: index statistics.
- `data/results.tsv`: official batch retrieval results, one ranked result per line as `query_id<TAB>rank<TAB>url<TAB>score`.
- `data/batch_summary.json`: compact batch retrieval summary.
- `src/crawler.py`: crawling, URL filtering, page extraction.
- `src/preprocessing.py`: tokenization, normalization, stopword removal, and NLTK Porter stemming.
- `src/text_representations.py`: term frequency, document frequency, IDF helpers.
- `src/indexer.py`: inverted index construction.
- `src/retrieval.py`: BM25 retrieval.
- `src/reranking.py`: second-stage ranking signals.
- `src/batch.py`: batch retrieval export.
- `src/utils.py`: shared helper functions.
- `frontend/app.py`: Streamlit user interface.
- `scripts/`: simple command-line entry points for crawling, preprocessing, indexing, batch retrieval, and smoke testing.

## Pipeline Stages

### 1. Seed URLs

- Input: manually selected Tuebingen-related start URLs
- Processing: read seed URLs and normalize them
- Output: initial crawl frontier
- Files: `seeds.json`, `src/crawler.py`

Example JSON:

```json
{
  "seeds": [
    "https://www.tuebingen.de/en/",
    "https://uni-tuebingen.de/en/",
    "https://www.tuebingen-info.de/en/",
    "https://www.unimuseum.uni-tuebingen.de/en/",
    "https://en.wikipedia.org/wiki/T%C3%BCbingen",
    "https://en.wikivoyage.org/wiki/T%C3%BCbingen"
  ]
}
```

### 2. Crawling

- Input: `seeds.json`, `data/frontier.json`, `data/visited.json`
- Processing: fetch HTML pages with a domain-aware, polite, multi-threaded frontier (`frontier_high`, i.e. Tuebingen-related discoveries, is always drained before `frontier_low`); respect robots.txt; extract title/body/headings/links; filter English Tuebingen-related pages; retry transient failures (connection errors, timeouts, 429/5xx) a few times before giving up on a URL; checkpoint progress periodically so a run can be safely interrupted with Ctrl-C and resumed later
- Output: `data/raw_pages.json`, updated `data/frontier.json`, updated `data/visited.json`, `data/crawl_summary.json`, `crawl.log` (one line per fetch attempt, e.g. `worker=2 t+4231ms domain=uni-tuebingen.de url=... saved: doc_id=7 (8/20 pages)`)
- Files: `src/crawler.py`, `scripts/crawl.py`, `src/utils.py`

Example JSON for `data/raw_pages.json`:

```json
{
  "pages": [
    {
      "doc_id": 0,
      "url": "https://uni-tuebingen.de/en/",
      "fetched_url": "https://uni-tuebingen.de/en/",
      "canonical_url": "https://uni-tuebingen.de/en",
      "title": "Home | University of Tuebingen",
      "headings": ["University of Tuebingen"],
      "body": "Example page text...",
      "outgoing_links": ["https://uni-tuebingen.de/en/study"],
      "language": "en",
      "is_tuebingen_related": true,
      "crawl_time": "2026-07-05T12:00:00Z"
    }
  ]
}
```

Example JSON for `data/frontier.json`. URLs are grouped by domain (for polite, per-domain rate limiting) and split into two priority levels:  `frontier_high` for links discovered on Tuebingen-related pages, `frontier_low` for everything else. `domain_next_time` records the earliest timestamp each domain may be fetched again:

```json
{
  "frontier_high": {
    "www.unimuseum.uni-tuebingen.de": [
      "https://www.unimuseum.uni-tuebingen.de/de/ausstellungen/sonderausstellungen/agora/die-agora-von-athen",
      "https://www.unimuseum.uni-tuebingen.de/de/ausstellungen/sonderausstellungen/agora-1"
    ]
  },
  "frontier_low": {
    "www.unimuseum.uni-tuebingen.de": [
      "https://www.unimuseum.uni-tuebingen.de/de/ausstellungen/sonderausstellungen/dirty-science"
    ]
  },
  "domain_next_time": {
    "www.unimuseum.uni-tuebingen.de": 1751713200.42
  }
}
```

Example JSON for `data/visited.json`. `status_code` is either the HTTP status, or one of `"robots_blocked"` / `"request_error"` / `"error"` for URLs that were permanently skipped rather than fetched. A URL only appears here once its retries are exhausted:

```json
{
  "visited": [
    {
      "url": "https://uni-tuebingen.de/en/",
      "visited_at": "2026-07-05T12:00:00Z",
      "status_code": 200
    }
  ]
}
```

Example JSON for `data/crawl_summary.json`. This accumulates across runs (`fresh=True` resets it) - `_this_run` fields describe the run that just finished, `_total` fields are cumulative since the last fresh start:

```json
{
  "step": "crawling",
  "runs_completed": 1,
  "total_pages": 1,
  "attempted_urls_this_run": 1,
  "attempted_urls_total": 1,
  "frontier_high_size": 0,
  "frontier_low_size": 1,
  "visited_size": 1,
  "timeout": 8.0,
  "polite_delay": 0.6,
  "workers": 4,
  "last_run_interrupted": false,
  "interrupted_runs": 0,
  "elapsed_seconds_this_run": 3.42,
  "elapsed_human_this_run": "3s",
  "elapsed_seconds_total": 3.42,
  "elapsed_human_total": "3s",
  "started_fresh_last_run": true,
  "last_updated": "2026-07-05T12:00:00Z"
}
```

### 3. Preprocessing

- Input: raw page text from `data/raw_pages.json`
- Processing: lowercase, tokenize, remove punctuation, normalize Tuebingen spelling variants, normalize German umlauts, remove stopwords, apply NLTK Porter stemming
- Tuebingen normalization must map common spelling variants and common encoding artifacts to `tubingen`, including `Tübingen`, `Tuebingen`, `Tubingen`, `TÃ¼bingen`, and `TÃœBINGEN`
- General German character normalization should also be applied before tokenization, for example `ä -> ae` or `a`, `ö -> oe` or `o`, `ü -> ue` or `u`, and `ß -> ss`
- The indexer and retrieval code must use the same `preprocess(...)` function, so document tokens and query tokens are normalized, filtered, and stemmed in exactly the same way
- Output: tokenized document fields in `data/preprocessed_pages.json`
- Files: `src/preprocessing.py`, `scripts/preprocess.py`

Example JSON:

```json
{
  "documents": [
    {
      "doc_id": 0,
      "url": "https://uni-tuebingen.de/en/",
      "fetched_url": "https://uni-tuebingen.de/en/",
      "canonical_url": "https://uni-tuebingen.de/en",
      "title": "Home | University of Tuebingen",
      "snippet": "Example snippet...",
      "title_tokens": ["home", "univers", "tubingen"],
      "heading_tokens": ["univers", "tubingen"],
      "body_tokens": ["student", "research", "campus"],
      "body_tokens_preview": ["student", "research", "campus"],
      "body_length": 320,
      "outgoing_links": ["https://uni-tuebingen.de/en/study"],
      "crawl_time": "2026-07-05T12:00:00Z"
    }
  ]
}
```

### 4. Indexing

- Input: preprocessed document tokens
- Processing: build manual inverted index with term frequencies, document lengths, document metadata, title and heading tokens for re-ranking, field lengths, and link graph
- Output: `data/index.json`, `data/index_summary.json`
- Files: `src/indexer.py`, `scripts/build_index.py`

Example JSON for `data/index.json`:

```json
{
  "documents": [
    {
      "doc_id": 0,
      "url": "https://uni-tuebingen.de/en/",
      "fetched_url": "https://uni-tuebingen.de/en/",
      "canonical_url": "https://uni-tuebingen.de/en",
      "title": "Home | University of Tuebingen",
      "snippet": "Example snippet...",
      "title_tokens": ["home", "univers", "tubingen"],
      "heading_tokens": ["univers", "tubingen"],
      "doc_length": 320,
      "outgoing_links": ["https://uni-tuebingen.de/en/study"],
      "crawl_time": "2026-07-05T12:00:00Z"
    }
  ],
  "inverted_index": {
    "tubingen": [
      {
        "doc_id": 0,
        "tf": 4
      }
    ]
  },
  "document_frequencies": {
    "tubingen": 1
  },
  "field_lengths": {
    "body": {
      "0": 320
    },
    "title": {
      "0": 3
    },
    "heading": {
      "0": 2
    }
  },
  "average_document_length": 320.0,
  "link_graph": {
    "0": []
  }
}
```

Example JSON for `data/index_summary.json`:

```json
{
  "step": "indexing",
  "num_docs": 1,
  "vocabulary_size": 1,
  "average_document_length": 320.0,
  "documents_with_outgoing_links": 1,
  "index_file_size_mb": 0.01,
  "elapsed_seconds": 0.0123
}
```

### 5. Retrieval / BM25

- Input: query from `queries.tsv` or the UI, plus `data/index.json`
- Processing: preprocess query with the same `preprocess(...)` function used for documents, score documents with manually implemented BM25, retrieve top candidates and attach document metadata such as title, URL, and snippet
- Output: first-stage ranked result list
- Files: `src/retrieval.py`, `src/preprocessing.py`

Example JSON:

```json
{
  "query_id": "1",
  "query": "tuebingen attractions",
  "query_tokens": ["tubingen", "attract"],
  "candidates": [
    {
      "doc_id": 0,
      "url": "https://uni-tuebingen.de/en/",
      "title": "Home | University of Tuebingen",
      "snippet": "Example snippet...",
      "bm25_score": 2.41,
      "matched_terms": ["tubingen"]
    }
  ]
}
```

### 6. Re-Ranking

- Input: BM25 candidate results
- Processing: add field boosts, pseudo relevance feedback bonus, link score bonus, optional semantic LSA/SVD score
- Output: final ranked result list
- Files: `src/reranking.py`, `src/retrieval.py`

Example JSON:

```json
{
  "query_id": "1",
  "query": "tuebingen attractions",
  "expansion_terms": ["museum", "castle"],
  "results": [
    {
      "rank": 1,
      "doc_id": 0,
      "url": "https://uni-tuebingen.de/en/",
      "title": "Home | University of Tuebingen",
      "snippet": "Example snippet...",
      "score": 0.84,
      "score_details": {
        "normalized_bm25": 0.7,
        "normalized_field_boost": 0.1,
        "normalized_prf": 0.02,
        "normalized_link": 0.01,
        "normalized_lsa": 0.01
      }
    }
  ]
}
```

### 7. Batch Output

- Input: `queries.tsv`, `data/index.json`
- Processing: run retrieval for each query and keep up to 100 results
- Output: `data/results.tsv`, `data/batch_summary.json`
- Files: `src/batch.py`, `scripts/run_batch.py`

Example TSV for `queries.tsv`:

```text
1	tübingen attractions
2	food and drinks
```

Example TSV for `data/results.tsv`:

```text
1	1	https://www.tuebingen.de/en/3521.html	0.725
1	2	https://www.komoot.com/guide/355570/castles-in-tuebingen-district	0.671
1	3	https://www.unimuseum.uni-tuebingen.de/en/museum-at-hohentuebingen-castle	0.529
1	100	https://www.tuebingen.de/en/3536.html	0.178
2	1	https://www.tuebingen.de/en/3773.html	0.956
2	2	https://www.tuebingen.de/en/4456.html	0.797
```

Optional internal JSON summary for `data/batch_summary.json`:

```json
{
  "step": "batch_retrieval_summary",
  "query_count": 1,
  "total_results": 1,
  "average_runtime_seconds": 0.05
}
```

### 8. Streamlit UI

- Input: interactive user query in the browser
- Processing: run retrieval, show result cards, score breakdown, highlighted query terms, and filters
- Output: interactive search interface
- Files: `frontend/app.py`, `src/retrieval.py`

Example JSON for an internal UI result object:

```json
{
  "query": "tuebingen attractions",
  "search_time_seconds": 0.05,
  "indexed_pages": 1,
  "selected_category": "Attractions",
  "result_cards": [
    {
      "rank": 1,
      "title": "Home | University of Tuebingen",
      "url": "https://uni-tuebingen.de/en/",
      "snippet": "Example snippet...",
      "highlighted_terms": ["tubingen", "attract"],
      "category": "Attractions",
      "score": 0.84,
      "why_this_result": "Matched query terms in body and title."
    }
  ]
}
```
