from __future__ import annotations

import heapq
import logging
import re
import random
import threading
import time
import urllib.robotparser
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langdetect import detect

from src.utils import (
    get_domain,
    is_probably_html_url,
    normalize_url,
    now_utc_iso,
    project_path,
    read_json,
    short_snippet,
    write_json,
)


USER_AGENT = "MSE-Tuebingen-StudentCrawler | Contact: janis.weller@student.uni-tuebingen.de"

# Domains we never want to fetch or queue.
BLACKLISTED_DOMAINS: frozenset[str] = frozenset({
    "archive.org",
})

# Regex
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_TUEBINGEN_RE = re.compile(
    r"""
    t                                    # leading 't'
    (?:
        [ue\u00fc]{1,6}                  # vowel-only fuzzy match: u, ue, uu, uuuu, ü ...
      | &\#?x?[0-9a-z]{2,6};             # un-decoded HTML entity: &uuml; &#252; &#xfc;
      | (?:%[0-9a-f]{2}){2}              # un-decoded percent-encoded UTF-8 bytes for ü/Ü
    )
    bingen
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_blacklisted_domain(domain: str) -> bool:
    """True if `domain` is (or is a subdomain of) a blacklisted domain."""
    domain = domain.lower()
    return any(domain == blocked or domain.endswith("." + blocked) for blocked in BLACKLISTED_DOMAINS)

# How often (seconds) an idle worker is allowed to log a "still waiting" heartbeat, so long
# waits on a domain cooldown are visible instead of just going silent.
_IDLE_HEARTBEAT_INTERVAL = 20.0

# HTTP statuses worth retrying rather than permanently blacklisting the URL
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

logger = logging.getLogger("crawler")
logger.setLevel(logging.INFO)


class _WorkerLineFilter(logging.Filter):
    """Only let genuine per-worker log lines (e.g. "worker=2 t+123ms ...") through."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().startswith("worker=")


def _setup_logging(log_path: str | Path) -> None:
    """Point the crawler's logger at log_path as well as the console."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_WorkerLineFilter())
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.propagate = False


def load_seed_urls(seeds_path: str | Path = project_path("seeds.json")) -> list[str]:
    """Read the flat list of seed URLs from seeds.json and normalize them (no priority or title fields)."""
    data = read_json(seeds_path, {"seeds": []})
    seeds = data.get("seeds", [])
    logger.info(f"loaded {len(seeds)} seed urls from {seeds_path}")
    return [normalize_url(seed) for seed in seeds if seed]


def _extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Pull all usable HTML links out of a parsed page, normalized against the base URL."""
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag.get("href", "").strip()
        # Skip non-navigable links (email, phone, inline scripts).
        if href.lower().startswith(("mailto:", "tel:", "javascript:", "sms:", "fax:")):
            continue
        if _EMAIL_RE.fullmatch(href):
            continue
        # Resolve relative links to absolute URLs and keep only likely HTML pages
        normalized = normalize_url(href, base_url)
        if is_probably_html_url(normalized):
            links.append(normalized)
    # Dedupe and return a stable, sorted order
    return sorted(set(links))


def _url_key(url: str) -> str:
    """Normalize a URL into a comparable key (used for dedup) by stripping the trailing slash."""
    return normalize_url(url).rstrip("/")


def _extract_body(soup: BeautifulSoup) -> str:
    """Strip non-content tags and return the visible text of the page's main/body content."""
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    main = soup.find("main") or soup.body or soup
    return " ".join(main.get_text(" ", strip=True).split())


def _detect_language(text: str, html_lang: str) -> str:
    """Determine the page language."""
    try:
        detected = detect(short_snippet(text, 1000))
    except Exception:
        detected = None

    if detected:
        return detected
    if html_lang:
        return html_lang.lower().split("-")[0]
    return "unknown"


def _is_tuebingen_related(url: str, title: str, body: str) -> bool:
    """Check whether the URL, title, or body text mentions Tuebingen in any common spelling variant."""
    text = f"{url} {title} {body}".lower()
    return bool(_TUEBINGEN_RE.search(text))


def _looks_english(url: str, language: str, body: str) -> bool:
    """Decide if a page is English, trusting langdetect first and the /en/ URL path as a fallback."""
    if language == "en":
        return True
    if language and language != "unknown":
        return False
    return "/en" in urlparse(url).path.lower()


def _format_elapsed(seconds: float) -> str:
    """Format a duration in seconds as a compact H:MM:SS-ish string."""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class _FrontierLevel:
    """The domain-aware frontier for a single priority level (high or low)."""

    __slots__ = ("domains", "heap", "url_set")

    def __init__(self) -> None:
        self.domains: dict[str, deque[str]] = {}
        self.heap: list[tuple[float, float, str]] = []
        self.url_set: set[str] = set()


class CrawlerState:
    """All mutable state shared across worker threads, guarded by a single lock"""

    def __init__(
        self,
        frontier_high: list[str],
        frontier_low: list[str],
        pages: list[dict[str, Any]],
        visited_entries: list[dict[str, Any]],
        max_pages: int,
        polite_delay: float,
        checkpoint_every: int,
        domain_next_time: dict[str, float] | None = None,
        elapsed_offset: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        self.start_time = time.time() - elapsed_offset
        self.lock = threading.Lock()
        # Guards the writing of checkpoint files.
        self.checkpoint_lock = threading.Lock()

        # Per-URL count of transient-failure retries so far.
        self.retry_counts: dict[str, int] = {}
        self.max_retries = max_retries

        self.frontier_high = _FrontierLevel()
        self.frontier_low = _FrontierLevel()

        # domain -> earliest timestamp (time.time()) at which we're allowed to fetch it again.
        self.domain_next_time: dict[str, float] = dict(domain_next_time or {})

        for url in frontier_high:
            self._enqueue(url, self.frontier_high)
        for url in frontier_low:
            self._enqueue(url, self.frontier_low)

        self.pages = pages
        self.visited_entries = visited_entries
        self.visited_urls = {item.get("url") for item in visited_entries}
        self.known_page_keys = {
            _url_key(page.get("canonical_url") or page.get("url", ""))
            for page in pages
            if page.get("canonical_url") or page.get("url")
        }
        self.next_doc_id = max([int(page.get("doc_id", -1)) for page in pages] + [-1]) + 1
        self.max_pages = max_pages
        self.polite_delay = polite_delay
        self.checkpoint_every = checkpoint_every
        self.next_checkpoint_at = checkpoint_every if checkpoint_every > 0 else None

        self.attempted = 0
        self.saved = 0
        self.in_flight = 0  # URLs currently being fetched/processed by a worker
        self.in_flight_urls: set[str] = set()
        self.interrupted = False
        self.stop_event = threading.Event()

        # robots.txt cache is read/written far less often than the frontier, so it gets its own lock instead of contending with the hot path above.
        self.robots_lock = threading.Lock()
        self.robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}


    def _enqueue(self, url: str, level: _FrontierLevel) -> None:
        """Add a single url to one priority level's domain-aware frontier."""
        if url in level.url_set:
            return
        if level == self.frontier_low:
            return
        try:
            domain = get_domain(url)
        except Exception:
            # Malformed URL - drop it rather than let it poison the frontier.
            return

        if _is_blacklisted_domain(domain) or not is_probably_html_url(url):
            return

        queue = level.domains.get(domain)
        if queue is None:
            queue = deque()
            level.domains[domain] = queue
            heapq.heappush(level.heap, (self.domain_next_time.get(domain, 0.0), random.random(), domain))

        queue.append(url)
        level.url_set.add(url)

    @staticmethod
    def _frontier_size(level: _FrontierLevel) -> int:
        """Total number of urls currently waiting across all domains at one priority level."""
        return sum(len(queue) for queue in level.domains.values())

    def _claim_from_heap(self, level: _FrontierLevel, now: float) -> str | None:
        """Pop one ready url from a single priority level's domain-aware frontier."""
        domains, url_set, heap = level.domains, level.url_set, level.heap
        while heap:
            stored_time, tiebreak, domain = heap[0]
            true_time = self.domain_next_time.get(domain, 0.0)
            if true_time > stored_time:
                # Stale entry - someone updated this domain's cooldown after we scheduled it. Fix it in place (keep the same tiebreak) and recheck from the top.
                heapq.heapreplace(heap, (true_time, tiebreak, domain))
                continue

            if stored_time > now:
                # Earliest domain in this heap still isn't ready; nothing else in the heap can be ready sooner, so stop here.
                return None

            queue = domains.get(domain)
            if not queue:
                # Domain drained already -> clean up and keep looking.
                heapq.heappop(heap)
                domains.pop(domain, None)
                continue

            # Domain is ready: pop its heap entry now so no other worker can claim from it until we push it back below (or it's found empty).
            heapq.heappop(heap)

            claimed_url: str | None = None
            while queue:
                url = queue.popleft()
                url_set.discard(url)
                if url in self.visited_urls or not is_probably_html_url(url):
                    continue
                claimed_url = url
                break

            if not queue:
                domains.pop(domain, None)

            if claimed_url is None:
                # Every url we found for this domain was stale/invalid; the domain itself isn't on cooldown, so if it still has urls left, put it
                # back at the front of the heap (unchanged cooldown/tiebreak) and keep looking at the next domain instead of returning empty-handed.
                if queue:
                    heapq.heappush(heap, (stored_time, tiebreak, domain))
                continue

            # Reserve this domain's next slot immediately, so a second worker can't grab another URL for the same domain before this fetch starts.
            self.domain_next_time[domain] = now + self.polite_delay
            if queue:
                heapq.heappush(heap, (self.domain_next_time[domain], random.random(), domain))
            self.in_flight += 1
            self.in_flight_urls.add(claimed_url)
            self.attempted += 1
            return claimed_url

        return None

    def _claim_least_saturated(self, level: _FrontierLevel, now: float) -> str | None:
        """Pop one ready url from a single priority level's domain-aware frontier, but deliberately target the ready domain with the fewest queued urls."""
        domains, url_set = level.domains, level.url_set
        while True:
            ready = [
                (len(queue), domain)
                for domain, queue in domains.items()
                if queue and self.domain_next_time.get(domain, 0.0) <= now
            ]
            if not ready:
                return None

            min_len = min(size for size, _domain in ready)
            domain = random.choice([d for size, d in ready if size == min_len])
            queue = domains[domain]

            claimed_url: str | None = None
            while queue:
                url = queue.popleft()
                url_set.discard(url)
                if url in self.visited_urls or not is_probably_html_url(url):
                    continue
                claimed_url = url
                break

            if not queue:
                domains.pop(domain, None)

            if claimed_url is None:
                # Every url queued for this domain was stale/invalid; try the next least-saturated domain instead of returning empty-handed.
                continue

            # Reserve this domain's next slot immediately, same as the heap-based claim path, and keep the
            # heap in sync so a subsequent Horse claim sees the up-to-date cooldown for this domain.
            self.domain_next_time[domain] = now + self.polite_delay
            if queue:
                heapq.heappush(level.heap, (self.domain_next_time[domain], random.random(), domain))
            self.in_flight += 1
            self.in_flight_urls.add(claimed_url)
            self.attempted += 1
            return claimed_url

    def _earliest_ready_time(self, level: _FrontierLevel) -> float | None:
        """Peek (without popping) the true next_allowed_time of the earliest domain in a level's heap."""
        if not level.heap:
            return None
        stored_time, _tiebreak, domain = level.heap[0]
        return max(stored_time, self.domain_next_time.get(domain, 0.0))

    def claim_next_url(self, role: str = "horse") -> tuple[str | None, str, str | None]:
        """Try to claim the next URL that is both in a frontier and not on its domain's cooldown. frontier_high (Tuebingen-related discoveries) is always scanned before frontier_low.

        `role` picks the claim strategy within whichever level is scanned:
        - "horse": claim in cooldown order off the heap (the original behavior) - whichever ready domain comes up next.
        - "explorer": claim from the ready domain with the fewest queued urls, deliberately spreading coverage to under-represented domains instead of following cooldown order.

        Returns (url, "ok", level) if a URL was claimed (this also marks it in-flight and reserves its domain's cooldown slot) - level is "high" or "low", identifying which frontier it came from.
        Returns (None, "wait", None) if nothing is ready yet but there's still work outstanding, or (None, "done", None) if the crawl should stop (max_pages reached, or both
        frontiers are empty with no other worker in-flight that could still add to them).
        """
        with self.lock:
            if self.stop_event.is_set() or len(self.pages) >= self.max_pages:
                self.stop_event.set()
                return None, "done", None

            now = time.time()
            claim = self._claim_least_saturated if role == "explorer" else self._claim_from_heap
            claimed_url = claim(self.frontier_high, now)
            level = "high"
            if claimed_url is None:
                claimed_url = claim(self.frontier_low, now)
                level = "low"

            if claimed_url is not None:
                return claimed_url, "ok", level

            if not self.frontier_high.domains and not self.frontier_low.domains and self.in_flight == 0:
                self.stop_event.set()
                return None, "done", None
            return None, "wait", None

    def frontier_status_summary(self) -> str:
        """Human-readable snapshot of frontier size/domain diversity, used for idle-worker heartbeats."""
        with self.lock:
            now = time.time()
            domains_high = len(self.frontier_high.domains)
            domains_low = len(self.frontier_low.domains)
            ready_high = sum(1 for d in self.frontier_high.domains if self.domain_next_time.get(d, 0.0) <= now)
            ready_low = sum(1 for d in self.frontier_low.domains if self.domain_next_time.get(d, 0.0) <= now)
            return (
                f"frontier_high={self._frontier_size(self.frontier_high)} urls/{domains_high} domains ({ready_high} ready), "
                f"frontier_low={self._frontier_size(self.frontier_low)} urls/{domains_low} domains ({ready_low} ready), "
                f"in_flight={self.in_flight}"
            )

    def seconds_until_next_domain(self) -> float:
        """How long a worker should sleep before it's worth calling claim_next_url() again."""
        with self.lock:
            now = time.time()
            candidates = [
                t
                for t in (self._earliest_ready_time(self.frontier_high), self._earliest_ready_time(self.frontier_low))
                if t is not None
            ]
            if not candidates:
                return 0.0
            soonest = min(candidates)
            return max(0.0, soonest - now)

    def finish_url(self, domain: str, url: str, finished_at: float) -> None:
        """Release the in-flight slot and make sure the domain's cooldown reflects when the fetch actually completed."""
        with self.lock:
            self.in_flight -= 1
            self.in_flight_urls.discard(url)
            candidate = finished_at + self.polite_delay
            if candidate > self.domain_next_time.get(domain, 0.0):
                self.domain_next_time[domain] = candidate

    def add_links(self, links: list[str], qualifies_for_high: bool) -> None:
        """Queue newly discovered links. If the page they were found on is both Tuebingen-related AND English,
        they go into frontier_high; otherwise frontier_low."""
        with self.lock:
            target_level = self.frontier_high if qualifies_for_high else self.frontier_low
            for link in links:
                if (
                    link not in self.visited_urls
                    and link not in self.frontier_high.url_set
                    and link not in self.frontier_low.url_set
                    and link not in self.in_flight_urls
                ):
                    # TODO: Remove this but instead of saving to low we throw it away for now since we never touch low
                    self._enqueue(link, target_level)

    def record_visited(self, url: str, status_code: Any) -> None:
        with self.lock:
            self.visited_entries.append({"url": url, "visited_at": now_utc_iso(), "status_code": status_code})
            self.visited_urls.add(url)

    def note_failure(self, url: str, level_name: str | None, transient: bool, status_for_record: Any) -> None:
        """Handle a failed fetch (connection error, timeout, or a retryable HTTP status like 429/5xx)."""
        with self.lock:
            if transient and level_name is not None:
                attempts = self.retry_counts.get(url, 0) + 1
                self.retry_counts[url] = attempts
                if attempts <= self.max_retries:
                    level = self.frontier_high if level_name == "high" else self.frontier_low
                    self._enqueue(url, level)
                    return
            self.visited_entries.append({"url": url, "visited_at": now_utc_iso(), "status_code": status_for_record})
            self.visited_urls.add(url)

    def try_save_page(self, canonical_url: str, page: dict[str, Any]) -> tuple[str, int, int, bool]:
        """Try to save a page.

        Returns (status, doc_id, total_pages, over_cap):
        - status is "saved", or "duplicate" if the canonical URL was already saved.
        - doc_id is the assigned doc id (only meaningful when status == "saved").
        - total_pages is len(self.pages) after this call.
        - over_cap is True if this save pushed the file above max_pages.

        Note: this does NOT reject a save just because max_pages was already reached. A worker
        may have already fetched and parsed a page by the time another worker's save hits the cap and sets
        stop_event. In-flight pages will get saved over the cap.
        """
        with self.lock:
            canonical_key = _url_key(canonical_url)
            if canonical_key in self.known_page_keys:
                return "(any common spelling variant), duplicate", -1, len(self.pages), False
            page["doc_id"] = self.next_doc_id
            self.pages.append(page)
            self.known_page_keys.add(canonical_key)
            self.next_doc_id += 1
            self.saved += 1
            total_pages = len(self.pages)
            over_cap = total_pages > self.max_pages
            if total_pages >= self.max_pages:
                self.stop_event.set()
            return "saved", page["doc_id"], total_pages, over_cap

    def robots_allowed(self, url: str, user_agent: str, session: requests.Session, timeout: float) -> bool:
        """Check robots.txt (cached per host) to see if we're allowed to fetch this URL."""
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        with self.robots_lock:
            parser = self.robots_cache.get(robots_url)
        if parser is None:
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)
            try:
                response = _fetch_with_hard_timeout(session, robots_url, timeout)
            except requests.RequestException:
                # If robots.txt can't be fetched within the timeout, default to allowing the crawl
                return True
            # Mirror RobotFileParser.read() own status-code handling
            if response.status_code in (401, 403):
                parser.disallow_all = True
            elif response.status_code >= 400:
                parser.allow_all = True
            else:
                parser.parse(response.text.splitlines())
            with self.robots_lock:
                self.robots_cache[robots_url] = parser
        try:
            return parser.can_fetch(user_agent, url)
        except Exception:
            return True

    def should_checkpoint(self) -> bool:
        """Decide whether the calling worker is the one that should run the next checkpoint."""
        if self.next_checkpoint_at is None:
            return False
        with self.lock:
            if self.attempted >= self.next_checkpoint_at:
                # Snap forward past the current attempted count so that a burst of workers finishing at once.
                self.next_checkpoint_at = self.attempted + self.checkpoint_every
                return True
            return False

    def snapshot_for_checkpoint(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, list[str]], dict[str, float], list[dict[str, Any]]]:
        """Take a consistent, lock-protected copy of everything that gets written to disk."""
        with self.lock:
            frontier_high = {domain: list(queue) for domain, queue in self.frontier_high.domains.items() if queue}
            frontier_low = {domain: list(queue) for domain, queue in self.frontier_low.domains.items() if queue}
            return (
                list(self.pages),
                frontier_high,
                frontier_low,
                dict(self.domain_next_time),
                list(self.visited_entries),
            )


def _frontier_total(frontier: dict[str, list[str]]) -> int:
    """Total number of urls across all domains in a checkpointed (domain -> urls) frontier dict."""
    return sum(len(urls) for urls in frontier.values())


def _flatten_frontier(value: dict[str, list[str]]) -> list[str]:
    """Flatten a checkpointed frontier ({domain: [urls...]}) into a flat url list for CrawlerState's constructor."""
    flat: list[str] = []
    for urls in value.values():
        flat.extend(urls)
    return flat


def _save_state(
    raw_pages_path: Path,
    frontier_path: Path,
    visited_path: Path,
    pages: list[dict[str, Any]],
    frontier_high: dict[str, list[str]],
    frontier_low: dict[str, list[str]],
    domain_next_time: dict[str, float],
    visited_entries: list[dict[str, Any]],
) -> None:
    """Persist the current crawler state (pages/frontier/visited) to disk.

    Both frontier queues are written into a single frontier.json file, as
    {"frontier_high": {domain: [urls...]}, "frontier_low": {domain: [urls...]},
    "domain_next_time": {domain: timestamp}} - enough to reconstruct each domain's
    url queue plus its heap position (next_allowed_time) on resume.
    """
    write_json(raw_pages_path, {"pages": pages})
    write_json(
        frontier_path,
        {"frontier_high": frontier_high, "frontier_low": frontier_low, "domain_next_time": domain_next_time},
    )
    write_json(visited_path, {"visited": visited_entries})


def _fetch_with_hard_timeout(session: requests.Session, url: str, timeout: float, **kwargs: Any) -> requests.Response:
    """Run session.get() in a background daemon thread and enforce a hard wall-clock deadline."""
    outcome: dict[str, Any] = {}
    done = threading.Event()

    def _do_get() -> None:
        try:
            outcome["response"] = session.get(url, timeout=timeout, **kwargs)
        except Exception as exc:  # re-raised on the caller's thread below
            outcome["error"] = exc
        finally:
            done.set()

    threading.Thread(target=_do_get, daemon=True, name="crawler-fetch").start()
    # Small grace period on top of requests' own timeout so genuine (non-stuck) slow connect/read timeouts have a chance to raise and be reported as their real error.
    if not done.wait(timeout=timeout + 1):
        raise requests.Timeout(f"hard timeout ({timeout + 1:.0f}s) exceeded while fetching {url}")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["response"]


def _process_url(state: CrawlerState, session: requests.Session, worker_id: int, role: str, url: str, timeout: float, level_name: str | None) -> None:
    """Fetch, parse, and (maybe) save a single URL. Always releases the URL's in-flight/cooldown slot when done."""
    domain = "unknown"
    status_code: Any = None
    reason = "unknown"
    visited_recorded = False
    try:
        domain = get_domain(url)
        if not state.robots_allowed(url, USER_AGENT, session, timeout):
            state.record_visited(url, "robots_blocked")
            visited_recorded = True
            reason = "skipped: blocked by robots.txt"
            return

        response = _fetch_with_hard_timeout(session, url, timeout, allow_redirects=True)
        status_code = response.status_code
        fetched_url = normalize_url(response.url)
        content_type = response.headers.get("content-type", "")

        if status_code in _TRANSIENT_STATUS_CODES:
            state.note_failure(url, level_name, transient=True, status_for_record=status_code)
            visited_recorded = True
            reason = f"retry-scheduled: transient status={status_code}"
            return

        state.record_visited(url, status_code)
        visited_recorded = True

        if status_code != 200 or "text/html" not in content_type.lower():
            reason = f"skipped: not a 200 html response (status={status_code}, content-type={content_type})"
            return

        soup = BeautifulSoup(response.text, "html.parser")
        canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
        canonical_url = normalize_url(canonical.get("href"), fetched_url) if canonical and canonical.get("href") else fetched_url
        title = soup.title.get_text(" ", strip=True) if soup.title else fetched_url
        headings = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"]) if h.get_text(strip=True)]
        body = _extract_body(soup)
        outgoing_links = _extract_links(soup, fetched_url)
        html_lang = soup.html.get("lang", "") if soup.html else ""
        language = _detect_language(body, html_lang)
        is_related = _is_tuebingen_related(canonical_url, title, body)
        is_english = _looks_english(canonical_url, language, body)

        # Queue newly discovered links for later, regardless of whether this page is saved.
        # Only pages that are both Tuebingen-related AND English promote their links to frontier_high.
        state.add_links(outgoing_links, is_related and is_english)

        if not is_related or not is_english:
            skip_reasons = []
            if not is_related:
                skip_reasons.append("not tuebingen-related")
            if not is_english:
                skip_reasons.append(f"not english (detected={language})")
            reason = f"skipped: {' + '.join(skip_reasons)}"
            return

        page = {
            "url": url,
            "fetched_url": fetched_url,
            "canonical_url": canonical_url,
            "title": title,
            "headings": headings,
            "body": body,
            "outgoing_links": outgoing_links,
            "language": language,
            "is_tuebingen_related": is_related,
            "crawl_time": now_utc_iso(),
        }
        status, doc_id, total_pages, over_cap = state.try_save_page(canonical_url, page)
        if status == "saved":
            reason = f"saved: doc_id={doc_id} ({total_pages}/{state.max_pages} pages)"
            if over_cap:
                reason += " [over max_pages cap, already-fetched work kept]"
        else:
            reason = f"skipped: canonical url already saved ({canonical_url})"
    except requests.RequestException as exc:
        # Connection errors/timeouts are transport-level hiccups, not judgments about the page retry a few times instead of instantly blacklisting.
        reason = f"retry-scheduled: request error: {exc}"
        if not visited_recorded:
            state.note_failure(url, level_name, transient=True, status_for_record=status_code or "request_error")
    except Exception as exc:
        reason = f"skipped: unexpected error: {type(exc).__name__}: {exc}"
        if not visited_recorded:
            state.record_visited(url, status_code or "error")
    finally:
        elapsed_ms = (time.time() - state.start_time) * 1000
        logger.info(f"worker={worker_id} role={role} t+{elapsed_ms:.0f}ms domain={domain} url={url} {reason}")
        state.finish_url(domain, url, time.time())


def _worker_loop(state: CrawlerState, worker_id: int, role: str, timeout: float, checkpoint_paths: tuple[Path, Path, Path]) -> None:
    """Main loop for a single worker thread: repeatedly claim a ready URL and process it.

    `role` is either "horse" (claims in the frontier's normal cooldown order) or "explorer" (deliberately claims from whichever ready domain currently has the fewest queued urls).
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "en;q=1.0, de;q=0.2",
    })
    last_heartbeat = 0.0
    while not state.stop_event.is_set():
        try:
            url, status, level_name = state.claim_next_url(role)
        except Exception as exc:
            # claim_next_url() guards its own risky calls, so this shouldn't fire -
            # but an uncaught exception here would silently kill this thread, and a
            # dead worker permanently loses its share of concurrency. Log and retry
            # rather than let the thread disappear.
            logger.info(f"worker={worker_id} role={role} error claiming next url: {type(exc).__name__}: {exc}")
            time.sleep(0.05)
            continue

        if status == "done":
            return
        if status == "wait":
            # Nothing is ready right now (either the frontier is empty but other workers are still in-flight and might add to it, or every candidate domain is on cooldown).
            wait_seconds = state.seconds_until_next_domain()
            now = time.time()
            if now - last_heartbeat >= _IDLE_HEARTBEAT_INTERVAL:
                last_heartbeat = now
                logger.info(
                    f"worker={worker_id} role={role} idle: waiting up to {wait_seconds:.1f}s for a domain cooldown to clear "
                    f"({state.frontier_status_summary()})"
                )
            time.sleep(max(0.01, wait_seconds))
            continue

        assert url is not None
        try:
            _process_url(state, session, worker_id, role, url, timeout, level_name)
        except Exception as exc:
            logger.info(f"worker={worker_id} role={role} unexpected crash processing {url}: {type(exc).__name__}: {exc}")

        try:
            # Once shutdown has started, skip periodic checkpointing here. crawl() already performs one final save after all workers join.
            if not state.stop_event.is_set() and state.should_checkpoint():
                with state.checkpoint_lock:
                    raw_pages_path, frontier_path, visited_path = checkpoint_paths
                    pages, frontier_high, frontier_low, domain_next_time, visited_entries = state.snapshot_for_checkpoint()
                    _save_state(
                        raw_pages_path,
                        frontier_path,
                        visited_path,
                        pages,
                        frontier_high,
                        frontier_low,
                        domain_next_time,
                        visited_entries,
                    )
                    logger.info(
                        f"checkpoint saved (attempted={state.attempted}, saved={state.saved}, "
                        f"frontier_high={_frontier_total(frontier_high)}, frontier_low={_frontier_total(frontier_low)})"
                    )
        except Exception as exc:
            # Disk-full, permissions, or a stray non-serializable value shouldn't be able to kill a worker thread.
            logger.info(f"worker={worker_id} checkpoint save failed: {type(exc).__name__}: {exc}")


def crawl(
    seeds_path: str | Path = project_path("seeds.json"),
    raw_pages_path: str | Path = project_path("data", "raw_pages.json"),
    frontier_path: str | Path = project_path("data", "frontier.json"),
    visited_path: str | Path = project_path("data", "visited.json"),
    log_path: str | Path = project_path("data", "crawl.log"),
    max_pages: int = 20,
    timeout: float = 8.0,
    polite_delay: float = 0.6,
    checkpoint_every: int = 5,
    workers: int = 8,
    explorer_workers: int = 2,
    fresh: bool = False,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Run the crawl: fetch frontier URLs (frontier_high first, then frontier_low) with `workers` concurrent
    threads, filter/save relevant pages, and persist crawler state.

    Workers come in two flavors, both always honoring frontier_high before frontier_low:
    - Horses (the default): claim in the frontier's normal cooldown order, same as before.
    - Explorers: deliberately claim from whichever ready domain currently has the fewest urls
      queued, to keep small/under-represented domains from being starved by big ones.

    `explorer_workers` sets how many of the `workers` threads are Explorers (the rest are Horses);
    it's clamped to [0, workers].
    """
    raw_pages_path = Path(raw_pages_path)
    frontier_path = Path(frontier_path)
    visited_path = Path(visited_path)
    log_path = Path(log_path)
    if fresh and log_path.exists():
        # Keep crawl.log consistent with the other state files that --fresh discards.
        log_path.unlink()
    _setup_logging(log_path)
    start_time = time.time()

    domain_next_time: dict[str, float] = {}
    if fresh:
        # Ignore anything on disk and start over from just the seeds.
        pages = []
        frontier_high = load_seed_urls(seeds_path)
        frontier_low = []
        visited_entries = []
        logger.info("--fresh: discarding any existing raw_pages.json/frontier.json/visited.json and starting from seeds")
    else:
        # Load any previously saved pages, and resume both frontiers from disk (or from 0).
        # Both queues live in one frontier.json: {"frontier_high": {domain: [urls...]},
        # "frontier_low": {domain: [urls...]}, "domain_next_time": {domain: timestamp}}.
        raw = read_json(raw_pages_path, {"pages": []})
        pages = raw.get("pages", [])
        frontier_data = read_json(frontier_path, {"frontier_high": {}, "frontier_low": {}, "domain_next_time": {}})
        frontier_high = _flatten_frontier(frontier_data.get("frontier_high", {}))
        #frontier_low = _flatten_frontier(frontier_data.get("frontier_low", {}))
        # TODO: Remove this and reset but frontier low is never touched so we overwrite it with nothing to save memory and cpu time
        frontier_low = []
        domain_next_time = frontier_data.get("domain_next_time", {}) or {}
        if not frontier_high and not frontier_low:
            # Fresh start: seed URLs are presumed Tuebingen-relevant, so they go straight into frontier_high.
            frontier_high = load_seed_urls(seeds_path)

        # Load which URLs were already visited, so we don't refetch them across runs
        visited_data = read_json(visited_path, {"visited": []})
        visited_entries = visited_data.get("visited", [])

    # Read the running summary.
    summary_path = raw_pages_path.parent / "crawl_summary.json"
    zeroed_totals = {"runs_completed": 0, "attempted_urls_total": 0, "elapsed_seconds_total": 0.0, "interrupted_runs": 0}
    prior = zeroed_totals if fresh else read_json(summary_path, zeroed_totals)

    state = CrawlerState(
        frontier_high=frontier_high,
        frontier_low=frontier_low,
        pages=pages,
        visited_entries=visited_entries,
        max_pages=max_pages,
        polite_delay=polite_delay,
        checkpoint_every=checkpoint_every,
        domain_next_time=domain_next_time,
        elapsed_offset=prior.get("elapsed_seconds_total", 0.0),
        max_retries=max_retries,
    )

    total_workers = max(1, workers)
    num_explorers = max(0, min(explorer_workers, total_workers))
    # First `num_explorers` worker slots are Explorers, the rest are Horses.
    roles = ["explorer" if i < num_explorers else "horse" for i in range(total_workers)]

    logger.info(
        f"starting crawl: {len(frontier_high)} urls in frontier_high, {len(frontier_low)} urls in frontier_low, "
        f"{len(pages)} pages already saved, max_pages={max_pages}, workers={total_workers} "
        f"(horses={total_workers - num_explorers}, explorers={num_explorers}), fresh={fresh}"
    )

    checkpoint_paths = (raw_pages_path, frontier_path, visited_path)
    threads = [
        threading.Thread(
            target=_worker_loop,
            args=(state, i, roles[i], timeout, checkpoint_paths),
            name=f"crawler-worker-{i}-{roles[i]}",
            daemon=True,
        )
        for i in range(total_workers)
    ]

    for t in threads:
        t.start()

    try:
        # Poll instead of a plain join() so Ctrl-C reaches the main thread promptly even though the actual work is happening on daemon worker threads.
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.2)
    except KeyboardInterrupt:
        state.interrupted = True
        state.stop_event.set()
        logger.info("interrupted by user (Ctrl-C) - waiting for in-flight requests to finish, then saving progress so the crawl can be resumed later")
        # A worker's in-flight request can involve up to two hard-bounded fetches in sequence (robots.txt and the page itself).
        join_deadline = 2 * (timeout + 1) + 2
        for t in threads:
            t.join(timeout=join_deadline)
            if t.is_alive():
                # This worker is stuck past its request timeout
                logger.info(f"warning: {t.name} did not finish within {join_deadline:.0f}s, abandoning it (it may still be running in the background)")

    pages, frontier_high, frontier_low, domain_next_time, visited_entries = state.snapshot_for_checkpoint()

    # Persist crawler state to disk so subsequent runs can resume where this one left off.
    with state.checkpoint_lock:
        _save_state(
            raw_pages_path,
            frontier_path,
            visited_path,
            pages,
            frontier_high,
            frontier_low,
            domain_next_time,
            visited_entries,
        )
    elapsed_seconds = time.time() - start_time

    # Merge this run's counts into the running totals so far.
    summary = {
        "step": "crawling",
        "runs_completed": prior.get("runs_completed", 0) + 1,
        "total_pages": len(pages),
        "attempted_urls_this_run": state.attempted,
        "attempted_urls_total": prior.get("attempted_urls_total", 0) + state.attempted,
        "frontier_high_size": _frontier_total(frontier_high),
        "frontier_low_size": _frontier_total(frontier_low),
        "visited_size": len(visited_entries),
        "timeout": timeout,
        "polite_delay": polite_delay,
        "workers": total_workers,
        "explorer_workers": num_explorers,
        "horse_workers": total_workers - num_explorers,
        "last_run_interrupted": state.interrupted,
        "interrupted_runs": prior.get("interrupted_runs", 0) + (1 if state.interrupted else 0),
        "elapsed_seconds_this_run": round(elapsed_seconds, 2),
        "elapsed_human_this_run": _format_elapsed(elapsed_seconds),
        "elapsed_seconds_total": round(prior.get("elapsed_seconds_total", 0.0) + elapsed_seconds, 2),
        "elapsed_human_total": _format_elapsed(prior.get("elapsed_seconds_total", 0.0) + elapsed_seconds),
        "started_fresh_last_run": fresh,
        "last_updated": now_utc_iso(),
    }
    write_json(summary_path, summary)
    status = "interrupted" if state.interrupted else "done"
    logger.info(
        f"{status}: saved={state.saved} attempted={state.attempted} "
        f"frontier_high_remaining={_frontier_total(frontier_high)} frontier_low_remaining={_frontier_total(frontier_low)} "
        f"elapsed={summary['elapsed_human_this_run']}"
    )
    return summary
