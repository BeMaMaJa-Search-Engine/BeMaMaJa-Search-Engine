from __future__ import annotations

import heapq
from collections import defaultdict

from src.utils import get_domain


def _result_domain(result: dict, position: int) -> str:
    """Return a shared domain key, or a unique key if no valid URL is available."""
    url = (
        result.get("canonical_url")
        or result.get("fetched_url")
        or result.get("url")
        or ""
    )
    domain = get_domain(str(url))
    return domain or f"__unknown_domain_{position}"


def _result_score(result: dict) -> float:
    """Return the score used as the basis for diversification."""
    try:
        return float(result.get("score", result.get("bm25_score", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _penalty_factor(
    selected_domain_count: int,
    unpenalized_results: int,
    decay: float,
    minimum_factor: float,
) -> float:
    """Increase the soft penalty only after a domain already has enough results."""
    repeat_count = max(0, selected_domain_count - unpenalized_results + 1)
    return max(minimum_factor, decay**repeat_count)


def diversify_results(
    results: list[dict],
    unpenalized_results: int = 1,
    decay: float = 0.93,
    minimum_factor: float = 0.75,
) -> list[dict]:
    """Diversify a ranking with a progressive soft penalty per domain.

    The first ``unpenalized_results`` from a domain keep their full score. Each
    later result receives an increasingly stronger discount. This avoids hard
    domain caps, so a highly relevant page can still remain near the top.
    """
    if unpenalized_results < 0:
        raise ValueError("unpenalized_results must be non-negative")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be greater than 0 and at most 1")
    if not 0.0 <= minimum_factor <= 1.0:
        raise ValueError("minimum_factor must be between 0 and 1")

    domain_results: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
    for position, result in enumerate(results):
        domain = _result_domain(result, position)
        domain_results[domain].append((_result_score(result), position, result))

    for domain in domain_results:
        domain_results[domain].sort(key=lambda item: (-item[0], item[1]))

    selected_counts: dict[str, int] = defaultdict(int)
    domain_offsets: dict[str, int] = defaultdict(int)
    heap: list[tuple[float, int, str, float]] = []

    for domain, domain_queue in domain_results.items():
        score, position, _ = domain_queue[0]
        factor = _penalty_factor(0, unpenalized_results, decay, minimum_factor)
        heapq.heappush(heap, (-(score * factor), position, domain, factor))

    diversified: list[dict] = []
    while heap:
        negative_score, _, domain, factor = heapq.heappop(heap)
        offset = domain_offsets[domain]
        original_score, _, original_result = domain_results[domain][offset]

        result = original_result.copy()
        result["score_before_diversity"] = original_score
        result["domain_diversity_factor"] = round(factor, 4)
        result["score"] = round(-negative_score, 4)

        score_details = dict(result.get("score_details", {}))
        score_details["score_before_diversity"] = round(original_score, 4)
        score_details["domain_diversity_factor"] = round(factor, 4)
        result["score_details"] = score_details
        diversified.append(result)

        selected_counts[domain] += 1
        domain_offsets[domain] += 1
        next_offset = domain_offsets[domain]
        if next_offset < len(domain_results[domain]):
            next_score, next_position, _ = domain_results[domain][next_offset]
            next_factor = _penalty_factor(
                selected_counts[domain],
                unpenalized_results,
                decay,
                minimum_factor,
            )
            heapq.heappush(
                heap,
                (-(next_score * next_factor), next_position, domain, next_factor),
            )

    for rank, result in enumerate(diversified, start=1):
        result["rank"] = rank

    return diversified
