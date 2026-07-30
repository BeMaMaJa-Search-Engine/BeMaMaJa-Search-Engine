from __future__ import annotations

import hashlib
from collections import defaultdict

SIMHASH_BITS = 64
DEFAULT_BANDS = 4  # 64 / 4 = 16 bits per band
DEFAULT_HAMMING_THRESHOLD = 3  # near-duplicate if <= this many bits differ
DEFAULT_SHINGLE_SIZE = 3

# hamming_threshold must be < num_bands for the LSH banding


def _token_shingles(tokens: list[str], n: int = DEFAULT_SHINGLE_SIZE) -> list[str]:
    """Turn a token list into overlapping n-grams (shingles) for simhash input."""
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _hash_feature(feature: str) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def compute_simhash(tokens: list[str], n: int = DEFAULT_SHINGLE_SIZE) -> int:
    """Compute a 64-bit simhash fingerprint for a page's tokens."""
    shingles = _token_shingles(tokens, n=n)
    if not shingles:
        return 0

    weights = [0] * SIMHASH_BITS
    for shingle in shingles:
        h = _hash_feature(shingle)
        for bit in range(SIMHASH_BITS):
            weights[bit] += 1 if (h >> bit) & 1 else -1

    fingerprint = 0
    for bit in range(SIMHASH_BITS):
        if weights[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _bands(fingerprint: int, num_bands: int) -> list[int]:
    """Split a fingerprint into `num_bands` equal chunks of bits."""
    bits_per_band = SIMHASH_BITS // num_bands
    mask = (1 << bits_per_band) - 1
    return [(fingerprint >> (i * bits_per_band)) & mask for i in range(num_bands)]


def _exact_key(tokens: list[str]) -> str:
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def _choose_canonical(pages: list[dict]) -> dict:
    """Pick a representative page out of a duplicate cluster.

    Prefers a page whose `canonical_url` points at itself, then falls back to the lowest doc_id for determinism.
    """
    for page in pages:
        if page.get("canonical_url") and page.get("canonical_url") == page.get("url"):
            return page
    return min(pages, key=lambda p: p.get("doc_id", 0))


def deduplicate_pages(
    pages: list[dict],
    hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD,
    num_bands: int = DEFAULT_BANDS,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> tuple[list[dict], dict]:
    """Remove exact- and near-duplicate pages from a list of crawled pages.

    Returns (kept_pages, report). The report lists every cluster that got
    collapsed and what was removed.
    """
    from src.preprocessing import preprocess

    if not pages:
        empty_report = {
            "input_pages": 0,
            "exact_duplicates_removed": 0,
            "near_duplicates_removed": 0,
            "output_pages": 0,
            "hamming_threshold": hamming_threshold,
            "num_bands": num_bands,
            "shingle_size": shingle_size,
            "exact_duplicate_clusters": [],
            "near_duplicate_clusters": [],
        }
        return [], empty_report

    # Precompute tokens/fingerprints/exact-keys once per page.
    fingerprints: dict[int, int] = {}
    exact_keys: dict[int, str] = {}
    for page in pages:
        text = f"{page.get('title', '')} {page.get('body', '')}"
        tokens = preprocess(text, use_stemming=True)
        doc_id = page["doc_id"]
        fingerprints[doc_id] = compute_simhash(tokens, n=shingle_size)
        exact_keys[doc_id] = _exact_key(tokens)

    # Pass 1: exact duplicates
    exact_groups: dict[str, list[dict]] = defaultdict(list)
    for page in pages:
        exact_groups[exact_keys[page["doc_id"]]].append(page)

    survivors: list[dict] = []
    exact_clusters_report = []
    for group in exact_groups.values():
        if len(group) == 1:
            survivors.append(group[0])
            continue
        canonical = _choose_canonical(group)
        survivors.append(canonical)
        exact_clusters_report.append(
            {
                "canonical_doc_id": canonical.get("doc_id"),
                "canonical_url": canonical.get("url"),
                "removed": [
                    {"doc_id": p.get("doc_id"), "url": p.get("url")}
                    for p in group
                    if p is not canonical
                ],
            }
        )

    # Pass 2: near duplicates via simhash + LSH banding
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for page in survivors:
        fp = fingerprints[page["doc_id"]]
        for band_idx, band_val in enumerate(_bands(fp, num_bands)):
            buckets[(band_idx, band_val)].append(page)

    # Union-find to merge pages connected via any shared bucket + real hamming check.
    parent: dict[int, int] = {p["doc_id"]: p["doc_id"] for p in survivors}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for bucket_pages in buckets.values():
        if len(bucket_pages) < 2:
            continue
        for i in range(len(bucket_pages)):
            for j in range(i + 1, len(bucket_pages)):
                a, b = bucket_pages[i], bucket_pages[j]
                dist = hamming_distance(fingerprints[a["doc_id"]], fingerprints[b["doc_id"]])
                if dist <= hamming_threshold:
                    union(a["doc_id"], b["doc_id"])

    clusters: dict[int, list[dict]] = defaultdict(list)
    for page in survivors:
        clusters[find(page["doc_id"])].append(page)

    kept: list[dict] = []
    near_clusters_report = []
    for cluster_pages in clusters.values():
        if len(cluster_pages) == 1:
            kept.append(cluster_pages[0])
            continue
        canonical = _choose_canonical(cluster_pages)
        kept.append(canonical)
        near_clusters_report.append(
            {
                "canonical_doc_id": canonical.get("doc_id"),
                "canonical_url": canonical.get("url"),
                "removed": [
                    {"doc_id": p.get("doc_id"), "url": p.get("url")}
                    for p in cluster_pages
                    if p is not canonical
                ],
            }
        )

    report = {
        "input_pages": len(pages),
        "exact_duplicates_removed": sum(len(c["removed"]) for c in exact_clusters_report),
        "near_duplicates_removed": sum(len(c["removed"]) for c in near_clusters_report),
        "output_pages": len(kept),
        "hamming_threshold": hamming_threshold,
        "num_bands": num_bands,
        "shingle_size": shingle_size,
        "exact_duplicate_clusters": exact_clusters_report,
        "near_duplicate_clusters": near_clusters_report,
    }
    return kept, report


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from src.utils import read_json, write_json

    parser = argparse.ArgumentParser(
        description="Detect and remove duplicate/near-duplicate crawled pages."
    )
    parser.add_argument("--input", default="../data/raw_pages.json", type=Path)
    parser.add_argument("--output", default="../data/raw_pages_deduped.json", type=Path)
    parser.add_argument("--report", default="../data/dedup_report.json", type=Path)
    parser.add_argument("--hamming-threshold", type=int, default=DEFAULT_HAMMING_THRESHOLD)
    parser.add_argument("--num-bands", type=int, default=DEFAULT_BANDS)
    parser.add_argument("--shingle-size", type=int, default=DEFAULT_SHINGLE_SIZE)
    args = parser.parse_args()

    raw = read_json(args.input, {"pages": []})
    kept_pages, dedup_report = deduplicate_pages(
        raw.get("pages", []),
        hamming_threshold=args.hamming_threshold,
        num_bands=args.num_bands,
        shingle_size=args.shingle_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"pages": kept_pages})
    write_json(args.report, dedup_report)

    print(
        f"dedup: {dedup_report['input_pages']} pages -> {dedup_report['output_pages']} "
        f"({dedup_report['exact_duplicates_removed']} exact + "
        f"{dedup_report['near_duplicates_removed']} near duplicates removed)"
    )
    print(f"wrote deduped pages to {args.output}")
    print(f"wrote report to {args.report}")
