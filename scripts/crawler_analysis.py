import contextlib
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from urllib.parse import urlparse
from collections import Counter


# Add your crawler runs here
RUNS = {
    "baseline": "../data_baseline/raw_pages.json",
    "Regex": "../data_regex/raw_pages.json",
    "Regex + Harsh frontier": "../data_regex_harsh/raw_pages.json",
    "Regex + Harsh frontier + blacklist + mail cut in link": "../data_at-cut/raw_pages.json",
    "... + Horses & Explorers": "../data/raw_pages.json",
}

# Domains with more than two labels get collapsed to their registrable domain.
TWO_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "net.uk",
    "co.jp", "ne.jp", "or.jp",
    "com.au", "net.au", "org.au",
    "co.nz", "co.za", "co.in", "co.at", "or.at",
    "com.br", "com.mx", "com.tr", "com.cn",
}

OUTPUT_FILE = "crawler-analysis.txt"
TOP_N_PAGES = 10
TOP_N_FRONTIER = 50

# If the gap between two consecutive timestamps exceeds this, we assume
# the crawler was stopped/restarted rather than actually stalling, and
# drop that interval from the timing stats so it doesn't show up as a
# huge spike/bump in the rolling-average graphs.
MAX_GAP_SECONDS = 120


def extract_domain(url: str) -> str:
    """
    Extract the host from a URL, stripping any userinfo (e.g. the "u.cress@" in "https://u.cress@iwm-tuebingen.de/...") and any
    port, since urlparse().netloc includes both.
    """
    if not url:
        return ""

    netloc = urlparse(url).netloc.lower()

    if not netloc:
        return ""

    # Drop "user:pass@" if present.
    netloc = netloc.split("@")[-1]
    # Drop ":port" if present.
    netloc = netloc.split(":")[0]

    return netloc


def base_domain(domain: str) -> str:
    """
    Collapse a host down to its registrable domain so that all subdomains (and stray userinfo-derived hosts) of the same site get grouped together. Example:
        u.cress@iwm-tuebingen.de -> iwm-tuebingen.de
        www.iwm-tuebingen.de     -> iwm-tuebingen.de
        iwm-tuebingen.de         -> iwm-tuebingen.de
    """
    if not domain:
        return domain

    parts = domain.split(".")

    if len(parts) <= 2:
        return domain

    last_two = ".".join(parts[-2:])
    last_three = ".".join(parts[-3:])

    if last_two in TWO_PART_SUFFIXES and len(parts) >= 3:
        return last_three

    return last_two


def clean_frontier_domain(key: str) -> str:
    """
    Normalize a frontier.json domain key into a bare, grouped domain.
    Handles bare hosts, full URLs, and defensively handles
    markdown-style "[label](url)" artifacts if they show up in the data.
    """
    if not key:
        return ""

    key = key.strip()

    if key.startswith("[") and "](" in key:
        key = key.split("](", 1)[1].rstrip(")")

    if "://" in key:
        domain = extract_domain(key)
    else:
        domain = key.lower().split("@")[-1].split(":")[0].strip("/")

    return base_domain(domain)


def load_pages(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pages", [])


def load_frontier(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_time(ts: str):
    """
    Parse timestamps like:
    2026-07-12T08:54:45Z
    """
    return datetime.fromisoformat(
        ts.replace("Z", "+00:00")
    )


def filtered_intervals(times, max_gap: float = MAX_GAP_SECONDS):
    """
    Turns a list of sorted timestamps into consecutive-difference intervals (in seconds), dropping any gap larger than max_gap.
    """
    intervals = [
        (times[i] - times[i - 1]).total_seconds()
        for i in range(1, len(times))
    ]

    return [iv for iv in intervals if iv <= max_gap]


def get_intervals(path: str):
    """
    Returns the time differences (in seconds) between consecutive discovered pages.
    Gaps larger than MAX_GAP_SECONDS (assumed to be crawler restarts/downtime) are excluded.
    """
    pages = load_pages(Path(path))

    times = [
        parse_time(page["crawl_time"])
        for page in pages
        if page.get("crawl_time")
    ]

    if len(times) < 2:
        return pd.Series(dtype=float)

    intervals = filtered_intervals(times)

    return pd.Series(intervals)


def moving_average_intervals(
    path: str,
    window: int = 10,
):
    """
    Rolling average of the page discovery intervals.
    """
    intervals = get_intervals(path)

    if intervals.empty:
        return intervals

    return intervals.rolling(
        window=window,
        min_periods=window,
    ).mean()


def moving_average_new_domain_time(
    path: str,
    window: int = 10,
):
    """
    Computes the moving average of the time between discoveries of previously unseen domains.
    """
    pages = load_pages(Path(path))

    seen_domains = set()
    new_domain_times = []

    for page in pages:
        url = (
            page.get("canonical_url")
            or page.get("url", "")
        )

        domain = base_domain(extract_domain(url))

        if not domain:
            continue

        if domain not in seen_domains:
            seen_domains.add(domain)

            if page.get("crawl_time"):
                new_domain_times.append(
                    parse_time(page["crawl_time"])
                )

    if len(new_domain_times) < 2:
        return pd.Series(dtype=float)

    intervals = filtered_intervals(new_domain_times)

    return pd.Series(intervals).rolling(
        window=window,
        min_periods=window,
    ).mean()


def plot_new_domain_discovery(window: int = 10):
    plt.figure(figsize=(12, 6))

    for name, path in RUNS.items():
        rolling = moving_average_new_domain_time(
            path,
            window=window,
        )

        plt.plot(
            rolling.index,
            rolling.values,
            label=name,
            linewidth=2,
        )

    plt.xlabel(
        "Number of discovered domains"
    )
    plt.ylabel(
        f"Average seconds between new domains "
        f"(window={window})"
    )
    plt.title(
        "Time Between Discovering New Domains"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_discovery_rate(window: int = 10):
    """
    Plot average seconds between discovered pages.
    """
    plt.figure(figsize=(12, 6))

    for name, path in RUNS.items():
        rolling = moving_average_intervals(
            path,
            window=window,
        )

        plt.plot(
            rolling.index,
            rolling.values,
            label=name,
            linewidth=2,
        )

    plt.xlabel("Page number")
    plt.ylabel(
        f"Average seconds between pages "
        f"(window={window})"
    )
    plt.title(
        "Moving Average Time Between "
        "Discovered Pages"
    )

    # Logarithmic y-axis
    plt.yscale("log")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pages_per_minute(window: int = 10):
    plt.figure(figsize=(12, 6))

    for name, path in RUNS.items():
        rolling = moving_average_intervals(
            path,
            window=window,
        )

        throughput = 60 / rolling

        plt.plot(
            throughput.index,
            throughput.values,
            label=name,
            linewidth=2,
        )

    plt.xlabel("Page number")
    plt.ylabel(
        f"Pages per minute (window={window})"
    )
    plt.title("Crawler Throughput Over Time")

    # Logarithmic y-axis
    plt.yscale("log")

    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_run_stats(top_n: int = TOP_N_PAGES):
    print("\n===== Crawl Statistics =====\n")

    for name, path in RUNS.items():
        pages = load_pages(Path(path))

        # Parse timestamps
        times = [
            parse_time(page["crawl_time"])
            for page in pages
            if page.get("crawl_time")
        ]

        total_pages = len(pages)

        if len(times) >= 2:
            # Use "active" crawl time rather than raw wall-clock time, so pauses/restarts afflict deflate pages/minute.
            active_duration_seconds = sum(filtered_intervals(times))

            pages_per_minute = (
                total_pages / (active_duration_seconds / 60)
                if active_duration_seconds > 0
                else float("inf")
            )
        else:
            active_duration_seconds = 0
            pages_per_minute = 0

        # Count domains (subdomains/userinfo variants grouped together)
        domains = Counter()

        for page in pages:
            url = (
                page.get("canonical_url")
                or page.get("url", "")
            )

            domain = base_domain(extract_domain(url))

            if domain:
                domains[domain] += 1

        print(f"{name}")
        print(f"  Pages found: {total_pages}")
        print(
            f"  Pages/minute: {pages_per_minute:.2f}"
        )
        print(f"  Top {top_n} domains:")

        for domain, count in domains.most_common(top_n):
            print(f"    {domain:<35} {count}")

        print()


def frontier_domain_counts(path: str) -> Counter:
    """
    Counts frontier_high entries per (grouped) domain, entry-wise.
    frontier_low is intentionally not in here since the crawler hasn't touched it once in my runs.
    """
    data = load_frontier(Path(path))

    counts = Counter()

    section_data = data.get("frontier_high", {})

    for key, urls in section_data.items():
        domain = clean_frontier_domain(key)

        if domain:
            counts[domain] += len(urls)

    return counts


def print_frontier_stats(top_n: int = TOP_N_FRONTIER):
    """
    Extends the crawl statistics with a per-run breakdown of the frontier.json, counting entries per domain and printing the top N.
    """
    print("\n===== Frontier Statistics (entry-wise) =====\n")

    for name, raw_path in RUNS.items():
        frontier_file = Path(raw_path).parent / "frontier.json"

        print(f"{name}")

        if not frontier_file.exists():
            print(f"  (no frontier.json found at {frontier_file})\n")
            continue

        counts = frontier_domain_counts(str(frontier_file))
        total_entries = sum(counts.values())
        total_domains = len(counts)

        print(f"  Total frontier entries: {total_entries}")
        print(f"  Total distinct domains: {total_domains}")
        print(f"  Top {top_n} domains:")

        for domain, count in counts.most_common(top_n):
            print(f"    {domain:<35} {count}")

        print()


def write_report(output_path: str = OUTPUT_FILE):
    """
    Runs all the text-based stats and writes them to a single report file instead of the console.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            print_run_stats()
            print_frontier_stats()

    print(f"Report written to {output_path}")


def main():
    plot_discovery_rate(window=50)
    plot_pages_per_minute(window=50)
    plot_new_domain_discovery(window=50)
    write_report()


if __name__ == "__main__":
    main()
