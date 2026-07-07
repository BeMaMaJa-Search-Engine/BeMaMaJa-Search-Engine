from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crawler import crawl


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl English Tuebingen-related web pages.")
    parser.add_argument(
        "--max-pages",
        dest="max_pages",
        type=int,
        default=1000,
        help=(
            "Maximum number of pages allowed in the saved pages file (raw_pages.json), "
            "counting pages already saved from previous runs. If the file already has "
            "this many pages (or more), the crawl stops immediately without fetching. "
            "Otherwise it keeps crawling until the file reaches this many pages."
        ),
    )
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--polite-delay",
        type=float,
        default=1.0,
        help="Minimum seconds between two requests to the same domain (subdomains share one timer). Not a global delay - different domains are fetched concurrently.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent worker threads fetching pages.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Write frontier/visited/pages to disk every N attempted fetches, so an interrupted run (Ctrl-C) can be resumed later instead of restarting.",
    )
    args = parser.parse_args()
    summary = crawl(
        max_pages=args.max_pages,
        timeout=args.timeout,
        polite_delay=args.polite_delay,
        checkpoint_every=args.checkpoint_every,
        workers=args.workers,
    )
    print(summary)


if __name__ == "__main__":
    main()
