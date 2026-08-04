import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.batch import run_batch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch retrieval."""
    parser = argparse.ArgumentParser(
        description="Run batch retrieval and write TSV results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_batch.py\n"
            "  python scripts/run_batch.py --top-k 100\n"
            "  python scripts/run_batch.py --no-reranking\n"
            "  python scripts/run_batch.py --assume-tuebingen-context\n"
            "  python scripts/run_batch.py --input queries.tsv --output data/results.tsv"
        ),
    )
    parser.add_argument("--input", default="queries.tsv", help="Path to the batch query TSV file.")
    parser.add_argument("--index", default="data/index.json", help="Path to the search index JSON file.")
    parser.add_argument("--output", default="data/results.tsv", help="Path for the official TSV result file.")
    parser.add_argument("--summary-output", default="data/batch_summary.json", help="Path for the internal batch summary JSON file.")
    parser.add_argument("--top-k", type=int, default=100, help="Maximum number of results per query.")
    parser.add_argument("--no-reranking", action="store_true", help="Disable second-stage reranking and output BM25 results directly.")
    parser.add_argument(
        "--assume-tuebingen-context",
        action="store_true",
        help="Append Tuebingen as an implicit context term when a query does not already contain it.",
    )
    return parser.parse_args()


def main() -> None:
    """Run batch retrieval from the command line."""
    args = parse_args()
    result = run_batch(
        input_path=args.input,
        index_path=args.index,
        output_path=args.output,
        summary_output_path=args.summary_output,
        top_k=args.top_k,
        use_reranking=not args.no_reranking,
        assume_tuebingen_context=args.assume_tuebingen_context,
    )
    summary = result["summary"]

    print("Batch retrieval complete")
    print(f"Queries: {summary['query_count']}")
    print(f"Total results: {summary['total_results']}")
    print(f"Average runtime seconds: {summary['average_runtime_seconds']}")
    print(f"Assume Tuebingen context: {summary['assume_tuebingen_context']}")
    print(f"Output: {args.output}")
    print(f"Summary output: {args.summary_output}")


if __name__ == "__main__":
    main()
