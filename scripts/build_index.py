import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexer import build_index


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for index construction."""
    parser = argparse.ArgumentParser(description="Build the search index from preprocessed documents.")
    parser.add_argument("--input", default="data/preprocessed_pages.json", help="Path to preprocessed pages JSON.")
    parser.add_argument("--output", default="data/index.json", help="Path for the generated index JSON.")
    parser.add_argument(
        "--summary-output",
        default="data/index_summary.json",
        help="Path for the generated index summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_index(
        input_path=args.input,
        output_path=args.output,
        summary_output_path=args.summary_output,
    )
    summary = result["summary"]

    print("Indexing complete")
    print(f"Documents: {summary['num_docs']}")
    print(f"Vocabulary size: {summary['vocabulary_size']}")
    print(f"Average document length: {summary['average_document_length']}")
    print(f"Index file size MB: {summary['index_file_size_mb']}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']}")
    print(f"Output: {args.output}")
    print(f"Summary output: {args.summary_output}")


if __name__ == "__main__":
    main()
