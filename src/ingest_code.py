"""Index a Python repo into the local_code collection.

Usage:
    python src/ingest_code.py <repo-path> [--name myproject] [--full]

The repo name (folder name by default) becomes the chunk source, so /code
can scope answers per repo later via the existing source machinery.
"""
import argparse
from pathlib import Path

from rag import index_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Path to the repository to index")
    parser.add_argument("--name", help="Source name (defaults to the folder name)")
    parser.add_argument(
        "--full", action="store_true", help="Re-embed every file, not just changed ones"
    )
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()

    if not repo.is_dir():
        raise SystemExit(f"No such directory: {args.repo}")

    added = index_code(repo, repo_name=args.name, full=args.full)
    print(f"Indexed {added} added/updated chunks from {repo}")


if __name__ == "__main__":
    main()
