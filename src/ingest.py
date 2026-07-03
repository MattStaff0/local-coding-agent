import sys

from rag import index_docs


def main() -> None:
    """Command-line entry point for updating the local docs index.

    Usage: python src/ingest.py [--full]
    Incremental by default (only changed/new/removed files); --full rebuilds
    the whole collection from scratch.
    """
    full = "--full" in sys.argv[1:]

    # index_docs does the real RAG indexing work:
    # read docs -> chunk text -> embed chunks -> store chunks in ChromaDB.
    chunk_count = index_docs(full=full)

    if chunk_count:
        verb = "Rebuilt with" if full else "Added/updated"
        print(f"Done. {verb} {chunk_count} chunks in ChromaDB.")
    else:
        print("Index already up to date.")


if __name__ == "__main__":
    main()
