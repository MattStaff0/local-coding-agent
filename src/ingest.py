from rag import index_docs


def main() -> None:
    """Command-line entry point for rebuilding the local docs index."""
    # index_docs does the real RAG indexing work:
    # read docs -> chunk text -> embed chunks -> store chunks in ChromaDB.
    chunk_count = index_docs()

    if chunk_count:
        print(f"Done. Added {chunk_count} chunks to ChromaDB.")


if __name__ == "__main__":
    main()
