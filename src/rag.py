import hashlib
import os
import re
from pathlib import Path
from typing import Any, Callable

import chromadb
import chromadb.errors
import httpx
import manifest as manifest_module
import ollama
from rank_bm25 import BM25Okapi


class EmptyIndexError(RuntimeError):
    """Raised when retrieval finds nothing — usually an empty or stale index."""


class NoRelevantDocsError(RuntimeError):
    """Raised when even the best retrieved chunk is too far from the question."""


# These constants are the main knobs for the RAG system.
# Keeping them here makes rag.py the source of truth for model/database settings.
# The model names read the environment first so switching models (3B on the
# laptop, 12B on the PC) never needs a code edit.
DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
MANIFEST_PATH = Path(os.getenv("RAG_MANIFEST_PATH", "manifest.jsonl"))
COLLECTION_NAME = "local_docs"
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5-coder:3b")
MAX_HISTORY_TURNS = 6

# Cosine distance (1 - cosine similarity, so 0 = identical, 2 = opposite)
# beyond which the best match is considered "nothing relevant". Requires the
# collection to be in cosine space; index_docs rebuilds a legacy L2 index
# automatically.
RELEVANCE_CUTOFF = float(os.getenv("RAG_RELEVANCE_CUTOFF", "0.65"))

# Hybrid retrieval: how many candidates each ranking contributes before
# fusion, and the standard RRF dampening constant.
HYBRID_CANDIDATES = 20
RRF_K = 60


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split long text into overlapping fixed-size chunks.

    Since chunk_markdown landed this is only the character-window fallback for
    a single paragraph that exceeds the chunk budget.
    """
    # An overlap >= chunk_size would make the loop step backwards (or spin
    # forever), so cap it at half a chunk.
    overlap = min(overlap, chunk_size // 2)

    chunks = []
    start = 0

    # The overlap keeps important text near a chunk boundary from being lost.
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_MARKERS = ("```", "~~~")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading breadcrumb, section body) pairs.

    Heading lines open a new section; `#` inside code fences is code, not a
    heading. The breadcrumb joins the active heading at each level, like
    "PyTorch Basics > Building a Model".
    """
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    body_lines: list[str] = []
    fence_marker: str | None = None

    def close_section() -> None:
        nonlocal body_lines
        sections.append((" > ".join(heading_stack), "\n".join(body_lines).strip()))
        body_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if fence_marker:
            body_lines.append(line)
            if stripped.startswith(fence_marker):
                fence_marker = None
            continue

        if stripped.startswith(_FENCE_MARKERS):
            fence_marker = stripped[:3]
            body_lines.append(line)
            continue

        match = _HEADING_RE.match(line)
        if match:
            close_section()
            level = len(match.group(1))
            heading_stack[:] = heading_stack[: level - 1] + [match.group(2)]
            continue

        body_lines.append(line)

    close_section()

    return [(path, body) for path, body in sections if body]


def _split_blocks(body: str) -> list[str]:
    """Split a section body into paragraph blocks; a code fence is one block."""
    blocks: list[str] = []
    current: list[str] = []
    fence_marker: str | None = None

    def close_block() -> None:
        nonlocal current
        if current:
            blocks.append("\n".join(current))
            current = []

    for line in body.splitlines():
        stripped = line.strip()

        if fence_marker:
            current.append(line)
            if stripped.startswith(fence_marker):
                fence_marker = None
                close_block()
            continue

        if stripped.startswith(_FENCE_MARKERS):
            close_block()
            fence_marker = stripped[:3]
            current.append(line)
            continue

        if not stripped:
            close_block()
            continue

        current.append(line)

    close_block()

    return blocks


def _pack_section(path: str, body: str, chunk_size: int) -> list[dict[str, str]]:
    """Pack one section's paragraphs into chunks that fit chunk_size.

    Exception: an oversized code fence is kept whole, so that one chunk may
    exceed chunk_size rather than shred a code example.
    """
    prefix = f"{path}\n\n" if path else ""
    budget = max(chunk_size - len(prefix), 1)

    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for block in _split_blocks(body):
        needed = len(block) + (2 if current else 0)

        if current and current_len + needed > budget:
            groups.append(current)

            # Carry the previous paragraph into the next chunk so context near
            # the boundary is not lost (paragraph-level overlap).
            last = current[-1]
            if (
                not last.lstrip().startswith(_FENCE_MARKERS)
                and len(last) + 2 + len(block) <= budget
            ):
                current, current_len = [last], len(last)
            else:
                current, current_len = [], 0

            needed = len(block) + (2 if current else 0)

        if not current and len(block) > budget:
            if block.lstrip().startswith(_FENCE_MARKERS):
                # Never split a code fence; an oversized one gets its own chunk.
                groups.append([block])
            else:
                for piece in chunk_text(block, chunk_size=budget):
                    groups.append([piece])
            continue

        current.append(block)
        current_len += needed

    if current:
        groups.append(current)

    return [
        {"heading": path, "text": prefix + "\n\n".join(group)} for group in groups
    ]


def _strip_frontmatter(text: str) -> str:
    """Drop leading YAML frontmatter blocks (fetch provenance, not doc content).

    Fetched raw-markdown docs can stack their own frontmatter right after the
    scraper's provenance block, so keep stripping while one leads the text.
    """
    while match := re.match(r"\A---\n.*?\n---\n\s*", text, flags=re.DOTALL):
        text = text[match.end() :]

    return text


def chunk_markdown(text: str, chunk_size: int = 1500) -> list[dict[str, str]]:
    """Chunk markdown by heading sections, keeping the breadcrumb in each chunk.

    Well-formatted docs split into one chunk per heading section so retrieval
    matches on headings instead of arbitrary character windows. Oversized
    sections fall back to paragraph-boundary packing with one-paragraph overlap.
    """
    text = _strip_frontmatter(text)
    chunks = []

    for path, body in _split_sections(text):
        chunks.extend(_pack_section(path, body, chunk_size))

    return chunks


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed many texts in one Ollama call — much faster than one call each."""
    response = ollama.embed(
        model=EMBED_MODEL,
        input=texts,
    )
    return [list(vector) for vector in response["embeddings"]]


def embed(text: str) -> list[float]:
    """Ask Ollama's embedding model to turn text into a vector of numbers."""
    return embed_batch([text])[0]


def get_client() -> chromadb.PersistentClient:
    """Open the local persistent Chroma database."""
    return chromadb.PersistentClient(path=DB_DIR)


def reset_collection(client: chromadb.PersistentClient):
    """Delete and recreate the docs collection so ingestion starts clean."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        # First run: there is no collection to delete yet. Any other error
        # (locked/corrupt database) must stay loud, not resurface later as a
        # confusing "collection already exists" on the create below.
        pass

    # Cosine space keeps query distances in a fixed 0..2 range regardless of
    # embedding model; the RELEVANCE_CUTOFF value itself may still need
    # retuning per model.
    return client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def source_for(doc_file: Path, docs_dir: Path = DOCS_DIR) -> str:
    """Name the source a doc belongs to: its top-level folder under docs/."""
    relative = doc_file.relative_to(docs_dir)

    # Files sitting directly in docs/ have no source folder.
    if len(relative.parts) == 1:
        return "general"

    return relative.parts[0]


def chunk_id_for(doc_file: Path, chunk_index: int, docs_dir: Path = DOCS_DIR) -> str:
    """Create a stable Chroma id for a document chunk."""
    relative_path = doc_file.relative_to(docs_dir).as_posix()
    safe_path = relative_path.replace("/", "__").replace("\\", "__")
    return f"{safe_path}-{chunk_index}"


def _file_hash(text: str) -> str:
    """Fingerprint used to decide whether a doc needs re-embedding.

    The embedding model is part of the fingerprint: switching
    OLLAMA_EMBED_MODEL must re-embed every file, or the index would keep the
    old model's vectors while queries use the new one.
    """
    return hashlib.sha256(f"{EMBED_MODEL}\n{text}".encode("utf-8")).hexdigest()


def _prepare_file(
    doc_file: Path, text: str, docs_dir: Path
) -> tuple[list[str], list[str], list[list[float]], list[dict[str, Any]]]:
    """Chunk and embed one doc file; returns (ids, documents, embeddings, metadatas)."""
    chunks = chunk_markdown(text)
    digest = _file_hash(text)

    print(f"Processing {doc_file.name}: {len(chunks)} chunks")

    if not chunks:
        # Heading-only/frontmatter-only files produce no chunks; calling
        # ollama.embed with an empty input is an endpoint error, not a no-op.
        print(f"Skipping {doc_file.name}: no indexable content")
        return [], [], [], []

    # One embedding call per file instead of per chunk keeps ingest fast
    # once the corpus is hundreds of pages.
    embeddings = embed_batch([chunk["text"] for chunk in chunks])
    relative_path = doc_file.relative_to(docs_dir).as_posix()

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(chunk_id_for(doc_file, i, docs_dir))
        documents.append(chunk["text"])
        metadatas.append(
            {
                "source": source_for(doc_file, docs_dir),
                "path": str(doc_file),
                "relative_path": relative_path,
                "heading": chunk["heading"],
                "chunk_index": i,
                "file_hash": digest,
            }
        )

    return ids, documents, embeddings, metadatas


def index_docs(docs_dir: Path = DOCS_DIR, full: bool = False) -> int:
    """Embed markdown docs into Chroma.

    Incremental by default: each file's fingerprint (content + embed model)
    is stored in its chunk metadata, so a re-ingest only embeds files that
    changed, adds new ones, and drops records for files deleted from disk;
    the return value counts only added/updated chunks. With full=True
    everything is re-embedded BEFORE the old collection is touched — a failed
    rebuild (bad file, Ollama down) never leaves an empty index behind — and
    the return value is the total chunk count.
    """
    # rglob picks up the per-source folders fetch_docs.py creates,
    # like docs/pytorch/ and docs/python/.
    doc_files = sorted(docs_dir.rglob("*.md"))

    if not doc_files:
        print(f"No docs found in {docs_dir}/ — keeping the existing index.")
        return 0

    if full:
        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []

        for doc_file in doc_files:
            text = doc_file.read_text(encoding="utf-8")
            file_ids, file_docs, file_embeddings, file_metadatas = _prepare_file(
                doc_file, text, docs_dir
            )
            ids.extend(file_ids)
            documents.extend(file_docs)
            embeddings.extend(file_embeddings)
            metadatas.extend(file_metadatas)

        # Everything embedded successfully — only now swap the index.
        collection = reset_collection(get_client())
        if ids:
            collection.add(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
            )

        rebuild_manifest(collection)
        return len(documents)

    collection = get_client().get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    # get_or_create ignores the metadata when the collection already exists,
    # so an index built before the cosine switch would silently stay in L2
    # space and make RELEVANCE_CUTOFF meaningless. Rebuild it once, loudly.
    if (collection.metadata or {}).get("hnsw:space") != "cosine":
        print("Existing index is not in cosine space — doing a full rebuild.")
        return index_docs(docs_dir, full=True)

    records = collection.get(include=["metadatas"])

    # Indexes built before relative_path landed can't be diffed reliably
    # (their keys are whatever cwd ingest ran from). Rebuild once, loudly.
    if any("relative_path" not in metadata for metadata in records["metadatas"]):
        print("Existing index predates relative-path metadata - doing a full rebuild.")
        return index_docs(docs_dir, full=True)

    stored_hashes: dict[str, str] = {}
    for metadata in records["metadatas"]:
        stored_hashes[metadata["relative_path"]] = metadata.get("file_hash", "")

    # Files that vanished from disk take their index records with them.
    on_disk = {doc_file.relative_to(docs_dir).as_posix() for doc_file in doc_files}
    for relative_path in stored_hashes:
        if relative_path not in on_disk:
            print(f"Removing indexed chunks for deleted {relative_path}")
            collection.delete(where={"relative_path": relative_path})

    added = 0

    for doc_file in doc_files:
        text = doc_file.read_text(encoding="utf-8")
        relative_path = doc_file.relative_to(docs_dir).as_posix()

        if stored_hashes.get(relative_path) == _file_hash(text):
            continue

        file_ids, file_docs, file_embeddings, file_metadatas = _prepare_file(
            doc_file, text, docs_dir
        )

        # Replace, not append: the old version may have had more chunks.
        collection.delete(where={"relative_path": relative_path})

        if file_ids:
            collection.add(
                ids=file_ids,
                documents=file_docs,
                embeddings=file_embeddings,
                metadatas=file_metadatas,
            )

        added += len(file_ids)

    rebuild_manifest(collection)
    return added


def rebuild_manifest(collection, path: Path | None = None) -> int:
    """Regenerate the manifest from the collection after ingest.

    One full collection scan per ingest buys zero scans per query.
    """
    path = path or MANIFEST_PATH
    records = collection.get(include=["documents", "metadatas"])

    rows = []
    for item_id, document, metadata in zip(
        records["ids"], records["documents"], records["metadatas"]
    ):
        rows.append(
            {
                "id": item_id,
                "relative_path": metadata["relative_path"],
                "source": metadata["source"],
                "heading": metadata["heading"],
                "file_hash": metadata["file_hash"],
                "approx_tokens": len(document) // 4,
                "tokens": _tokenize(document),
            }
        )

    manifest_module.write_manifest(rows, path)
    return len(rows)


# Stopwords never count as keyword evidence: without this, "how do I ..."
# matches something in nearly every chunk and BM25 hits stop meaning anything
# (which would also let junk chunks slip past the relevance refusal).
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "my", "of", "on", "or",
    "that", "the", "this", "to", "use", "what", "when", "where", "which",
    "who", "why", "with", "you",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens minus stopwords; `nn.Module` becomes nn, module."""
    return [
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if token not in _STOPWORDS
    ]


# Loaded once per process and reused across queries; the mtime check makes a
# fresh ingest visible without restarting chat.
_manifest_cache: dict[str, Any] = {"mtime": None, "records": [], "bm25": {}}
_warned_no_manifest = False


def _load_manifest_cached() -> list[dict[str, Any]]:
    try:
        mtime = MANIFEST_PATH.stat().st_mtime_ns
    except FileNotFoundError:
        _manifest_cache.update(mtime=None, records=[], bm25={})
        return []

    if _manifest_cache["mtime"] != mtime:
        _manifest_cache.update(
            mtime=mtime,
            records=manifest_module.load_manifest(MANIFEST_PATH),
            bm25={},
        )

    return _manifest_cache["records"]


def _bm25_for_source(source: str | None) -> tuple[BM25Okapi, list[str]] | None:
    records = _load_manifest_cached()
    subset = [record for record in records if source is None or record["source"] == source]

    if not subset:
        return None

    key = source or ""
    if key not in _manifest_cache["bm25"]:
        _manifest_cache["bm25"][key] = (
            BM25Okapi([record["tokens"] for record in subset]),
            [record["id"] for record in subset],
        )

    return _manifest_cache["bm25"][key]


def _bm25_rank_ids(question: str, bm25: BM25Okapi, ids: list[str]) -> list[str]:
    """Rank manifest chunk ids by BM25 keyword relevance to the question."""
    scores = bm25.get_scores(_tokenize(question))
    ranked = sorted(zip(ids, scores), key=lambda pair: pair[1], reverse=True)

    # A score of zero or below is treated as no keyword match (BM25's epsilon
    # floor can push common-term scores negative in a tiny corpus).
    return [item_id for item_id, score in ranked if score > 0][:HYBRID_CANDIDATES]


def _rrf_scores(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Fuse rankings with Reciprocal Rank Fusion: score = sum of 1/(k+rank).

    Rank-based fusion sidesteps the scale mismatch between BM25 scores and
    cosine distances entirely.
    """
    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    return scores


def _bm25_ranking(question: str, ids: list[str], documents: list[str]) -> list[str]:
    """Rank chunk ids by BM25 keyword relevance to the question."""
    bm25 = BM25Okapi([_tokenize(document) for document in documents])
    scores = bm25.get_scores(_tokenize(question))

    ranked = sorted(zip(ids, scores), key=lambda pair: pair[1], reverse=True)

    # A score of zero or below is treated as no keyword match (BM25's epsilon
    # floor can push common-term scores negative in a tiny corpus).
    return [item_id for item_id, score in ranked if score > 0][:HYBRID_CANDIDATES]


def _retrieve_hybrid_slow(
    question: str,
    collection,
    vector_results: dict[str, Any],
    where: dict[str, str] | None,
    n_results: int,
) -> dict[str, Any]:
    global _warned_no_manifest
    if not _warned_no_manifest:
        print("(no manifest - falling back to slow per-query BM25; re-run ingest to fix)")
        _warned_no_manifest = True

    corpus = collection.get(where=where, include=["documents", "metadatas"])

    if not corpus["ids"]:
        return vector_results

    vector_ids = vector_results["ids"][0]
    bm25_ids = _bm25_ranking(question, corpus["ids"], corpus["documents"])

    fused = _rrf_scores([vector_ids, bm25_ids])
    top_ids = sorted(fused, key=fused.get, reverse=True)[:n_results]

    by_id = {
        item_id: (document, metadata)
        for item_id, document, metadata in zip(
            corpus["ids"], corpus["documents"], corpus["metadatas"]
        )
    }
    # Distances stay honest: measured vector distances where known, the
    # cosine maximum (2.0) where a BM25-only hit was never measured. Keyword
    # relevance is reported separately as keyword_hits - a strong (top-3)
    # BM25 match is evidence of relevance even when the embedding disagrees,
    # and answer_question's refusal checks that signal explicitly.
    distance_by_id = dict(zip(vector_ids, vector_results["distances"][0]))
    strong_keyword_ids = set(bm25_ids[:3])

    return {
        "ids": [top_ids],
        "documents": [[by_id[item_id][0] for item_id in top_ids]],
        "metadatas": [[by_id[item_id][1] for item_id in top_ids]],
        "distances": [[distance_by_id.get(item_id, 2.0) for item_id in top_ids]],
        "keyword_hits": [[item_id in strong_keyword_ids for item_id in top_ids]],
    }


def retrieve(
    question: str,
    n_results: int = 4,
    source: str | None = None,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Find the most relevant indexed doc chunks for a user question.

    Pass a source name (a top-level docs/ folder) to search only that source.
    The default hybrid mode fuses vector and BM25 keyword rankings with RRF —
    embeddings catch paraphrases, BM25 catches exact identifiers; pass
    mode="vector" for pure semantic search.
    """
    if mode not in ("hybrid", "vector"):
        raise ValueError(f"Unknown retrieval mode '{mode}'; use 'hybrid' or 'vector'.")

    client = get_client()
    collection = client.get_collection(name=COLLECTION_NAME)
    where = {"source": source} if source else None

    # The question has to be embedded with the same model used for the docs.
    question_embedding = embed(question)

    vector_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=HYBRID_CANDIDATES if mode == "hybrid" else n_results,
        where=where,
    )

    if mode != "hybrid":
        return vector_results

    cached = _bm25_for_source(source)

    if cached is None:
        return _retrieve_hybrid_slow(
            question, collection, vector_results, where, n_results
        )

    bm25, manifest_ids = cached
    vector_ids = vector_results["ids"][0]
    bm25_ids = _bm25_rank_ids(question, bm25, manifest_ids)

    fused = _rrf_scores([vector_ids, bm25_ids])
    top_ids = sorted(fused, key=fused.get, reverse=True)[:n_results]

    fetched = collection.get(ids=top_ids, include=["documents", "metadatas"])
    by_id = {
        item_id: (document, metadata)
        for item_id, document, metadata in zip(
            fetched["ids"], fetched["documents"], fetched["metadatas"]
        )
    }
    # A manifest id can be stale for the few seconds between a collection
    # delete and the post-ingest manifest rebuild; drop ids Chroma no longer has.
    top_ids = [item_id for item_id in top_ids if item_id in by_id]

    # Distances stay honest: measured vector distances where known, the
    # cosine maximum (2.0) where a BM25-only hit was never measured. Keyword
    # relevance is reported separately as keyword_hits - a strong (top-3)
    # BM25 match is evidence of relevance even when the embedding disagrees,
    # and answer_question's refusal checks that signal explicitly.
    distance_by_id = dict(zip(vector_ids, vector_results["distances"][0]))
    strong_keyword_ids = set(bm25_ids[:3])

    return {
        "ids": [top_ids],
        "documents": [[by_id[item_id][0] for item_id in top_ids]],
        "metadatas": [[by_id[item_id][1] for item_id in top_ids]],
        "distances": [[distance_by_id.get(item_id, 2.0) for item_id in top_ids]],
        "keyword_hits": [[item_id in strong_keyword_ids for item_id in top_ids]],
    }


def list_sources() -> list[str]:
    """List the source names currently present in the index."""
    client = get_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    records = collection.get(include=["metadatas"])

    return sorted({metadata["source"] for metadata in records["metadatas"]})


def format_history(history: list[dict[str, str]]) -> str:
    """Convert recent chat history into plain text for the prompt."""
    if not history:
        return "No previous messages in this session."

    lines = []

    # Each turn has two messages: one user message and one assistant response.
    for message in history[-MAX_HISTORY_TURNS * 2 :]:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")

    return "\n".join(lines)


def chunk_label(number: int, metadata: dict[str, Any]) -> str:
    """Label one retrieved chunk, like '[1] docs/pytorch/tensors.md § Tensors'."""
    path = metadata.get("path", metadata.get("source", "unknown"))
    heading = metadata.get("heading", "")

    label = f"[{number}] {path}"
    if heading:
        label += f" § {heading}"

    return label


def source_legend(metadatas: list[dict[str, Any]]) -> list[str]:
    """Map citation numbers to their source file and heading, for display."""
    return [chunk_label(i, metadata) for i, metadata in enumerate(metadatas, start=1)]


def format_context(
    context_chunks: list[str],
    metadatas: list[dict[str, Any]] | None = None,
) -> str:
    """Join retrieved chunks, numbering and labeling them when metadata exists."""
    if not metadatas:
        return "\n\n---\n\n".join(context_chunks)

    blocks = []
    for i, (chunk, metadata) in enumerate(zip(context_chunks, metadatas), start=1):
        blocks.append(f"{chunk_label(i, metadata)}\n{chunk}")

    return "\n\n---\n\n".join(blocks)


def build_prompt(
    question: str,
    context_chunks: list[str],
    history: list[dict[str, str]] | None = None,
    metadatas: list[dict[str, Any]] | None = None,
) -> str:
    """Build the full prompt sent to the chat model."""
    context = format_context(context_chunks, metadatas)
    history_text = format_history(history or [])

    # This is where RAG happens: retrieved docs are inserted into the prompt.
    # The rules force doc-grounded answers: a small local model should teach
    # from the retrieved documentation, never from its own training data.
    return f"""
You are a local coding tutor. Answer the user's question using ONLY the numbered documentation context below.

Rules:
- Answer only from the documentation context. Do not use your own general knowledge.
- Cite the context you used with its number, like [1] or [2], right after each claim.
- If the context does not fully answer the question, say exactly what is missing or not covered, and do not guess.
- Be clear and practical.
- Include code examples when the context contains them.
- Use the conversation history only to understand follow-up questions.

Conversation history:
{history_text}

Documentation context:
{context}

User question:
{question}
""".strip()


def rewrite_query(question: str, history: list[dict[str, str]]) -> str:
    """Rewrite a follow-up like "how do I train it?" into a standalone query.

    Pronouns embed terribly without their referent, so retrieval for
    follow-ups works off a model-written standalone rewrite. Any failure
    falls back to the original question — a worse query beats no answer.
    """
    if not history:
        return question

    prompt = (
        "Rewrite the user's follow-up question as one standalone search "
        "query, using the conversation for missing context. Reply with ONLY "
        "the rewritten query.\n\n"
        f"Conversation:\n{format_history(history)}\n\n"
        f"Follow-up question: {question}"
    )

    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    except (httpx.HTTPError, ollama.ResponseError, ConnectionError, TimeoutError) as error:
        # Infrastructure hiccups degrade gracefully but never silently; a
        # malformed response shape is a bug and crashes loudly instead.
        print(f"(query rewrite failed: {error} — searching with the original question)")
        return question

    rewritten = response["message"]["content"].strip()

    return rewritten or question


def ask_model(prompt: str, on_token: Callable[[str], None] | None = None) -> str:
    """Send the completed prompt to the local Ollama chat model.

    The response is streamed; on_token (when given) receives each token as it
    arrives so the caller can print incrementally instead of waiting for a
    12B model to finish the whole answer.
    """
    parts: list[str] = []

    for part in ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=True,
    ):
        token = part["message"]["content"]
        parts.append(token)

        if on_token is not None:
            on_token(token)

    return "".join(parts)


def answer_question(
    question: str,
    history: list[dict[str, str]] | None = None,
    source: str | None = None,
    on_token: Callable[[str], None] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the full RAG question-answer flow, optionally scoped to one source."""
    search_query = rewrite_query(question, history) if history else question
    results = retrieve(search_query, source=source)

    # Chroma returns a list per query. We only send one query at a time, so [0].
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not docs:
        # Without this, the grounded prompt would make the model answer "the
        # docs don't cover this" to everything and the empty index stays hidden.
        raise EmptyIndexError(
            "Retrieval returned no chunks — the index may be empty or stale. "
            "Run 'python src/ingest.py' to rebuild it."
        )

    # Distances may be absent (test doubles or results that omit them); only
    # refuse when we can actually see that even the closest chunk is far from
    # the question AND no chunk got there on a strong keyword match.
    distances = (results.get("distances") or [[]])[0]
    keyword_hits = (results.get("keyword_hits") or [[]])[0]
    if distances and min(distances) > RELEVANCE_CUTOFF and not any(keyword_hits):
        raise NoRelevantDocsError(
            "Nothing relevant is indexed for that question — try /sources to "
            "see what's available, or add docs to sources.yaml and re-ingest."
        )

    prompt = build_prompt(question, docs, history or [], metadatas)
    answer = ask_model(prompt, on_token=on_token)

    return answer, metadatas
