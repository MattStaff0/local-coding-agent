import re
from pathlib import Path
from typing import Any

import chromadb
import chromadb.errors
import ollama


class EmptyIndexError(RuntimeError):
    """Raised when retrieval finds nothing — usually an empty or stale index."""


# These constants are the main knobs for the RAG system.
# Keeping them here makes rag.py the source of truth for model/database settings.
DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
COLLECTION_NAME = "local_docs"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5-coder:3b"
MAX_HISTORY_TURNS = 6


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


def embed(text: str) -> list[float]:
    """Ask Ollama's embedding model to turn text into a vector of numbers."""
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text,
    )
    return response["embedding"]


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

    return client.create_collection(name=COLLECTION_NAME)


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


def index_docs(docs_dir: Path = DOCS_DIR) -> int:
    """Read markdown docs, embed every chunk, and store them in Chroma.

    All reading, chunking, and embedding happens BEFORE the old collection is
    touched, so a failed run (bad file, Ollama down) never leaves an empty
    index behind.
    """
    # rglob picks up the per-source folders fetch_docs.py creates,
    # like docs/pytorch/ and docs/python/.
    doc_files = sorted(docs_dir.rglob("*.md"))

    if not doc_files:
        print(f"No docs found in {docs_dir}/ — keeping the existing index.")
        return 0

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for doc_file in doc_files:
        text = doc_file.read_text(encoding="utf-8")
        chunks = chunk_markdown(text)

        print(f"Processing {doc_file.name}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            # Each chunk needs its own id, breadcrumb-prefixed text, embedding,
            # and metadata.
            ids.append(chunk_id_for(doc_file, i, docs_dir))
            documents.append(chunk["text"])
            embeddings.append(embed(chunk["text"]))
            metadatas.append(
                {
                    "source": source_for(doc_file, docs_dir),
                    "path": str(doc_file),
                    "heading": chunk["heading"],
                    "chunk_index": i,
                }
            )

    # Everything embedded successfully — only now is it safe to swap the index.
    client = get_client()
    collection = reset_collection(client)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(documents)


def retrieve(
    question: str,
    n_results: int = 4,
    source: str | None = None,
) -> dict[str, Any]:
    """Find the most relevant indexed doc chunks for a user question.

    Pass a source name (a top-level docs/ folder) to search only that source.
    """
    client = get_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    # The question has to be embedded with the same model used for the docs.
    question_embedding = embed(question)

    return collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        where={"source": source} if source else None,
    )


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


def ask_model(prompt: str) -> str:
    """Send the completed prompt to the local Ollama chat model."""
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return response["message"]["content"]


def answer_question(
    question: str,
    history: list[dict[str, str]] | None = None,
    source: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the full RAG question-answer flow, optionally scoped to one source."""
    results = retrieve(question, source=source)

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

    prompt = build_prompt(question, docs, history or [], metadatas)
    answer = ask_model(prompt)

    return answer, metadatas
