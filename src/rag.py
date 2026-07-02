from pathlib import Path
from typing import Any

import chromadb
import ollama


# These constants are the main knobs for the RAG system.
# Keeping them here makes rag.py the source of truth for model/database settings.
DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
COLLECTION_NAME = "local_docs"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5-coder:3b"
MAX_HISTORY_TURNS = 6


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split a long document into overlapping chunks for embedding."""
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
    except Exception:
        # Chroma raises if the collection does not exist yet. That is fine on
        # the first run because there is nothing to delete.
        pass

    return client.create_collection(name=COLLECTION_NAME)


def source_for(doc_file: Path, docs_dir: Path = DOCS_DIR) -> str:
    """Name the source collection a doc belongs to: its top folder under docs/."""
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
    """Read markdown docs, embed every chunk, and store them in Chroma."""
    client = get_client()
    collection = reset_collection(client)

    # rglob lets us support docs in folders later, like docs/python/venv.md.
    doc_files = sorted(docs_dir.rglob("*.md"))

    if not doc_files:
        print(f"No docs found in {docs_dir}/")
        return 0

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for doc_file in doc_files:
        text = doc_file.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        print(f"Processing {doc_file.name}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            # Each chunk needs its own id, original text, embedding, and metadata.
            ids.append(chunk_id_for(doc_file, i, docs_dir))
            documents.append(chunk)
            embeddings.append(embed(chunk))
            metadatas.append(
                {
                    "source": source_for(doc_file, docs_dir),
                    "path": str(doc_file),
                    "chunk_index": i,
                }
            )

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


def build_prompt(
    question: str,
    context_chunks: list[str],
    history: list[dict[str, str]] | None = None,
) -> str:
    """Build the full prompt sent to the chat model."""
    context = "\n\n---\n\n".join(context_chunks)
    history_text = format_history(history or [])

    # This is where RAG happens: retrieved docs are inserted into the prompt.
    return f"""
You are a local coding tutor. Answer the user's question using the provided documentation context.

Rules:
- Use the context first.
- If the context does not contain the answer, say what is missing.
- Be clear and practical.
- Include code examples when helpful.
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

    prompt = build_prompt(question, docs, history or [])
    answer = ask_model(prompt)

    return answer, metadatas
