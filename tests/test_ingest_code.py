"""Incremental code ingestion into the local_code collection."""
import json

import pytest

import manifest
import rag


@pytest.fixture()
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(
        rag, "embed_batch", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )


@pytest.fixture()
def mini_repo(tmp_path):
    repo = tmp_path / "myproject"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text(
        "def main():\n    return 1\n", encoding="utf-8"
    )
    (repo / "util.py").write_text(
        "class Helper:\n    def go(self):\n        return 2\n", encoding="utf-8"
    )
    cache = repo / "__pycache__"
    cache.mkdir()
    (cache / "junk.py").write_text("def skipped(): pass\n", encoding="utf-8")
    return repo


@pytest.fixture()
def code_env(tmp_path, monkeypatch, fake_embeddings):
    """Point the code index at throwaway chroma + manifest paths."""
    import chromadb

    db_dir = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(db_dir))
    monkeypatch.setattr(rag, "get_client", lambda: client)

    code_manifest = tmp_path / "code-manifest.jsonl"
    monkeypatch.setattr(rag, "CODE_MANIFEST_PATH", code_manifest)
    return client, code_manifest


def test_index_code_walks_chunks_and_writes_manifest(mini_repo, code_env):
    client, code_manifest = code_env

    added = rag.index_code(mini_repo)

    assert added == 2  # one chunk per file: main() and Helper
    collection = client.get_collection(rag.CODE_COLLECTION_NAME)
    records = collection.get(include=["metadatas"])

    paths = {m["relative_path"] for m in records["metadatas"]}
    assert paths == {"src/app.py", "util.py"}  # __pycache__ skipped
    assert all(m["source"] == "myproject" for m in records["metadatas"])
    assert all("start_line" in m for m in records["metadatas"])

    rows = [
        json.loads(line)
        for line in code_manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2


def test_reingest_is_a_noop(mini_repo, code_env):
    rag.index_code(mini_repo)
    assert rag.index_code(mini_repo) == 0


def test_edited_file_reembeds_only_that_file(mini_repo, code_env):
    rag.index_code(mini_repo)
    (mini_repo / "util.py").write_text(
        "class Helper:\n    def go(self):\n        return 3\n", encoding="utf-8"
    )
    assert rag.index_code(mini_repo) == 1


def test_second_repo_keeps_the_first(mini_repo, code_env, tmp_path):
    client, _ = code_env
    rag.index_code(mini_repo)

    other = tmp_path / "other"
    other.mkdir()
    (other / "solo.py").write_text("def solo():\n    return 9\n", encoding="utf-8")
    rag.index_code(other, repo_name="second")

    collection = client.get_collection(rag.CODE_COLLECTION_NAME)
    sources = {m["source"] for m in collection.get(include=["metadatas"])["metadatas"]}
    assert sources == {"myproject", "second"}
