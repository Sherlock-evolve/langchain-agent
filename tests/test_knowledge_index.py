import json

import pytest
from langchain_core.embeddings import Embeddings

from knowledge_index import (
    IncrementalEmbeddingIndex,
    KnowledgeIndexCacheError,
)
from knowledge_runtime import (
    KnowledgeEmbeddingError,
    create_knowledge_runtime,
)


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [
            [
                float(len(text)),
                float(sum(ord(character) for character in text) % 997),
            ]
            for text in texts
        ]

    def embed_query(self, text):
        self.query_calls.append(text)
        return [float(len(text)), 1.0]


def runtime(tmp_path, embeddings):
    return create_knowledge_runtime(
        workspace_root=tmp_path,
        knowledge_directory="docs",
        environment={
            "EMBEDDING_MODEL": "stable-test-model",
            "EMBEDDING_API_KEY": "test-key",
            "EMBEDDING_BASE_URL": "https://embedding.invalid/v1",
        },
        embeddings_factory=lambda **kwargs: embeddings,
    )


def test_incremental_index_reuses_unchanged_chunks_and_prunes_old(
    tmp_path,
):
    docs = tmp_path / "docs"
    docs.mkdir()
    first_path = docs / "first.txt"
    second_path = docs / "second.txt"
    first_path.write_text("alpha reference", encoding="utf-8")
    second_path.write_text("beta reference", encoding="utf-8")

    first_embeddings = CountingEmbeddings()
    first_runtime = runtime(tmp_path, first_embeddings)

    assert first_runtime.created_embedding_count == 2
    assert first_runtime.reused_embedding_count == 0
    assert len(first_embeddings.document_calls) == 1
    assert len(first_embeddings.document_calls[0]) == 2

    second_embeddings = CountingEmbeddings()
    second_runtime = runtime(tmp_path, second_embeddings)

    assert second_runtime.created_embedding_count == 0
    assert second_runtime.reused_embedding_count == 2
    assert second_embeddings.document_calls == []

    first_path.write_text("alpha reference changed", encoding="utf-8")
    third_embeddings = CountingEmbeddings()
    third_runtime = runtime(tmp_path, third_embeddings)

    assert third_runtime.created_embedding_count == 1
    assert third_runtime.reused_embedding_count == 1
    assert third_embeddings.document_calls == [
        ["alpha reference changed"]
    ]

    cache_files = list(
        (tmp_path / ".knowledge_index").glob("*.json")
    )
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 2
    raw_cache = cache_files[0].read_text(encoding="utf-8")
    assert "alpha reference" not in raw_cache
    assert "beta reference" not in raw_cache


def test_corrupt_or_symlinked_incremental_index_fails_closed(
    tmp_path,
):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.txt").write_text("guide", encoding="utf-8")
    first = runtime(tmp_path, CountingEmbeddings())
    assert first.created_embedding_count == 1

    cache_file = next(
        (tmp_path / ".knowledge_index").glob("*.json")
    )
    cache_file.write_text("{broken", encoding="utf-8")

    with pytest.raises(KnowledgeEmbeddingError):
        runtime(tmp_path, CountingEmbeddings())
    assert cache_file.read_text(encoding="utf-8") == "{broken"

    cache_file.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    cache_file.symlink_to(outside)
    with pytest.raises(KnowledgeEmbeddingError):
        runtime(tmp_path, CountingEmbeddings())
    assert outside.read_text(encoding="utf-8") == "{}"


def test_incremental_index_uses_bounded_cross_process_file_lock(
    tmp_path,
):
    fingerprint = "a" * 64
    index = IncrementalEmbeddingIndex(
        tmp_path,
        "docs",
        fingerprint,
        lock_timeout_seconds=0.02,
        lock_poll_seconds=0.005,
    )

    with index._file_lock():
        with pytest.raises(
            KnowledgeIndexCacheError,
            match="Timed out",
        ):
            IncrementalEmbeddingIndex(
                tmp_path,
                "docs",
                fingerprint,
                lock_timeout_seconds=0.02,
                lock_poll_seconds=0.005,
            )

    reopened = IncrementalEmbeddingIndex(
        tmp_path,
        "docs",
        fingerprint,
    )
    assert reopened.get("missing") is None
    assert reopened.lock_path.stat().st_mode & 0o777 == 0o600
