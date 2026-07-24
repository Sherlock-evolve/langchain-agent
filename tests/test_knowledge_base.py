from __future__ import annotations

import pytest
from langchain_core.documents import Document

from knowledge_base import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    KnowledgeBaseBuilder,
    build_knowledge_base,
)


def make_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    docs = workspace / "docs"
    docs.mkdir(parents=True)
    return workspace, docs


def chunk_snapshot(result):
    return [
        (chunk.page_content, dict(chunk.metadata))
        for chunk in result.chunks
    ]


def test_build_is_deterministic_and_sources_are_posix_sorted(tmp_path):
    workspace, docs = make_workspace(tmp_path)
    nested = docs / "guides"
    nested.mkdir()
    (docs / "z-last.txt").write_text("last document", encoding="utf-8")
    (nested / "first.md").write_text("# First\n\nDetails", encoding="utf-8")
    (docs / "middle.txt").write_text("middle document", encoding="utf-8")

    first = build_knowledge_base(workspace)
    second = build_knowledge_base(workspace)

    expected_sources = [
        "docs/guides/first.md",
        "docs/middle.txt",
        "docs/z-last.txt",
    ]
    assert first.indexed_files == expected_sources
    assert [chunk.metadata["source"] for chunk in first.chunks] == (
        expected_sources
    )
    assert chunk_snapshot(first) == chunk_snapshot(second)
    assert all(isinstance(chunk, Document) for chunk in first.chunks)
    assert all(
        set(chunk.metadata)
        == {
            "source",
            "chunk_index",
            "start_index",
            "start_line",
            "end_line",
            "document_sha256",
            "chunk_id",
        }
        for chunk in first.chunks
    )
    assert all(
        len(chunk.metadata["chunk_id"]) == 64
        for chunk in first.chunks
    )


def test_chunks_have_overlap_start_positions_and_accurate_lines(tmp_path):
    workspace, docs = make_workspace(tmp_path)
    text = "".join(
        f"line-{index:04d} {'x' * 90}\n"
        for index in range(30)
    )
    (docs / "long.txt").write_text(text, encoding="utf-8")

    result = build_knowledge_base(workspace)

    assert len(result.chunks) > 1
    for index, chunk in enumerate(result.chunks):
        metadata = chunk.metadata
        start_index = metadata["start_index"]
        assert metadata["chunk_index"] == index
        assert (
            text[start_index : start_index + len(chunk.page_content)]
            == chunk.page_content
        )
        assert metadata["start_line"] == (
            text.count("\n", 0, start_index) + 1
        )
        last_index = start_index + len(chunk.page_content) - 1
        assert metadata["end_line"] == (
            text.count("\n", 0, last_index) + 1
        )

    for previous, current in zip(result.chunks, result.chunks[1:]):
        previous_end = (
            previous.metadata["start_index"] + len(previous.page_content)
        )
        overlap = previous_end - current.metadata["start_index"]
        assert overlap > 0
        assert previous.page_content[-overlap:] == (
            current.page_content[:overlap]
        )


def test_modifying_one_source_does_not_change_other_chunk_ids(tmp_path):
    workspace, docs = make_workspace(tmp_path)
    first_path = docs / "first.txt"
    second_path = docs / "second.txt"
    first_path.write_text("first version", encoding="utf-8")
    second_path.write_text("unchanged", encoding="utf-8")
    before = build_knowledge_base(workspace)

    first_path.write_text("second version", encoding="utf-8")
    after = build_knowledge_base(workspace)

    def ids_by_source(result):
        return {
            source: [
                chunk.metadata["chunk_id"]
                for chunk in result.chunks
                if chunk.metadata["source"] == source
            ]
            for source in result.indexed_files
        }

    before_ids = ids_by_source(before)
    after_ids = ids_by_source(after)
    assert before_ids["docs/first.txt"] != after_ids["docs/first.txt"]
    assert before_ids["docs/second.txt"] == after_ids["docs/second.txt"]


def test_unsafe_and_unreadable_sources_are_skipped_without_content(
    tmp_path,
):
    workspace, docs = make_workspace(tmp_path)
    (docs / ".hidden.txt").write_text(
        "hidden-secret-sentinel",
        encoding="utf-8",
    )
    hidden_directory = docs / ".hidden"
    hidden_directory.mkdir()
    (hidden_directory / "inside.txt").write_text(
        "hidden-directory-sentinel",
        encoding="utf-8",
    )
    (docs / "data.json").write_text(
        "unsupported-secret-sentinel",
        encoding="utf-8",
    )
    (docs / "invalid.txt").write_bytes(b"\xff\xfe")
    (docs / "large.txt").write_bytes(
        b"x" * (DEFAULT_MAX_FILE_SIZE_BYTES + 1)
    )
    outside = workspace / "outside.txt"
    outside.write_text("symlink-secret-sentinel", encoding="utf-8")
    (docs / "linked.txt").symlink_to(outside)
    (docs / "valid.md").write_text("safe", encoding="utf-8")

    result = build_knowledge_base(workspace)
    reports = {
        report.source: report.reason
        for report in result.skipped_files
    }

    assert result.indexed_files == ["docs/valid.md"]
    assert reports == {
        "docs/data.json": "unsupported_extension",
        "docs/invalid.txt": "invalid_utf8",
        "docs/large.txt": "file_too_large",
        "docs/linked.txt": "symlink",
    }
    report_text = repr(result.skipped_files)
    assert "secret-sentinel" not in report_text

    with pytest.raises(ValueError):
        build_knowledge_base(workspace, "../docs")
    with pytest.raises(ValueError):
        build_knowledge_base(workspace, docs.resolve())

    external_docs = tmp_path / "external-docs"
    external_docs.mkdir()
    (workspace / "linked-docs").symlink_to(
        external_docs,
        target_is_directory=True,
    )
    with pytest.raises(ValueError):
        build_knowledge_base(workspace, "linked-docs")


def test_global_file_byte_and_chunk_budgets_are_explicit(tmp_path):
    workspace, docs = make_workspace(tmp_path)
    (docs / "a.txt").write_text("a" * 10, encoding="utf-8")
    (docs / "b.txt").write_text("b" * 10, encoding="utf-8")

    file_limited = KnowledgeBaseBuilder(
        workspace,
        max_files=1,
    ).build()
    byte_limited = KnowledgeBaseBuilder(
        workspace,
        max_total_bytes=10,
    ).build()

    assert file_limited.truncated is True
    assert file_limited.indexed_files == ["docs/a.txt"]
    assert any(
        report.reason == "file_limit"
        for report in file_limited.skipped_files
    )
    assert byte_limited.truncated is True
    assert byte_limited.indexed_files == ["docs/a.txt"]
    assert any(
        report.reason == "total_bytes_limit"
        for report in byte_limited.skipped_files
    )

    chunk_workspace = tmp_path / "chunk-workspace"
    chunk_docs = chunk_workspace / "docs"
    chunk_docs.mkdir(parents=True)
    (chunk_docs / "long.txt").write_text("x" * 2_200, encoding="utf-8")
    chunk_limited = KnowledgeBaseBuilder(
        chunk_workspace,
        max_chunks=1,
    ).build()

    assert chunk_limited.truncated is True
    assert len(chunk_limited.chunks) == 1
    assert chunk_limited.indexed_files == ["docs/long.txt"]
    assert any(
        report.reason == "chunk_limit"
        for report in chunk_limited.skipped_files
    )
