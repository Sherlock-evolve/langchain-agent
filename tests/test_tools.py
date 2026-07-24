from threading import Thread

import pytest

import tools as workspace_tools


def use_temporary_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(
        workspace_tools,
        "WORKSPACE_ROOT",
        tmp_path.resolve(),
    )


def test_read_file_paginates_with_line_numbers(tmp_path, monkeypatch):
    use_temporary_workspace(monkeypatch, tmp_path)
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "\n".join(f"line-{line_number}" for line_number in range(1, 251)),
        encoding="utf-8",
    )

    first_page = workspace_tools.read_file.invoke({"path": "sample.txt"})
    second_page = workspace_tools.read_file.invoke(
        {
            "path": "sample.txt",
            "start_line": 201,
            "line_count": 50,
        }
    )

    assert first_page.startswith("1: line-1\n")
    assert "200: line-200" in first_page
    assert "201: line-201" not in first_page
    assert (
        "[仅显示第 1–200 行，可使用 start_line=201 继续读取]"
        in first_page
    )
    assert second_page.startswith("201: line-201\n")
    assert second_page.endswith("250: line-250")
    assert "继续读取" not in second_page


def test_read_file_rejects_invalid_ranges(
    tmp_path,
    monkeypatch,
):
    use_temporary_workspace(monkeypatch, tmp_path)
    (tmp_path / "sample.txt").write_text("content", encoding="utf-8")

    invalid_arguments = [
        {"path": "sample.txt", "start_line": 0},
        {"path": "sample.txt", "start_line": -1},
        {"path": "sample.txt", "line_count": 0},
        {"path": "sample.txt", "line_count": 201},
    ]
    for arguments in invalid_arguments:
        with pytest.raises(ValueError):
            workspace_tools.read_file.invoke(arguments)


def test_large_files_and_outputs_are_limited(tmp_path, monkeypatch):
    use_temporary_workspace(monkeypatch, tmp_path)
    oversized_file = tmp_path / "oversized.txt"
    oversized_file.write_bytes(
        b"oversize-needle\n"
        + b"x" * workspace_tools.MAX_FILE_SIZE_BYTES
    )
    long_file = tmp_path / "long.txt"
    long_file.write_text(
        "\n".join("x" * 1000 for _ in range(100)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="1 MB"):
        workspace_tools.read_file.invoke({"path": "oversized.txt"})

    long_result = workspace_tools.read_file.invoke({"path": "long.txt"})
    search_result = workspace_tools.search_text.invoke(
        {"query": "oversize-needle"}
    )

    assert len(long_result) <= workspace_tools.MAX_READ_CHARACTERS
    assert "继续读取" in long_result
    assert search_result == "未找到匹配结果"

    many_files = tmp_path / "many"
    many_files.mkdir()
    for index in range(205):
        (many_files / f"file-{index:03}.txt").write_text(
            "content",
            encoding="utf-8",
        )

    listed_entries = workspace_tools.list_files.invoke(
        {"directory": "many"}
    ).splitlines()
    assert len(listed_entries[:-1]) == workspace_tools.MAX_LIST_ENTRIES
    assert listed_entries[-1] == "结果已截断"


def test_sensitive_files_cannot_be_read_or_searched(
    tmp_path,
    monkeypatch,
):
    use_temporary_workspace(monkeypatch, tmp_path)
    sensitive_names = [
        ".env",
        ".env.local",
        "certificate.pem",
        "private.key",
    ]
    for name in sensitive_names:
        (tmp_path / name).write_text(
            "sensitive-needle",
            encoding="utf-8",
        )
    (tmp_path / "public.txt").write_text("public", encoding="utf-8")

    for name in sensitive_names:
        with pytest.raises(ValueError):
            workspace_tools.read_file.invoke({"path": name})

    listing = workspace_tools.list_files.invoke({}).splitlines()
    search_result = workspace_tools.search_text.invoke(
        {"query": "sensitive-needle"}
    )

    assert listing == ["public.txt"]
    assert search_result == "未找到匹配结果"


def test_bound_workspace_tools_isolate_concurrent_tenants(tmp_path):
    alice_root = tmp_path / "alice"
    bob_root = tmp_path / "bob"
    alice_root.mkdir()
    bob_root.mkdir()
    (alice_root / "note.txt").write_text("alice", encoding="utf-8")
    (bob_root / "note.txt").write_text("bob", encoding="utf-8")
    alice = workspace_tools.create_workspace_tool_bundle(alice_root)
    bob = workspace_tools.create_workspace_tool_bundle(bob_root)
    alice_read = next(tool for tool in alice.tools if tool.name == "read_file")
    bob_read = next(tool for tool in bob.tools if tool.name == "read_file")
    results = {}

    first = Thread(
        target=lambda: results.setdefault(
            "alice",
            alice_read.invoke({"path": "note.txt"}),
        )
    )
    second = Thread(
        target=lambda: results.setdefault(
            "bob",
            bob_read.invoke({"path": "note.txt"}),
        )
    )
    first.start()
    second.start()
    first.join(timeout=1)
    second.join(timeout=1)

    assert results["alice"].endswith("alice")
    assert results["bob"].endswith("bob")

    prepared = alice.approval_preparers["write_file"](
        path="note.txt",
        content="alice-updated",
    )
    prepared.execute()
    assert (alice_root / "note.txt").read_text(
        encoding="utf-8"
    ) == "alice-updated"
    assert (bob_root / "note.txt").read_text(encoding="utf-8") == "bob"
