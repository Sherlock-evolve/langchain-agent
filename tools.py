import os
from pathlib import Path, PureWindowsPath

from langchain_core.tools import tool


WORKSPACE_ROOT = Path(__file__).resolve().parent
IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
MAX_SEARCH_RESULTS = 50
MAX_SEARCH_LINE_LENGTH = 300


def _is_sensitive_name(name: str) -> bool:
    lowercase_name = name.lower()
    return (
        lowercase_name == ".env"
        or lowercase_name.startswith(".env.")
        or lowercase_name.endswith(".pem")
        or lowercase_name.endswith(".key")
    )


def _is_hidden_path(path: Path) -> bool:
    return any(
        part in IGNORED_DIRECTORIES or _is_sensitive_name(part)
        for part in path.parts
    )


def _resolve_workspace_path(path: str) -> Path:
    """校验并解析工作区内的相对路径。"""
    relative_path = Path(path)
    windows_path = PureWindowsPath(path)

    if (
        relative_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.root)
    ):
        raise ValueError("不允许使用绝对路径")

    if ".." in relative_path.parts or ".." in windows_path.parts:
        raise ValueError("不允许使用 '..' 进行路径穿越")

    if _is_hidden_path(relative_path):
        raise ValueError("不允许访问被忽略或敏感的路径")

    resolved_path = (WORKSPACE_ROOT / relative_path).resolve()
    try:
        resolved_relative_path = resolved_path.relative_to(WORKSPACE_ROOT)
    except ValueError as error:
        raise ValueError("路径超出当前项目目录") from error

    if _is_hidden_path(resolved_relative_path):
        raise ValueError("不允许访问被忽略或敏感的路径")

    return resolved_path


@tool
def list_files(directory: str = ".") -> str:
    """列出当前项目指定目录的直接子项；目录必须是项目根目录下的相对路径。"""
    directory_path = _resolve_workspace_path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"目录不存在：{directory}")
    if not directory_path.is_dir():
        raise NotADirectoryError(f"不是目录：{directory}")

    entries = []
    for entry in directory_path.iterdir():
        if entry.name in IGNORED_DIRECTORIES or _is_sensitive_name(entry.name):
            continue

        resolved_entry = entry.resolve()
        try:
            relative_entry = resolved_entry.relative_to(WORKSPACE_ROOT)
        except ValueError:
            continue

        if _is_hidden_path(relative_entry):
            continue

        display_path = entry.relative_to(WORKSPACE_ROOT).as_posix()
        if resolved_entry.is_dir():
            display_path += "/"
        entries.append(display_path)

    return "\n".join(sorted(entries)) or "（目录为空）"


@tool
def read_file(path: str) -> str:
    """读取当前项目内指定 UTF-8 文本文件；文件必须使用项目根目录下的相对路径。"""
    file_path = _resolve_workspace_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"不是文件：{path}")

    return file_path.read_text(encoding="utf-8")


@tool
def search_text(query: str, directory: str = ".") -> str:
    """在当前项目的指定目录中递归搜索文本；目录必须是项目内的相对路径。"""
    if not query.strip():
        raise ValueError("搜索内容不能为空")

    directory_path = _resolve_workspace_path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"目录不存在：{directory}")
    if not directory_path.is_dir():
        raise NotADirectoryError(f"不是目录：{directory}")

    results = []
    results_truncated = False
    result_limit_reached = False

    for current_directory, directory_names, file_names in os.walk(
        directory_path,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current_directory)
        allowed_directories = []

        for directory_name in directory_names:
            child_directory = current_path / directory_name
            if (
                directory_name in IGNORED_DIRECTORIES
                or _is_sensitive_name(directory_name)
                or child_directory.is_symlink()
            ):
                continue

            try:
                relative_child = child_directory.resolve().relative_to(WORKSPACE_ROOT)
            except (OSError, ValueError):
                continue

            if _is_hidden_path(relative_child):
                continue
            allowed_directories.append(directory_name)

        directory_names[:] = sorted(allowed_directories)

        for file_name in sorted(file_names):
            file_path = current_path / file_name
            if _is_sensitive_name(file_name):
                continue

            try:
                resolved_file = file_path.resolve()
                relative_file = resolved_file.relative_to(WORKSPACE_ROOT)
            except (OSError, ValueError):
                continue

            if _is_hidden_path(relative_file):
                continue
            if not resolved_file.is_file():
                continue

            try:
                content = resolved_file.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue

            display_path = file_path.relative_to(WORKSPACE_ROOT).as_posix()
            for line_number, line in enumerate(content.splitlines(), start=1):
                if query not in line:
                    continue

                if len(results) >= MAX_SEARCH_RESULTS:
                    results_truncated = True
                    result_limit_reached = True
                    break

                result = f"{display_path}:{line_number}:{line}"
                if len(result) > MAX_SEARCH_LINE_LENGTH:
                    result = result[:MAX_SEARCH_LINE_LENGTH]
                    results_truncated = True
                results.append(result)

            if result_limit_reached:
                break

        if result_limit_reached:
            break

    if not results:
        return "未找到匹配结果"

    if results_truncated:
        results.append("结果已截断")

    return "\n".join(results)
