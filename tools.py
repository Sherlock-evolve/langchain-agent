import difflib
import hashlib
import os
import stat
import tempfile
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from langchain_core.tools import BaseTool, StructuredTool, tool

from contracts import PreparedToolAction, ToolActionConflictError


WORKSPACE_ROOT = Path(__file__).resolve().parent
_BOUND_WORKSPACE_ROOT: ContextVar[Path | None] = ContextVar(
    "workspace_agent_bound_root",
    default=None,
)
IGNORED_DIRECTORIES = {
    ".agent_audit",
    ".agent_sessions",
    ".git",
    ".knowledge_index",
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
MAX_FILE_SIZE_BYTES = 1024 * 1024
MAX_READ_LINES = 200
MAX_READ_CHARACTERS = 20_000
MAX_READ_BODY_CHARACTERS = 19_700
MAX_LIST_ENTRIES = 200
MAX_WRITE_PREVIEW_CHARACTERS = 8_000
WRITE_PREVIEW_TRUNCATION_MARKER = "\n[预览已截断]"


def _workspace_root() -> Path:
    return _BOUND_WORKSPACE_ROOT.get() or WORKSPACE_ROOT


def _validated_workspace_root(root: Path | str) -> Path:
    candidate = Path(root)
    if candidate.is_symlink():
        raise ValueError("工作区根目录不能是符号链接")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise ValueError("工作区根目录不存在") from None
    if not resolved.is_dir():
        raise ValueError("工作区根路径不是目录")
    return resolved


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

    workspace_root = _workspace_root()
    resolved_path = (workspace_root / relative_path).resolve()
    try:
        resolved_relative_path = resolved_path.relative_to(workspace_root)
    except ValueError as error:
        raise ValueError("路径超出当前项目目录") from error

    if _is_hidden_path(resolved_relative_path):
        raise ValueError("不允许访问被忽略或敏感的路径")

    return resolved_path


def _validate_write_content(content: str) -> bytes:
    try:
        encoded_content = content.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("写入内容不是有效的 UTF-8 文本") from error

    if len(encoded_content) > MAX_FILE_SIZE_BYTES:
        raise ValueError("写入内容超过 1 MB，拒绝写入")
    return encoded_content


def _resolve_write_target(path: str) -> Path:
    file_path = _resolve_workspace_path(path)
    if file_path.exists() and not file_path.is_file():
        raise IsADirectoryError(f"不是文件：{path}")
    if not file_path.parent.exists():
        raise FileNotFoundError(f"父目录不存在：{file_path.parent}")
    if not file_path.parent.is_dir():
        raise NotADirectoryError(f"父路径不是目录：{file_path.parent}")
    return file_path


def _file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_file(
    file_path: Path,
    display_path: str,
    content: str,
    encoded_content: bytes,
    before_replace: Callable[[], None] | None = None,
) -> str:
    file_exists = file_path.exists()
    status = "更新" if file_exists else "创建"
    existing_mode = (
        stat.S_IMODE(file_path.stat().st_mode)
        if file_exists
        else None
    )
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(encoded_content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        if existing_mode is not None:
            os.chmod(temporary_path, existing_mode)
        if before_replace is not None:
            before_replace()
        os.replace(temporary_path, file_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass

    return f"已{status} {display_path}，共 {len(content)} 个字符"


def _build_write_preview(
    file_path: Path,
    old_content: str,
    new_content: str,
) -> str:
    display_path = file_path.relative_to(_workspace_root()).as_posix()
    preview = "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{display_path}",
            tofile=f"b/{display_path}",
        )
    )
    if not preview:
        return "（文件内容无变化）"

    if len(preview) > MAX_WRITE_PREVIEW_CHARACTERS:
        body_limit = (
            MAX_WRITE_PREVIEW_CHARACTERS
            - len(WRITE_PREVIEW_TRUNCATION_MARKER)
        )
        preview = preview[:body_limit] + WRITE_PREVIEW_TRUNCATION_MARKER
    return preview


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
            relative_entry = resolved_entry.relative_to(_workspace_root())
        except ValueError:
            continue

        if _is_hidden_path(relative_entry):
            continue

        display_path = entry.relative_to(_workspace_root()).as_posix()
        if resolved_entry.is_dir():
            display_path += "/"
        entries.append(display_path)

    entries = sorted(entries)
    if len(entries) > MAX_LIST_ENTRIES:
        entries = entries[:MAX_LIST_ENTRIES]
        entries.append("结果已截断")

    return "\n".join(entries) or "（目录为空）"


@tool
def read_file(
    path: str,
    start_line: int = 1,
    line_count: int = 200,
) -> str:
    """分页读取项目内 UTF-8 文本文件；路径相对项目根目录，每次最多读取 200 行。"""
    if start_line < 1:
        raise ValueError("start_line 必须大于等于 1")
    if line_count < 1 or line_count > MAX_READ_LINES:
        raise ValueError(f"line_count 必须在 1 到 {MAX_READ_LINES} 之间")

    file_path = _resolve_workspace_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"不是文件：{path}")
    if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
        raise ValueError("文件超过 1 MB，拒绝读取")

    output_lines = []
    output_length = 0
    next_start_line = None
    truncated_line = None

    with file_path.open("r", encoding="utf-8") as file:
        for _ in range(start_line - 1):
            if file.readline() == "":
                return f"（从第 {start_line} 行起没有内容）"

        for offset in range(line_count):
            line_number = start_line + offset
            raw_line = file.readline()
            if raw_line == "":
                break

            formatted_line = f"{line_number}: {raw_line.rstrip(chr(13) + chr(10))}"
            separator_length = 1 if output_lines else 0
            remaining = (
                MAX_READ_BODY_CHARACTERS
                - output_length
                - separator_length
            )

            if len(formatted_line) > remaining:
                if output_lines:
                    next_start_line = line_number
                else:
                    output_lines.append(
                        formatted_line[: MAX_READ_BODY_CHARACTERS - 1] + "…"
                    )
                    output_length = len(output_lines[0])
                    truncated_line = line_number
                    if file.readline() != "":
                        next_start_line = line_number + 1
                break

            output_lines.append(formatted_line)
            output_length += separator_length + len(formatted_line)
        else:
            if file.readline() != "":
                next_start_line = start_line + line_count

    if not output_lines:
        return f"（从第 {start_line} 行起没有内容）"

    hints = []
    if truncated_line is not None:
        hints.append(f"[第 {truncated_line} 行内容过长，已截断]")
    if next_start_line is not None:
        last_shown_line = start_line + len(output_lines) - 1
        hints.append(
            f"[仅显示第 {start_line}–{last_shown_line} 行，"
            f"可使用 start_line={next_start_line} 继续读取]"
        )

    result = "\n".join([*output_lines, *hints])
    return result[:MAX_READ_CHARACTERS]


@tool
def write_file(path: str, content: str) -> str:
    """原子创建或更新项目内文本文件；写入前必须由 Agent 获得用户审批。"""
    file_path = _resolve_write_target(path)
    encoded_content = _validate_write_content(content)
    return _atomic_write_file(
        file_path=file_path,
        display_path=path,
        content=content,
        encoded_content=encoded_content,
    )


def prepare_write_file(path: str, content: str) -> PreparedToolAction:
    """准备带文件快照校验的写入操作，但不修改文件。"""
    file_path = _resolve_write_target(path)
    encoded_content = _validate_write_content(content)
    file_existed = file_path.exists()

    if file_existed:
        if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            raise ValueError("原文件超过 1 MB，拒绝生成写入预览")
        old_bytes = file_path.read_bytes()
        if len(old_bytes) > MAX_FILE_SIZE_BYTES:
            raise ValueError("原文件超过 1 MB，拒绝生成写入预览")
        old_content = old_bytes.decode("utf-8")
        expected_sha256 = hashlib.sha256(old_bytes).hexdigest()
    else:
        old_content = ""
        expected_sha256 = None

    preview = _build_write_preview(
        file_path=file_path,
        old_content=old_content,
        new_content=content,
    )

    def validate_snapshot() -> None:
        try:
            current_path = _resolve_write_target(path)
        except Exception as error:
            raise ToolActionConflictError(
                f"写入目标状态已变化：{error}。"
            ) from error

        if current_path != file_path:
            raise ToolActionConflictError("写入目标解析位置已变化。")

        current_exists = current_path.exists()
        if current_exists != file_existed:
            if file_existed:
                detail = "审批时文件存在，但执行前已不存在。"
            else:
                detail = "审批时文件不存在，但执行前已被创建。"
            raise ToolActionConflictError(detail)

        if file_existed:
            try:
                current_sha256 = _file_sha256(current_path)
            except OSError as error:
                raise ToolActionConflictError(
                    f"无法确认当前文件状态：{error}。"
                ) from error
            if current_sha256 != expected_sha256:
                raise ToolActionConflictError(
                    "文件内容在审批后发生变化。"
                )

    def execute() -> str:
        validate_snapshot()
        return _atomic_write_file(
            file_path=file_path,
            display_path=path,
            content=content,
            encoded_content=encoded_content,
            before_replace=validate_snapshot,
        )

    return PreparedToolAction(
        preview=preview,
        execute=execute,
    )


@tool
def preview_write_file(path: str, content: str) -> str:
    """预览 write_file 将产生的统一 diff；不会修改文件。"""
    return prepare_write_file(path=path, content=content).preview


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
                relative_child = child_directory.resolve().relative_to(
                    _workspace_root()
                )
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
                relative_file = resolved_file.relative_to(_workspace_root())
            except (OSError, ValueError):
                continue

            if _is_hidden_path(relative_file):
                continue
            if not resolved_file.is_file():
                continue

            try:
                if resolved_file.stat().st_size > MAX_FILE_SIZE_BYTES:
                    continue
                content = resolved_file.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue

            display_path = file_path.relative_to(
                _workspace_root()
            ).as_posix()
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


@dataclass(frozen=True)
class WorkspaceToolBundle:
    """Workspace-bound tools and their approval preparers."""

    tools: tuple[BaseTool, ...]
    approval_preparers: dict[str, Callable[..., PreparedToolAction]]


def create_workspace_tool_bundle(
    workspace_root: Path | str,
) -> WorkspaceToolBundle:
    """Create concurrency-safe tools bound to one tenant workspace."""

    root = _validated_workspace_root(workspace_root)

    def run_bound(function, *args, **kwargs):
        context_token = _BOUND_WORKSPACE_ROOT.set(root)
        try:
            return function(*args, **kwargs)
        finally:
            _BOUND_WORKSPACE_ROOT.reset(context_token)

    def bound_list_files(directory: str = ".") -> str:
        """列出租户工作区指定目录的直接子项。"""
        return run_bound(list_files.func, directory=directory)

    def bound_read_file(
        path: str,
        start_line: int = 1,
        line_count: int = 200,
    ) -> str:
        """读取租户工作区文本文件的指定行范围。"""
        return run_bound(
            read_file.func,
            path=path,
            start_line=start_line,
            line_count=line_count,
        )

    def bound_write_file(path: str, content: str) -> str:
        """原子创建或更新租户工作区文本文件。"""
        return run_bound(
            write_file.func,
            path=path,
            content=content,
        )

    def bound_search_text(
        query: str,
        directory: str = ".",
    ) -> str:
        """在租户工作区中递归搜索文本。"""
        return run_bound(
            search_text.func,
            query=query,
            directory=directory,
        )

    def bound_prepare_write_file(
        path: str,
        content: str,
    ) -> PreparedToolAction:
        prepared = run_bound(
            prepare_write_file,
            path=path,
            content=content,
        )

        def execute() -> str:
            return run_bound(prepared.execute)

        return PreparedToolAction(
            preview=prepared.preview,
            execute=execute,
        )

    bound_tools = (
        StructuredTool.from_function(
            func=bound_list_files,
            name="list_files",
            description=list_files.description,
        ),
        StructuredTool.from_function(
            func=bound_read_file,
            name="read_file",
            description=read_file.description,
        ),
        StructuredTool.from_function(
            func=bound_search_text,
            name="search_text",
            description=search_text.description,
        ),
        StructuredTool.from_function(
            func=bound_write_file,
            name="write_file",
            description=write_file.description,
        ),
    )
    return WorkspaceToolBundle(
        tools=bound_tools,
        approval_preparers={
            "write_file": bound_prepare_write_file,
        },
    )
