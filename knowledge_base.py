from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_DOCUMENTS_DIRECTORY = "docs"
SUPPORTED_EXTENSIONS = frozenset({".md", ".txt"})
IGNORED_DIRECTORIES = frozenset(
    {
        ".agent_audit",
        ".agent_sessions",
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
)

DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_MAX_FILE_SIZE_BYTES = 1024 * 1024
DEFAULT_MAX_FILES = 500
DEFAULT_MAX_TOTAL_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_CHUNKS = 5_000


@dataclass(frozen=True)
class SkippedFileReport:
    """A content-free explanation of why a source was not indexed."""

    source: str
    reason: str


@dataclass(frozen=True)
class KnowledgeBuildResult:
    chunks: list[Document]
    indexed_files: list[str]
    skipped_files: list[SkippedFileReport]
    truncated: bool


class KnowledgeBaseBuilder:
    """Build a deterministic, traceable corpus from a workspace document tree."""

    def __init__(
        self,
        workspace_root: str | os.PathLike[str] = WORKSPACE_ROOT,
        docs_directory: str | os.PathLike[str] = DEFAULT_DOCUMENTS_DIRECTORY,
        *,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        max_files: int = DEFAULT_MAX_FILES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
    ) -> None:
        self.workspace_root = self._validate_workspace_root(workspace_root)
        self.docs_root = self._validate_docs_root(docs_directory)
        self.max_file_size_bytes = self._positive_limit(
            "max_file_size_bytes",
            max_file_size_bytes,
        )
        self.max_files = self._positive_limit("max_files", max_files)
        self.max_total_bytes = self._positive_limit(
            "max_total_bytes",
            max_total_bytes,
        )
        self.max_chunks = self._positive_limit("max_chunks", max_chunks)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            add_start_index=True,
        )

    @staticmethod
    def _positive_limit(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _validate_workspace_root(
        workspace_root: str | os.PathLike[str],
    ) -> Path:
        root_path = Path(workspace_root)
        if root_path.is_symlink():
            raise ValueError("workspace_root must not be a symbolic link")
        try:
            resolved_root = root_path.resolve(strict=True)
        except OSError as error:
            raise ValueError("workspace_root does not exist") from error
        if not resolved_root.is_dir():
            raise ValueError("workspace_root must be a directory")
        return resolved_root

    def _validate_docs_root(
        self,
        docs_directory: str | os.PathLike[str],
    ) -> Path:
        raw_directory = os.fspath(docs_directory)
        if not raw_directory:
            raise ValueError("docs_directory must not be empty")

        relative_path = Path(raw_directory)
        windows_path = PureWindowsPath(raw_directory)
        if (
            relative_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.root)
        ):
            raise ValueError("docs_directory must be relative to workspace_root")
        if ".." in relative_path.parts or ".." in windows_path.parts:
            raise ValueError("docs_directory must not contain '..'")
        if any(self._is_ignored_name(part) for part in relative_path.parts):
            raise ValueError("docs_directory is hidden or ignored")

        unresolved_root = self.workspace_root / relative_path
        current_path = self.workspace_root
        for part in relative_path.parts:
            if part in {"", "."}:
                continue
            current_path /= part
            if current_path.is_symlink():
                raise ValueError("docs_directory must not contain symbolic links")

        try:
            resolved_root = unresolved_root.resolve(strict=True)
            resolved_root.relative_to(self.workspace_root)
        except (OSError, ValueError) as error:
            raise ValueError(
                "docs_directory must be an existing workspace directory"
            ) from error
        if not resolved_root.is_dir():
            raise ValueError("docs_directory must be a directory")
        return resolved_root

    @staticmethod
    def _is_ignored_name(name: str) -> bool:
        return name.startswith(".") or name in IGNORED_DIRECTORIES

    def _source_for(self, path: Path) -> str:
        return path.relative_to(self.workspace_root).as_posix()

    def _discover_sources(
        self,
    ) -> tuple[list[tuple[str, Path]], list[SkippedFileReport]]:
        candidates: list[tuple[str, Path]] = []
        skipped: list[SkippedFileReport] = []
        pending_directories = [self.docs_root]

        while pending_directories:
            directory = pending_directories.pop()
            try:
                entries = sorted(
                    os.scandir(directory),
                    key=lambda entry: entry.name,
                    reverse=True,
                )
            except OSError:
                skipped.append(
                    SkippedFileReport(
                        source=self._source_for(directory),
                        reason="directory_read_error",
                    )
                )
                continue

            for entry in entries:
                if self._is_ignored_name(entry.name):
                    continue

                entry_path = Path(entry.path)
                source = self._source_for(entry_path)
                try:
                    if entry.is_symlink():
                        skipped.append(
                            SkippedFileReport(source=source, reason="symlink")
                        )
                    elif entry.is_dir(follow_symlinks=False):
                        pending_directories.append(entry_path)
                    elif entry.is_file(follow_symlinks=False):
                        if entry_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                            candidates.append((source, entry_path))
                        else:
                            skipped.append(
                                SkippedFileReport(
                                    source=source,
                                    reason="unsupported_extension",
                                )
                            )
                    else:
                        skipped.append(
                            SkippedFileReport(
                                source=source,
                                reason="not_regular_file",
                            )
                        )
                except OSError:
                    skipped.append(
                        SkippedFileReport(
                            source=source,
                            reason="metadata_read_error",
                        )
                    )

        candidates.sort(key=lambda item: item[0])
        return candidates, skipped

    def _read_source(
        self,
        path: Path,
        remaining_total_bytes: int,
    ) -> tuple[bytes | None, str | None]:
        try:
            if path.is_symlink():
                return None, "symlink"
            resolved_path = path.resolve(strict=True)
            resolved_path.relative_to(self.workspace_root)
            if resolved_path != path:
                return None, "symlink"
        except ValueError:
            return None, "outside_workspace"
        except OSError:
            return None, "read_error"

        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        file_descriptor: int | None = None
        try:
            file_descriptor = os.open(path, flags)
            file_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                return None, "not_regular_file"
            if file_stat.st_size > self.max_file_size_bytes:
                return None, "file_too_large"
            if file_stat.st_size > remaining_total_bytes:
                return None, "total_bytes_limit"

            read_limit = min(
                self.max_file_size_bytes,
                remaining_total_bytes,
            )
            with os.fdopen(file_descriptor, "rb") as source_file:
                file_descriptor = None
                content = source_file.read(read_limit + 1)

            if len(content) > self.max_file_size_bytes:
                return None, "file_too_large"
            if len(content) > remaining_total_bytes:
                return None, "total_bytes_limit"
            return content, None
        except OSError as error:
            if error.errno == errno.ELOOP:
                return None, "symlink"
            return None, "read_error"
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)

    @staticmethod
    def _line_number(text: str, character_index: int) -> int:
        return text.count("\n", 0, character_index) + 1

    @staticmethod
    def _chunk_id(
        source: str,
        document_sha256: str,
        start_index: int,
        page_content: str,
    ) -> str:
        identity = json.dumps(
            [source, document_sha256, start_index, page_content],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(identity).hexdigest()

    def _split_source(
        self,
        source: str,
        content: str,
        document_sha256: str,
    ) -> list[Document]:
        split_documents = self._splitter.create_documents([content])
        chunks: list[Document] = []

        for chunk_index, split_document in enumerate(split_documents):
            start_index = split_document.metadata.get("start_index")
            if (
                isinstance(start_index, bool)
                or not isinstance(start_index, int)
                or start_index < 0
            ):
                raise ValueError("text splitter did not provide a start_index")

            page_content = split_document.page_content
            start_line = self._line_number(content, start_index)
            if page_content:
                final_character_index = start_index + len(page_content) - 1
                end_line = self._line_number(content, final_character_index)
            else:
                end_line = start_line

            chunks.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": source,
                        "chunk_index": chunk_index,
                        "start_index": start_index,
                        "start_line": start_line,
                        "end_line": end_line,
                        "document_sha256": document_sha256,
                        "chunk_id": self._chunk_id(
                            source,
                            document_sha256,
                            start_index,
                            page_content,
                        ),
                    },
                )
            )
        return chunks

    def build(self) -> KnowledgeBuildResult:
        candidates, skipped = self._discover_sources()
        chunks: list[Document] = []
        indexed_files: list[str] = []
        total_bytes = 0
        attempted_files = 0
        truncated = False

        for source, path in candidates:
            if len(chunks) >= self.max_chunks:
                skipped.append(
                    SkippedFileReport(source=source, reason="chunk_limit")
                )
                truncated = True
                break
            if attempted_files >= self.max_files:
                skipped.append(
                    SkippedFileReport(source=source, reason="file_limit")
                )
                truncated = True
                break
            attempted_files += 1

            raw_content, read_error = self._read_source(
                path,
                self.max_total_bytes - total_bytes,
            )
            if read_error is not None:
                skipped.append(
                    SkippedFileReport(source=source, reason=read_error)
                )
                if read_error == "total_bytes_limit":
                    truncated = True
                    break
                continue
            assert raw_content is not None
            total_bytes += len(raw_content)

            try:
                content = raw_content.decode("utf-8")
            except UnicodeDecodeError:
                skipped.append(
                    SkippedFileReport(source=source, reason="invalid_utf8")
                )
                continue

            document_sha256 = hashlib.sha256(raw_content).hexdigest()
            try:
                source_chunks = self._split_source(
                    source,
                    content,
                    document_sha256,
                )
            except (TypeError, ValueError):
                skipped.append(
                    SkippedFileReport(source=source, reason="split_error")
                )
                continue

            remaining_chunks = self.max_chunks - len(chunks)
            indexed_files.append(source)
            if len(source_chunks) > remaining_chunks:
                chunks.extend(source_chunks[:remaining_chunks])
                skipped.append(
                    SkippedFileReport(source=source, reason="chunk_limit")
                )
                truncated = True
                break
            chunks.extend(source_chunks)

        skipped.sort(key=lambda report: (report.source, report.reason))
        return KnowledgeBuildResult(
            chunks=chunks,
            indexed_files=indexed_files,
            skipped_files=skipped,
            truncated=truncated,
        )


def build_knowledge_base(
    workspace_root: str | os.PathLike[str] = WORKSPACE_ROOT,
    docs_directory: str | os.PathLike[str] = DEFAULT_DOCUMENTS_DIRECTORY,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> KnowledgeBuildResult:
    """Build and return a deterministic snapshot of the local document corpus."""

    return KnowledgeBaseBuilder(
        workspace_root=workspace_root,
        docs_directory=docs_directory,
        max_file_size_bytes=max_file_size_bytes,
        max_files=max_files,
        max_total_bytes=max_total_bytes,
        max_chunks=max_chunks,
    ).build()
