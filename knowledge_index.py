"""Secure persistent cache for incremental document embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import fcntl
from langchain_core.embeddings import Embeddings


INDEX_SCHEMA_VERSION = 1
INDEX_DIRECTORY_NAME = ".knowledge_index"
MAX_INDEX_BYTES = 100 * 1024 * 1024
MAX_INDEX_ENTRIES = 100_000
MAX_VECTOR_DIMENSIONS = 65_536
DEFAULT_LOCK_TIMEOUT_SECONDS = 10.0
DEFAULT_LOCK_POLL_SECONDS = 0.05


class KnowledgeIndexCacheError(RuntimeError):
    """The persistent embedding cache could not be used safely."""


def embedding_configuration_fingerprint(
    *,
    model: str,
    base_url: str | None,
) -> str:
    """Hash non-secret provider identity used to isolate cached vectors."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("embedding model must be a non-empty string")
    if base_url is not None and not isinstance(base_url, str):
        raise ValueError("embedding base_url must be a string or None")
    payload = json.dumps(
        {
            "model": model.strip(),
            "base_url": (base_url or "").strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class IncrementalEmbeddingIndex:
    """A workspace-local, model-isolated mapping of content hashes to vectors."""

    def __init__(
        self,
        workspace_root: str | os.PathLike[str],
        knowledge_directory: str | os.PathLike[str],
        embedding_fingerprint: str,
        *,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        lock_poll_seconds: float = DEFAULT_LOCK_POLL_SECONDS,
    ) -> None:
        workspace = Path(workspace_root)
        if workspace.is_symlink():
            raise KnowledgeIndexCacheError(
                "Knowledge index workspace cannot be a symbolic link."
            )
        try:
            self.workspace_root = workspace.resolve(strict=True)
        except OSError:
            raise KnowledgeIndexCacheError(
                "Knowledge index workspace does not exist."
            ) from None
        if not self.workspace_root.is_dir():
            raise KnowledgeIndexCacheError(
                "Knowledge index workspace is not a directory."
            )
        if (
            not isinstance(embedding_fingerprint, str)
            or len(embedding_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in embedding_fingerprint
            )
        ):
            raise KnowledgeIndexCacheError(
                "Knowledge index embedding fingerprint is invalid."
            )

        directory_text = os.fspath(knowledge_directory)
        if not isinstance(directory_text, str) or not directory_text:
            raise KnowledgeIndexCacheError(
                "Knowledge directory identity is invalid."
            )
        namespace_payload = json.dumps(
            [directory_text, embedding_fingerprint],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        namespace = hashlib.sha256(namespace_payload).hexdigest()
        for name, value in (
            ("lock_timeout_seconds", lock_timeout_seconds),
            ("lock_poll_seconds", lock_poll_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise KnowledgeIndexCacheError(
                    f"{name} must be a positive finite number."
                )
        self.embedding_fingerprint = embedding_fingerprint
        self.root = self.workspace_root / INDEX_DIRECTORY_NAME
        self.path = self.root / f"{namespace}.json"
        self.lock_path = self.root / f"{namespace}.lock"
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.lock_poll_seconds = float(lock_poll_seconds)
        with self._file_lock():
            self._entries = self._load()

    @staticmethod
    def content_key(text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("embedding text must be a string")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def validate_vector(vector) -> list[float]:
        if not isinstance(vector, (list, tuple)):
            raise KnowledgeIndexCacheError(
                "Embedding vector must be a list."
            )
        if not vector or len(vector) > MAX_VECTOR_DIMENSIONS:
            raise KnowledgeIndexCacheError(
                "Embedding vector dimensions are invalid."
            )
        normalized = []
        for value in vector:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise KnowledgeIndexCacheError(
                    "Embedding vector contains an invalid value."
                )
            normalized.append(float(value))
        return normalized

    def get(self, key: str) -> list[float] | None:
        vector = self._entries.get(key)
        return list(vector) if vector is not None else None

    def commit(
        self,
        entries: dict[str, list[float]],
        active_keys: set[str],
        corpus_id: str,
    ) -> None:
        if (
            not isinstance(corpus_id, str)
            or not corpus_id
            or len(corpus_id) > 256
        ):
            raise KnowledgeIndexCacheError(
                "Knowledge corpus identifier is invalid."
            )
        if not isinstance(entries, dict) or not isinstance(active_keys, set):
            raise KnowledgeIndexCacheError(
                "Knowledge index entries are invalid."
            )
        if len(active_keys) > MAX_INDEX_ENTRIES:
            raise KnowledgeIndexCacheError(
                "Knowledge index contains too many entries."
            )

        normalized_entries = {}
        vector_dimensions = None
        for key in sorted(active_keys):
            if (
                not isinstance(key, str)
                or len(key) != 64
                or key not in entries
            ):
                raise KnowledgeIndexCacheError(
                    "Knowledge index content key is invalid."
                )
            normalized_vector = self.validate_vector(entries[key])
            if vector_dimensions is None:
                vector_dimensions = len(normalized_vector)
            elif len(normalized_vector) != vector_dimensions:
                raise KnowledgeIndexCacheError(
                    "Knowledge index vectors have inconsistent dimensions."
                )
            normalized_entries[key] = normalized_vector

        payload = {
            "version": INDEX_SCHEMA_VERSION,
            "embedding_fingerprint": self.embedding_fingerprint,
            "corpus_id": corpus_id,
            "entries": normalized_entries,
        }
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise KnowledgeIndexCacheError(
                "Knowledge index could not be encoded safely."
            ) from None
        if len(encoded) > MAX_INDEX_BYTES:
            raise KnowledgeIndexCacheError(
                "Knowledge index exceeds its size limit."
            )

        with self._file_lock():
            self._atomic_write(encoded)
            self._entries = normalized_entries

    def _prepare_root(self) -> None:
        if self.root.is_symlink():
            raise KnowledgeIndexCacheError(
                "Knowledge index directory cannot be a symbolic link."
            )
        try:
            if self.root.exists():
                if not self.root.is_dir():
                    raise KnowledgeIndexCacheError(
                        "Knowledge index root is not a directory."
                    )
            else:
                self.root.mkdir(mode=0o700)
            os.chmod(self.root, 0o700)
        except KnowledgeIndexCacheError:
            raise
        except OSError:
            raise KnowledgeIndexCacheError(
                "Knowledge index directory could not be prepared."
            ) from None

    @contextmanager
    def _file_lock(self):
        """Hold an exclusive, cross-process lock for this index namespace."""
        self._prepare_root()
        if self.lock_path.is_symlink():
            raise KnowledgeIndexCacheError(
                "Knowledge index lock cannot be a symbolic link."
            )
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            lock_fd = os.open(self.lock_path, flags, 0o600)
        except OSError:
            raise KnowledgeIndexCacheError(
                "Knowledge index lock could not be opened safely."
            ) from None

        acquired = False
        try:
            lock_stat = os.fstat(lock_fd)
            if not stat.S_ISREG(lock_stat.st_mode):
                raise KnowledgeIndexCacheError(
                    "Knowledge index lock is not a regular file."
                )
            os.fchmod(lock_fd, 0o600)
            deadline = time.monotonic() + self.lock_timeout_seconds
            while True:
                try:
                    fcntl.flock(
                        lock_fd,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    acquired = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise KnowledgeIndexCacheError(
                            "Timed out waiting for the knowledge index lock."
                        ) from None
                    time.sleep(self.lock_poll_seconds)
            yield
        finally:
            if acquired:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _load(self) -> dict[str, list[float]]:
        if self.root.is_symlink() or self.path.is_symlink():
            raise KnowledgeIndexCacheError(
                "Knowledge index paths cannot be symbolic links."
            )
        if not self.path.exists():
            return {}
        try:
            file_stat = self.path.lstat()
        except OSError:
            raise KnowledgeIndexCacheError(
                "Knowledge index metadata could not be read."
            ) from None
        if not stat.S_ISREG(file_stat.st_mode):
            raise KnowledgeIndexCacheError(
                "Knowledge index path is not a regular file."
            )
        if file_stat.st_size > MAX_INDEX_BYTES:
            raise KnowledgeIndexCacheError(
                "Knowledge index exceeds its size limit."
            )

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_descriptor = os.open(self.path, flags)
            with os.fdopen(file_descriptor, "rb") as cache_file:
                encoded = cache_file.read(MAX_INDEX_BYTES + 1)
        except OSError:
            raise KnowledgeIndexCacheError(
                "Knowledge index could not be opened safely."
            ) from None
        if len(encoded) > MAX_INDEX_BYTES:
            raise KnowledgeIndexCacheError(
                "Knowledge index exceeds its size limit."
            )
        try:
            payload = json.loads(
                encoded.decode("utf-8"),
                parse_constant=self._reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise KnowledgeIndexCacheError(
                "Knowledge index contains invalid JSON."
            ) from None

        if (
            not isinstance(payload, dict)
            or set(payload) != {
                "version",
                "embedding_fingerprint",
                "corpus_id",
                "entries",
            }
            or type(payload["version"]) is not int
            or payload["version"] != INDEX_SCHEMA_VERSION
            or payload["embedding_fingerprint"]
            != self.embedding_fingerprint
            or not isinstance(payload["corpus_id"], str)
            or not isinstance(payload["entries"], dict)
            or len(payload["entries"]) > MAX_INDEX_ENTRIES
        ):
            raise KnowledgeIndexCacheError(
                "Knowledge index manifest is invalid."
            )

        entries = {}
        vector_dimensions = None
        for key, vector in payload["entries"].items():
            if (
                not isinstance(key, str)
                or len(key) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in key
                )
            ):
                raise KnowledgeIndexCacheError(
                    "Knowledge index content key is invalid."
                )
            normalized_vector = self.validate_vector(vector)
            if vector_dimensions is None:
                vector_dimensions = len(normalized_vector)
            elif len(normalized_vector) != vector_dimensions:
                raise KnowledgeIndexCacheError(
                    "Knowledge index vectors have inconsistent dimensions."
                )
            entries[key] = normalized_vector
        return entries

    def _atomic_write(self, encoded: bytes) -> None:
        self._prepare_root()

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.root,
                prefix=f".{self.path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                os.fchmod(temporary_file.fileno(), 0o600)
                temporary_file.write(encoded)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            if self.path.is_symlink():
                raise KnowledgeIndexCacheError(
                    "Knowledge index file cannot be a symbolic link."
                )
            os.replace(temporary_path, self.path)
            temporary_path = None
            self._fsync_directory()
        except KnowledgeIndexCacheError:
            raise
        except OSError:
            raise KnowledgeIndexCacheError(
                "Knowledge index could not be saved safely."
            ) from None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

    def _fsync_directory(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(self.root, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _reject_json_constant(value: str):
        raise ValueError(f"unsupported JSON constant: {value}")


class CachedEmbeddings(Embeddings):
    """Embedding adapter that reuses and persists unchanged document vectors."""

    def __init__(
        self,
        delegate: Embeddings,
        index: IncrementalEmbeddingIndex,
    ) -> None:
        if not isinstance(delegate, Embeddings):
            raise TypeError("delegate must implement Embeddings")
        if not isinstance(index, IncrementalEmbeddingIndex):
            raise TypeError("index must be IncrementalEmbeddingIndex")
        self.delegate = delegate
        self.index = index
        self._entries = dict(index._entries)
        self._vector_dimensions = (
            len(next(iter(self._entries.values())))
            if self._entries
            else None
        )
        self.reused_count = 0
        self.created_count = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or any(
            not isinstance(text, str) for text in texts
        ):
            raise ValueError("embedding documents must be a list of strings")

        keys = [self.index.content_key(text) for text in texts]
        missing_keys = []
        missing_texts = []
        seen_missing = set()
        for key, text in zip(keys, texts):
            if key in self._entries:
                self.reused_count += 1
            elif key not in seen_missing:
                seen_missing.add(key)
                missing_keys.append(key)
                missing_texts.append(text)

        if missing_texts:
            vectors = self.delegate.embed_documents(missing_texts)
            if (
                not isinstance(vectors, list)
                or len(vectors) != len(missing_texts)
            ):
                raise KnowledgeIndexCacheError(
                    "Embedding result count did not match documents."
                )
            for key, vector in zip(missing_keys, vectors):
                normalized_vector = self.index.validate_vector(vector)
                if self._vector_dimensions is None:
                    self._vector_dimensions = len(normalized_vector)
                elif len(normalized_vector) != self._vector_dimensions:
                    raise KnowledgeIndexCacheError(
                        "Embedding vector dimensions changed."
                    )
                self._entries[key] = normalized_vector
                self.created_count += 1

        return [list(self._entries[key]) for key in keys]

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise ValueError("embedding query must be a string")
        vector = self.index.validate_vector(
            self.delegate.embed_query(text)
        )
        if (
            self._vector_dimensions is not None
            and len(vector) != self._vector_dimensions
        ):
            raise KnowledgeIndexCacheError(
                "Query embedding dimensions do not match the index."
            )
        return vector

    def commit(self, active_texts: list[str], corpus_id: str) -> None:
        active_keys = {
            self.index.content_key(text)
            for text in active_texts
        }
        self.index.commit(
            self._entries,
            active_keys,
            corpus_id,
        )
