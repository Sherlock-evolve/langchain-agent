"""Explicit, optional assembly of the local corpus and semantic search tool.

Security boundary: when enabled with a production OpenAIEmbeddings
implementation, document chunks and queries may be sent to the configured
external service. The caller must deliberately select and trust that service.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool

from knowledge_base import (
    DEFAULT_DOCUMENTS_DIRECTORY,
    WORKSPACE_ROOT,
    KnowledgeBuildResult,
    build_knowledge_base,
)
from knowledge_citation_validator import validate_knowledge_citations
from knowledge_retriever import KnowledgeRetriever
from knowledge_tools import create_search_knowledge_tool


@dataclass(frozen=True)
class KnowledgeRuntime:
    search_tool: BaseTool
    corpus_id: str
    indexed_file_count: int
    chunk_count: int
    skipped_file_count: int
    citation_validator: Callable = validate_knowledge_citations
    citation_guard_tool_names: frozenset[str] = frozenset(
        {"search_knowledge"}
    )


class KnowledgeRuntimeError(RuntimeError):
    """Base class for safe knowledge-runtime startup failures."""


class KnowledgeConfigurationError(KnowledgeRuntimeError):
    """Required embedding configuration is missing or invalid."""


class KnowledgeCorpusBuildError(KnowledgeRuntimeError):
    """The local document corpus could not be built safely."""


class KnowledgeCorpusTruncatedError(KnowledgeRuntimeError):
    """The corpus exceeded a configured completeness budget."""


class KnowledgeEmbeddingError(KnowledgeRuntimeError):
    """The embedding client or in-memory index could not be created."""


def _nonempty_environment_value(
    environment: Mapping[str, str],
    name: str,
) -> str | None:
    value = environment.get(name)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _embedding_configuration(
    environment: Mapping[str, str],
) -> dict[str, str | None]:
    model = _nonempty_environment_value(
        environment,
        "EMBEDDING_MODEL",
    )
    if model is None:
        raise KnowledgeConfigurationError(
            "EMBEDDING_MODEL must be explicitly configured."
        )

    api_key = (
        _nonempty_environment_value(environment, "EMBEDDING_API_KEY")
        or _nonempty_environment_value(environment, "ZHIPU_API_KEY")
    )
    if api_key is None:
        raise KnowledgeConfigurationError(
            "An embedding API key must be explicitly configured."
        )

    base_url = (
        _nonempty_environment_value(environment, "EMBEDDING_BASE_URL")
        or _nonempty_environment_value(environment, "ZHIPU_BASE_URL")
    )
    return {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }


def _create_openai_embeddings(
    *,
    model: str,
    api_key: str,
    base_url: str | None,
) -> Embeddings:
    # Import and construction only happen after the CLI's explicit opt-in.
    # A production provider may receive document chunks and later queries.
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def create_knowledge_runtime(
    *,
    workspace_root: str | os.PathLike[str] = WORKSPACE_ROOT,
    knowledge_directory: str | os.PathLike[str] = (
        DEFAULT_DOCUMENTS_DIRECTORY
    ),
    environment: Mapping[str, str] | None = None,
    embeddings_factory: Callable[..., Embeddings] | None = None,
    knowledge_builder: Callable[..., KnowledgeBuildResult] = (
        build_knowledge_base
    ),
    retriever_factory: Callable[..., KnowledgeRetriever] = (
        KnowledgeRetriever
    ),
    search_tool_factory: Callable[..., BaseTool] = (
        create_search_knowledge_tool
    ),
) -> KnowledgeRuntime:
    """Build the complete knowledge runtime after an explicit caller opt-in."""

    if environment is None:
        environment = os.environ
    if embeddings_factory is None:
        embeddings_factory = _create_openai_embeddings

    try:
        build_result = knowledge_builder(
            workspace_root=Path(workspace_root),
            docs_directory=knowledge_directory,
        )
    except Exception:
        raise KnowledgeCorpusBuildError(
            "The knowledge corpus could not be built safely."
        ) from None

    if not isinstance(build_result, KnowledgeBuildResult):
        raise KnowledgeCorpusBuildError(
            "The knowledge corpus builder returned an invalid result."
        )
    if build_result.truncated:
        raise KnowledgeCorpusTruncatedError(
            "The knowledge corpus was truncated and will not be indexed."
        )

    configuration = _embedding_configuration(environment)
    try:
        embeddings = embeddings_factory(**configuration)
    except Exception:
        raise KnowledgeEmbeddingError(
            "The embedding client could not be created safely."
        ) from None

    try:
        retriever = retriever_factory(
            chunks=build_result.chunks,
            embeddings=embeddings,
        )
        search_tool = search_tool_factory(retriever)
    except Exception:
        raise KnowledgeEmbeddingError(
            "The in-memory knowledge index could not be created safely."
        ) from None

    return KnowledgeRuntime(
        search_tool=search_tool,
        corpus_id=retriever.corpus_id,
        indexed_file_count=len(build_result.indexed_files),
        chunk_count=len(build_result.chunks),
        skipped_file_count=len(build_result.skipped_files),
    )
