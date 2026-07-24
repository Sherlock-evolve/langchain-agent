"""Canonical citation formatting, extraction, and answer validation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from knowledge_retriever import RetrievedChunk


MAX_ANSWER_CHARACTERS = 100_000
SUPPORTED_CITATION_EXTENSIONS = frozenset({".md", ".txt"})
_SOURCE_SEGMENT = r"[\w@%+=,.~-]+"
_SOURCE_PATTERN = (
    rf"(?P<source>(?:{_SOURCE_SEGMENT}/)*"
    rf"{_SOURCE_SEGMENT}\.(?:md|txt))"
)
_SOURCE_ONLY_PATTERN = re.compile(
    _SOURCE_PATTERN,
    flags=re.IGNORECASE,
)
_CITATION_PATTERN = re.compile(
    rf"(?<![\w@%+=,.~/-])"
    rf"{_SOURCE_PATTERN}"
    rf":L(?P<start>[1-9]\d*)"
    rf"(?:-L(?P<end>[1-9]\d*))?"
    rf"(?![\w@%+=,.~/-]|-L\d)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationValidationResult:
    status: Literal[
        "valid",
        "missing",
        "unknown",
        "not_applicable",
    ]
    citation_count: int
    valid_citation_count: int
    unknown_citation_count: int
    retrieved_chunk_count: int


def has_unsafe_control_characters(
    value: str,
    *,
    allow_text_whitespace: bool = False,
) -> bool:
    allowed_controls = {"\n", "\r", "\t"} if allow_text_whitespace else set()
    for character in value:
        if character in allowed_controls:
            continue
        category = unicodedata.category(character)
        if category.startswith("C") or character in {"\u2028", "\u2029"}:
            return True
    return False


def _validate_source(source: str) -> None:
    if (
        not isinstance(source, str)
        or not source
        or has_unsafe_control_characters(source)
        or ":" in source
        or "\\" in source
        or any(character.isspace() for character in source)
        or _SOURCE_ONLY_PATTERN.fullmatch(source) is None
    ):
        raise ValueError("citation source is invalid")

    source_path = PurePosixPath(source)
    if (
        source_path.is_absolute()
        or any(part in {"", ".", ".."} for part in source_path.parts)
        or source_path.suffix.lower() not in SUPPORTED_CITATION_EXTENSIONS
    ):
        raise ValueError("citation source is invalid")


def _validate_line_number(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def format_citation(
    source: str,
    start_line: int,
    end_line: int | None = None,
) -> str:
    """Format one validated workspace citation in canonical form."""

    _validate_source(source)
    _validate_line_number("start_line", start_line)
    if end_line is None:
        end_line = start_line
    _validate_line_number("end_line", end_line)
    if end_line < start_line:
        raise ValueError("end_line must not precede start_line")

    if start_line == end_line:
        return f"{source}:L{start_line}"
    return f"{source}:L{start_line}-L{end_line}"


def _validate_answer(answer: str) -> None:
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    if len(answer) > MAX_ANSWER_CHARACTERS:
        raise ValueError(
            f"answer must not exceed {MAX_ANSWER_CHARACTERS} characters"
        )
    if has_unsafe_control_characters(
        answer,
        allow_text_whitespace=True,
    ):
        raise ValueError("answer contains unsafe control characters")


def extract_citations(answer: str) -> tuple[str, ...]:
    """Extract canonical .md/.txt citations in answer order."""

    _validate_answer(answer)
    citations = []
    for match in _CITATION_PATTERN.finditer(answer):
        start_line = int(match.group("start"))
        end_group = match.group("end")
        end_line = int(end_group) if end_group is not None else start_line
        citations.append(
            format_citation(
                match.group("source"),
                start_line,
                end_line,
            )
        )
    return tuple(citations)


def _known_citations(
    retrieved_chunks: Iterable[RetrievedChunk],
) -> tuple[set[str], int]:
    try:
        chunks = list(retrieved_chunks)
    except Exception:
        raise ValueError(
            "retrieved_chunks must be an iterable of RetrievedChunk"
        ) from None

    known = set()
    for chunk in chunks:
        if not isinstance(chunk, RetrievedChunk):
            raise ValueError(
                "retrieved_chunks contains an invalid result"
            )
        known.add(
            format_citation(
                chunk.source,
                chunk.start_line,
                chunk.end_line,
            )
        )
    return known, len(chunks)


def validate_answer_citations(
    answer: str,
    retrieved_chunks: Iterable[RetrievedChunk],
) -> CitationValidationResult:
    """Classify answer citations without retaining the answer or its text."""

    citations = extract_citations(answer)
    known_citations, chunk_count = _known_citations(retrieved_chunks)
    valid_count = sum(
        citation in known_citations for citation in citations
    )
    unknown_count = len(citations) - valid_count

    if unknown_count:
        status = "unknown"
    elif citations:
        status = "valid"
    elif known_citations:
        status = "missing"
    else:
        status = "not_applicable"

    return CitationValidationResult(
        status=status,
        citation_count=len(citations),
        valid_citation_count=valid_count,
        unknown_citation_count=unknown_count,
        retrieved_chunk_count=chunk_count,
    )
