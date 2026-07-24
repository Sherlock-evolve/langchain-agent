"""Stateless validation of citations against one completed Agent turn."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from numbers import Real

from langchain_core.messages import AIMessage, ToolMessage

from citations import (
    format_citation,
    has_unsafe_control_characters,
    validate_answer_citations,
)
from contracts import CitationValidationEvent
from knowledge_retriever import RetrievedChunk


SEARCH_KNOWLEDGE_TOOL_NAME = "search_knowledge"
_PAYLOAD_FIELDS = {
    "corpus_id",
    "returned_count",
    "truncated",
    "notice",
    "results",
}
_RESULT_FIELDS = {
    "rank",
    "score",
    "content",
    "source",
    "start_line",
    "end_line",
    "chunk_id",
    "citation",
}


class KnowledgeCitationValidationError(RuntimeError):
    """The current turn could not be validated without trusting tool data."""


def _reject_json_constant(_value):
    raise ValueError("non-standard JSON constant")


def _strict_json_object(pairs):
    parsed = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError("duplicate JSON key")
        parsed[key] = value
    return parsed


def _parse_payload(
    content: str,
) -> tuple[str, list[RetrievedChunk]]:
    if not isinstance(content, str):
        raise KnowledgeCitationValidationError(
            "Knowledge tool output is not text."
        )
    try:
        payload = json.loads(
            content,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
    except Exception:
        raise KnowledgeCitationValidationError(
            "Knowledge tool output is not valid JSON."
        ) from None

    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_FIELDS:
        raise KnowledgeCitationValidationError(
            "Knowledge tool output has an invalid structure."
        )
    if (
        not isinstance(payload["corpus_id"], str)
        or not payload["corpus_id"]
        or has_unsafe_control_characters(payload["corpus_id"])
        or type(payload["returned_count"]) is not int
        or payload["returned_count"] < 0
        or type(payload["truncated"]) is not bool
        or not isinstance(payload["notice"], str)
        or not payload["notice"]
        or not isinstance(payload["results"], list)
        or payload["returned_count"] != len(payload["results"])
    ):
        raise KnowledgeCitationValidationError(
            "Knowledge tool output metadata is invalid."
        )

    chunks = []
    for raw_result in payload["results"]:
        if not isinstance(raw_result, dict) or set(raw_result) != _RESULT_FIELDS:
            raise KnowledgeCitationValidationError(
                "Knowledge result has an invalid structure."
            )
        if (
            type(raw_result["rank"]) is not int
            or raw_result["rank"] < 1
            or isinstance(raw_result["score"], bool)
            or not isinstance(raw_result["score"], Real)
            or not math.isfinite(float(raw_result["score"]))
            or not -1 <= raw_result["score"] <= 1
            or not isinstance(raw_result["content"], str)
            or not raw_result["content"]
            or not isinstance(raw_result["chunk_id"], str)
            or not raw_result["chunk_id"]
            or has_unsafe_control_characters(raw_result["chunk_id"])
            or not isinstance(raw_result["citation"], str)
        ):
            raise KnowledgeCitationValidationError(
                "Knowledge result fields are invalid."
            )
        try:
            expected_citation = format_citation(
                raw_result["source"],
                raw_result["start_line"],
                raw_result["end_line"],
            )
        except Exception:
            raise KnowledgeCitationValidationError(
                "Knowledge result citation metadata is invalid."
            ) from None
        if raw_result["citation"] != expected_citation:
            raise KnowledgeCitationValidationError(
                "Knowledge result citation does not match its metadata."
            )

        chunks.append(
            RetrievedChunk(
                rank=raw_result["rank"],
                score=float(raw_result["score"]),
                content=raw_result["content"],
                source=raw_result["source"],
                start_line=raw_result["start_line"],
                end_line=raw_result["end_line"],
                chunk_id=raw_result["chunk_id"],
            )
        )
    return payload["corpus_id"], chunks


def _final_answer(messages: Sequence) -> str:
    if not messages:
        raise KnowledgeCitationValidationError(
            "Current turn does not contain a final answer."
        )
    final_message = messages[-1]
    if not isinstance(final_message, AIMessage) or final_message.tool_calls:
        raise KnowledgeCitationValidationError(
            "Current turn does not end with a final answer."
        )
    if isinstance(final_message.content, str):
        return final_message.content
    try:
        answer = final_message.text
    except Exception:
        raise KnowledgeCitationValidationError(
            "Final answer could not be read safely."
        ) from None
    if not isinstance(answer, str):
        raise KnowledgeCitationValidationError(
            "Final answer is not text."
        )
    return answer


def validate_knowledge_citations(
    current_turn_messages: Sequence,
) -> CitationValidationEvent:
    """Validate only the supplied turn, without retaining mutable state."""

    search_messages = [
        message
        for message in current_turn_messages
        if (
            isinstance(message, ToolMessage)
            and message.name == SEARCH_KNOWLEDGE_TOOL_NAME
        )
    ]
    if not search_messages:
        return CitationValidationEvent(
            status="not_applicable",
            citation_count=0,
            valid_citation_count=0,
            unknown_citation_count=0,
            retrieved_chunk_count=0,
        )

    retrieved_chunks = []
    corpus_id = None
    for message in search_messages:
        message_corpus_id, message_chunks = _parse_payload(message.content)
        if corpus_id is None:
            corpus_id = message_corpus_id
        elif message_corpus_id != corpus_id:
            raise KnowledgeCitationValidationError(
                "Knowledge results came from inconsistent corpora."
            )
        retrieved_chunks.extend(message_chunks)

    result = validate_answer_citations(
        _final_answer(current_turn_messages),
        retrieved_chunks,
    )
    return CitationValidationEvent(
        status=result.status,
        citation_count=result.citation_count,
        valid_citation_count=result.valid_citation_count,
        unknown_citation_count=result.unknown_citation_count,
        retrieved_chunk_count=result.retrieved_chunk_count,
    )
