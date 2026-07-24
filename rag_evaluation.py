"""Deterministic, offline retrieval evaluation without an LLM judge."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

from citations import has_unsafe_control_characters
from knowledge_retriever import (
    MAX_QUERY_CHARACTERS,
    MAX_SEARCH_RESULTS,
    RetrievedChunk,
)


METRIC_DECIMAL_PLACES = 6


class RagEvaluationError(RuntimeError):
    """A safe evaluation failure without query or provider details."""


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    case_id: str
    query: str
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class RetrievalCaseResult:
    case_id: str
    hit: bool
    reciprocal_rank: float
    recall: float
    retrieved_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    corpus_id: str
    case_count: int
    hit_rate: float
    mean_reciprocal_rank: float
    mean_recall: float
    cases: tuple[RetrievalCaseResult, ...]


def _validate_k(k: int) -> None:
    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or k < 1
        or k > MAX_SEARCH_RESULTS
    ):
        raise ValueError(
            f"k must be an integer from 1 to {MAX_SEARCH_RESULTS}"
        )


def _valid_plain_string(value, *, allow_text_whitespace=False) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and not has_unsafe_control_characters(
            value,
            allow_text_whitespace=allow_text_whitespace,
        )
    )


def _validate_cases(
    cases: Iterable[RetrievalEvaluationCase],
) -> tuple[RetrievalEvaluationCase, ...]:
    try:
        validated_cases = tuple(cases)
    except Exception:
        raise ValueError(
            "cases must be an iterable of evaluation cases"
        ) from None

    if not validated_cases:
        raise ValueError("at least one evaluation case is required")

    seen_case_ids = set()
    for case in validated_cases:
        if not isinstance(case, RetrievalEvaluationCase):
            raise ValueError(
                "cases contains an invalid evaluation case"
            )
        if not _valid_plain_string(case.case_id):
            raise ValueError("case_id must be a non-empty safe string")
        if case.case_id in seen_case_ids:
            raise ValueError("case_id values must be unique")
        seen_case_ids.add(case.case_id)

        if (
            not _valid_plain_string(
                case.query,
                allow_text_whitespace=True,
            )
            or len(case.query) > MAX_QUERY_CHARACTERS
        ):
            raise ValueError("case query is invalid")
        if (
            not isinstance(case.relevant_chunk_ids, frozenset)
            or not case.relevant_chunk_ids
        ):
            raise ValueError(
                "relevant_chunk_ids must be a non-empty frozenset"
            )
        if any(
            not _valid_plain_string(chunk_id)
            for chunk_id in case.relevant_chunk_ids
        ):
            raise ValueError(
                "relevant_chunk_ids contains an invalid identifier"
            )

    return validated_cases


def _validated_corpus_id(retriever) -> str:
    try:
        corpus_id = retriever.corpus_id
    except Exception:
        raise RagEvaluationError(
            "Retrieval evaluation could not read the corpus identifier."
        ) from None
    if (
        not _valid_plain_string(corpus_id)
        or len(corpus_id) > 256
    ):
        raise RagEvaluationError(
            "Retrieval evaluation received an invalid corpus identifier."
        )
    return corpus_id


def _search_case(retriever, case, k: int) -> tuple[str, ...]:
    try:
        results = retriever.search(case.query, k=k)
        limited_results = tuple(islice(iter(results), k))
        chunk_ids = []
        for result in limited_results:
            if not isinstance(result, RetrievedChunk):
                raise TypeError("invalid retrieval result")
            if not _valid_plain_string(result.chunk_id):
                raise ValueError("invalid retrieved chunk identifier")
            chunk_ids.append(result.chunk_id)
        return tuple(chunk_ids)
    except Exception:
        raise RagEvaluationError(
            "Retrieval evaluation search failed safely."
        ) from None


def _rounded(value: float) -> float:
    rounded_value = round(value, METRIC_DECIMAL_PLACES)
    return 0.0 if rounded_value == 0 else rounded_value


def evaluate_retrieval(
    retriever,
    cases: Iterable[RetrievalEvaluationCase],
    *,
    k: int = 4,
) -> RetrievalEvaluationReport:
    """Evaluate Hit Rate, MRR, and mean Recall at k deterministically."""

    _validate_k(k)
    validated_cases = _validate_cases(cases)
    corpus_id = _validated_corpus_id(retriever)
    case_results = []
    reciprocal_ranks = []
    recalls = []

    for case in validated_cases:
        retrieved_chunk_ids = _search_case(retriever, case, k)
        relevant_ids = case.relevant_chunk_ids
        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(
                    retrieved_chunk_ids,
                    start=1,
                )
                if chunk_id in relevant_ids
            ),
            None,
        )
        hit = first_relevant_rank is not None
        reciprocal_rank = (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )
        retrieved_unique_ids = set(retrieved_chunk_ids)
        recall = len(
            retrieved_unique_ids.intersection(relevant_ids)
        ) / len(relevant_ids)
        reciprocal_ranks.append(reciprocal_rank)
        recalls.append(recall)

        case_results.append(
            RetrievalCaseResult(
                case_id=case.case_id,
                hit=hit,
                reciprocal_rank=_rounded(reciprocal_rank),
                recall=_rounded(recall),
                retrieved_chunk_ids=retrieved_chunk_ids,
            )
        )

    case_count = len(case_results)
    return RetrievalEvaluationReport(
        corpus_id=corpus_id,
        case_count=case_count,
        hit_rate=_rounded(
            sum(result.hit for result in case_results) / case_count
        ),
        mean_reciprocal_rank=_rounded(
            sum(reciprocal_ranks) / case_count
        ),
        mean_recall=_rounded(
            sum(recalls) / case_count
        ),
        cases=tuple(case_results),
    )


run_retrieval_evaluation = evaluate_retrieval
