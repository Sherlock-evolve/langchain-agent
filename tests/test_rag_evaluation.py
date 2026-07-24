from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from citations import (
    MAX_ANSWER_CHARACTERS,
    extract_citations,
    format_citation,
    validate_answer_citations,
)
from knowledge_retriever import RetrievedChunk
from rag_evaluation import (
    RagEvaluationError,
    RetrievalEvaluationCase,
    evaluate_retrieval,
)


def result(
    chunk_id,
    *,
    rank=1,
    source="docs/source.md",
    start_line=1,
    end_line=1,
    content="document body",
):
    return RetrievedChunk(
        rank=rank,
        score=1.0,
        content=content,
        source=source,
        start_line=start_line,
        end_line=end_line,
        chunk_id=chunk_id,
    )


class ControlledRetriever:
    def __init__(self, results_by_query, corpus_id="corpus-evaluation"):
        self.results_by_query = results_by_query
        self.corpus_id = corpus_id
        self.calls = []

    def search(self, query, k=4):
        self.calls.append((query, k))
        response = self.results_by_query[query]
        if isinstance(response, BaseException):
            raise response
        return response


def test_citation_formatting_and_extraction_are_canonical():
    answer = (
        "参考 docs/file.md:L10 和 docs/guide.txt:L10-L18。\n"
        "以下内容不识别：docs/code.py:L1、docs/zero.md:L0。"
    )

    assert format_citation("docs/file.md", 10) == "docs/file.md:L10"
    assert (
        format_citation("docs/guide.txt", 10, 18)
        == "docs/guide.txt:L10-L18"
    )
    assert extract_citations(answer) == (
        "docs/file.md:L10",
        "docs/guide.txt:L10-L18",
    )

    with pytest.raises(ValueError, match="precede"):
        format_citation("docs/file.md", 18, 10)
    with pytest.raises(ValueError, match="precede"):
        extract_citations("伪造 docs/file.md:L18-L10")
    with pytest.raises(ValueError, match="control"):
        extract_citations("docs/file.md:L1\x00")
    with pytest.raises(ValueError, match="exceed"):
        extract_citations("x" * (MAX_ANSWER_CHARACTERS + 1))
    for unsafe_source in (
        "docs/file.json",
        "../docs/file.md",
        "docs/forged.md\nother.txt",
    ):
        with pytest.raises(ValueError):
            format_citation(unsafe_source, 1)


def test_answer_citation_validation_classifies_all_statuses():
    retrieved_chunks = [
        result(
            "known-a",
            source="docs/a.md",
            start_line=2,
            end_line=4,
        ),
        result(
            "known-b",
            source="docs/b.txt",
            start_line=8,
            end_line=8,
        ),
    ]

    valid = validate_answer_citations(
        "结论 docs/a.md:L2-L4",
        retrieved_chunks,
    )
    missing = validate_answer_citations(
        "没有提供引用",
        retrieved_chunks,
    )
    unknown = validate_answer_citations(
        "docs/a.md:L2-L4 和 docs/fake.md:L1",
        retrieved_chunks,
    )
    not_applicable = validate_answer_citations("普通回答", [])
    unknown_without_results = validate_answer_citations(
        "伪造 docs/fake.md:L1",
        [],
    )

    assert valid.status == "valid"
    assert valid.citation_count == 1
    assert valid.valid_citation_count == 1
    assert valid.unknown_citation_count == 0
    assert valid.retrieved_chunk_count == 2
    assert missing.status == "missing"
    assert unknown.status == "unknown"
    assert unknown.valid_citation_count == 1
    assert unknown.unknown_citation_count == 1
    assert not_applicable.status == "not_applicable"
    assert unknown_without_results.status == "unknown"
    with pytest.raises(FrozenInstanceError):
        valid.status = "missing"
    assert "没有提供引用" not in repr(missing)


def test_known_rankings_compute_hit_rate_mrr_and_mean_recall():
    retriever = ControlledRetriever(
        {
            "first-query": [
                result("irrelevant"),
                result("a"),
            ],
            "second-query": [
                result("irrelevant"),
            ],
            "third-query": [
                result("c"),
            ],
        }
    )
    cases = [
        RetrievalEvaluationCase(
            case_id="first",
            query="first-query",
            relevant_chunk_ids=frozenset({"a"}),
        ),
        RetrievalEvaluationCase(
            case_id="second",
            query="second-query",
            relevant_chunk_ids=frozenset({"b"}),
        ),
        RetrievalEvaluationCase(
            case_id="third",
            query="third-query",
            relevant_chunk_ids=frozenset({"c", "d"}),
        ),
    ]

    report = evaluate_retrieval(retriever, cases, k=2)

    assert report.corpus_id == "corpus-evaluation"
    assert report.case_count == 3
    assert report.hit_rate == 0.666667
    assert report.mean_reciprocal_rank == 0.5
    assert report.mean_recall == 0.5
    assert [case.case_id for case in report.cases] == [
        "first",
        "second",
        "third",
    ]
    assert [case.hit for case in report.cases] == [True, False, True]
    assert [
        case.reciprocal_rank for case in report.cases
    ] == [0.5, 0.0, 1.0]
    assert [case.recall for case in report.cases] == [1.0, 0.0, 0.5]
    assert retriever.calls == [
        ("first-query", 2),
        ("second-query", 2),
        ("third-query", 2),
    ]
    assert "query" not in repr(report)
    assert "document body" not in repr(report)


def test_multiple_relevant_ids_and_duplicate_results_are_deterministic():
    results_by_query = {
        "duplicates": [
            result("x"),
            result("a"),
            result("a"),
            result("b"),
        ],
        "ordered": [
            result("z"),
            result("y"),
        ],
    }
    cases = (
        RetrievalEvaluationCase(
            case_id="duplicates-case",
            query="duplicates",
            relevant_chunk_ids=frozenset({"a", "b"}),
        ),
        RetrievalEvaluationCase(
            case_id="ordered-case",
            query="ordered",
            relevant_chunk_ids=frozenset({"z"}),
        ),
    )

    first_report = evaluate_retrieval(
        ControlledRetriever(results_by_query),
        cases,
        k=4,
    )
    second_report = evaluate_retrieval(
        ControlledRetriever(results_by_query),
        cases,
        k=4,
    )

    duplicate_case = first_report.cases[0]
    assert first_report == second_report
    assert [case.case_id for case in first_report.cases] == [
        "duplicates-case",
        "ordered-case",
    ]
    assert duplicate_case.retrieved_chunk_ids == ("x", "a", "a", "b")
    assert duplicate_case.reciprocal_rank == 0.5
    assert duplicate_case.recall == 1.0
    with pytest.raises(FrozenInstanceError):
        duplicate_case.hit = False


def test_invalid_cases_and_retrieval_failures_are_safe_and_call_free():
    retriever = ControlledRetriever({"valid-query": []})
    valid_case = RetrievalEvaluationCase(
        case_id="valid",
        query="valid-query",
        relevant_chunk_ids=frozenset({"relevant"}),
    )
    invalid_inputs = [
        ([], 4),
        (
            [
                valid_case,
                RetrievalEvaluationCase(
                    case_id="valid",
                    query="other",
                    relevant_chunk_ids=frozenset({"other"}),
                ),
            ],
            4,
        ),
        (
            [
                valid_case,
                RetrievalEvaluationCase(
                    case_id="",
                    query="other",
                    relevant_chunk_ids=frozenset({"other"}),
                ),
            ],
            4,
        ),
        (
            [
                RetrievalEvaluationCase(
                    case_id="bad-query",
                    query="\x00",
                    relevant_chunk_ids=frozenset({"other"}),
                )
            ],
            4,
        ),
        (
            [
                RetrievalEvaluationCase(
                    case_id="empty-relevant",
                    query="other",
                    relevant_chunk_ids=frozenset(),
                )
            ],
            4,
        ),
        (
            [
                RetrievalEvaluationCase(
                    case_id="unsafe-id",
                    query="other",
                    relevant_chunk_ids=frozenset({"secret\x00id"}),
                )
            ],
            4,
        ),
        ([valid_case], 0),
        ([valid_case], 21),
    ]

    for cases, k in invalid_inputs:
        with pytest.raises(ValueError):
            evaluate_retrieval(retriever, cases, k=k)
    assert retriever.calls == []

    secret_query = "QUERY-SECRET-SENTINEL"
    failing_retriever = ControlledRetriever(
        {
            secret_query: RuntimeError(
                "EMBEDDING-CREDENTIAL-SECRET-SENTINEL"
            )
        }
    )
    with pytest.raises(RagEvaluationError) as failure:
        evaluate_retrieval(
            failing_retriever,
            [
                RetrievalEvaluationCase(
                    case_id="failure",
                    query=secret_query,
                    relevant_chunk_ids=frozenset({"relevant"}),
                )
            ],
        )

    failure_message = str(failure.value)
    assert secret_query not in failure_message
    assert "EMBEDDING-CREDENTIAL-SECRET-SENTINEL" not in failure_message
    assert failing_retriever.calls == [(secret_query, 4)]
