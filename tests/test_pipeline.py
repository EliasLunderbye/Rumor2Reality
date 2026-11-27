import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.demo import abstract_chunks
from app.graph.nodes.analyze_paper import analyze_single_paper
from app.graph.nodes.group_papers import group_chunks
from app.graph.workflow import verify_claim
from app.llm.schemas import RetrievedChunk, PaperAssessment


@pytest.fixture
def chunks():
    return [RetrievedChunk.model_validate(r) for r in json.loads(
        (Path(__file__).parent / "fixtures/retrieved_chunks.json").read_text())]


class FakeLLM:
    def __init__(self, *, bad_paper=False, bad_aggregate=False, fail_paper=None, barrier=None):
        self.bad_paper = bad_paper
        self.bad_aggregate = bad_aggregate
        self.fail_paper = fail_paper
        self.barrier = barrier

    def with_structured_output(self, schema, **kwargs):
        def invoke(messages):
            content = messages[1].content
            if schema is PaperAssessment:
                paper = json.loads(content.split("Paper evidence JSON:\n")[1])
                if self.barrier:
                    self.barrier.wait(timeout=5)
                if paper["paper_id"] == self.fail_paper:
                    raise TimeoutError("simulated timeout")
                return {"stance": "mixed", "explanation": "Synthetic evidence only.",
                        "study_design": "observational cohort", "findings": [
                            {"statement": "A synthetic association was examined.",
                             "chunk_id": "invented" if self.bad_paper else paper["chunks"][0]["chunk_id"]}]}
            payload = json.loads(content)
            return {"verdict": "mixed", "confidence": "low", "confidence_reason": "Synthetic test.",
                    "evidence_summary": [{"statement": "Synthetic evidence is mixed.",
                        "citation_ids": ["999"] if self.bad_aggregate else
                            [s["citation_id"] for s in payload["sources"]]}]}
        return SimpleNamespace(invoke=invoke)


def test_end_to_end_preserves_provenance_and_parallel_execution(chunks):
    result = verify_claim("Synthetic claim", chunks, llm=FakeLLM(barrier=threading.Barrier(2)), retries=0)
    assert result.verdict == "mixed"
    assert [s.citation_id for s in result.sources] == ["1", "2"]
    assert [s.paper_id for s in result.sources] == ["synthetic-a", "synthetic-b"]
    assert result.sources[0].excerpt == chunks[0].text
    assert result.sources[0].page_number == 2
    assert result.sources[0].metadata == {"synthetic": True}
    assert len(result.analyses) == 2


def test_sequential_progress_counts_grouped_papers(chunks):
    updates = []
    result = verify_claim(
        "Synthetic claim", [chunks[0], chunks[0], chunks[1]], llm=FakeLLM(),
        max_workers=1, paper_progress=lambda index, total: updates.append((index, total)),
    )
    assert updates == [(1, 2), (2, 2)]
    assert len(result.analyses) == 2


def test_duplicate_and_conflicting_metadata(chunks):
    papers, notes = group_chunks([chunks[0], chunks[0]])
    assert len(papers[0].chunks) == 1
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        group_chunks([chunks[0], chunks[0].model_copy(update={"text": "different"})])
    with pytest.raises(ValueError, match="Conflicting paper metadata"):
        group_chunks([chunks[0], chunks[0].model_copy(update={"chunk_id": "other", "title": "different"})])


def test_final_sources_only_include_cited_papers(chunks):
    class SelectiveLLM(FakeLLM):
        def with_structured_output(self, schema, **kwargs):
            parent = super().with_structured_output(schema, **kwargs)

            def invoke(messages):
                result = parent.invoke(messages)
                if schema is not PaperAssessment:
                    result["evidence_summary"][0]["citation_ids"] = ["2"]
                return result

            return SimpleNamespace(invoke=invoke)

    result = verify_claim("Synthetic claim", chunks, llm=SelectiveLLM(), retries=0)
    assert [source.citation_id for source in result.sources] == ["2"]
    assert len(result.analyses) == 2


def test_abstract_only_confidence_does_not_assume_independent_evidence(chunks):
    class OverconfidentLLM(FakeLLM):
        def with_structured_output(self, schema, **kwargs):
            parent = super().with_structured_output(schema, **kwargs)

            def invoke(messages):
                result = parent.invoke(messages)
                if schema is not PaperAssessment:
                    result.update(confidence="high", confidence_reason="Independent trials.")
                return result

            return SimpleNamespace(invoke=invoke)

    abstracts = [chunk.model_copy(update={"metadata": {"evidence_type": "abstract"}})
                 for chunk in chunks]
    result = verify_claim("Synthetic claim", abstracts, llm=OverconfidentLLM(), retries=0)
    assert result.confidence == "moderate"
    assert "overlap" in result.confidence_reason
    assert "Independent trials" not in result.confidence_reason


def test_ranking_limits_and_order(chunks):
    papers, notes = group_chunks(chunks, higher_score_better=False, max_papers=1)
    assert papers[0].paper_id == "synthetic-b"
    assert notes
    papers, notes = group_chunks(chunks, max_chars_per_paper=1)
    assert all(not p.chunks for p in papers)
    assert len(notes) == 2
    unscored = [c.model_copy(update={"score": None}) for c in reversed(chunks)]
    assert group_chunks(unscored)[0][0].paper_id == "synthetic-b"


def test_invalid_paper_citation_rejected(chunks):
    paper = group_chunks(chunks)[0][0]
    with pytest.raises(ValueError, match="Unknown citation"):
        analyze_single_paper("claim", paper, llm=FakeLLM(bad_paper=True))
    paper.chunks[0].paper_id = "foreign"
    with pytest.raises(ValueError, match="different paper"):
        analyze_single_paper("claim", paper, llm=FakeLLM())


def test_invalid_aggregate_citation_fails_closed(chunks):
    with pytest.raises(ValueError, match="unknown citation"):
        verify_claim("claim", chunks, llm=FakeLLM(bad_aggregate=True), retries=0)


def test_partial_failure_is_explicit(chunks):
    result = verify_claim("claim", chunks, llm=FakeLLM(fail_paper="synthetic-a"), retries=0)
    assert result.analyses[0].status == "failed"
    assert result.analyses[0].error == "TimeoutError"
    assert [s.paper_id for s in result.sources] == ["synthetic-b"]
    assert any("failed" in note for note in result.limitations)


@pytest.mark.parametrize("all_insufficient", [False, True])
def test_treatment_only_evidence_is_excluded_from_prevention_synthesis(chunks, all_insufficient):
    class PreventionLLM(FakeLLM):
        def with_structured_output(self, schema, **kwargs):
            parent = super().with_structured_output(schema, **kwargs)

            def invoke(messages):
                if schema is PaperAssessment:
                    paper = json.loads(messages[1].content.split("Paper evidence JSON:\n")[1])
                    assessment = parent.invoke(messages)
                    if all_insufficient or paper["paper_id"] == "synthetic-a":
                        assessment["stance"] = "insufficient"
                        assessment["explanation"] = "This study assessed treatment after onset."
                        assessment["findings"] = []
                    return assessment
                assert not all_insufficient, "Synthesis should be skipped without eligible evidence"
                payload = json.loads(messages[1].content)
                assert [a["paper_id"] for a in payload["analyses"]] == ["synthetic-b"]
                assert [s["paper_id"] for s in payload["sources"]] == ["synthetic-b"]
                return parent.invoke(messages)

            return SimpleNamespace(invoke=invoke)

    result = verify_claim("Synthetic prevention claim", chunks, llm=PreventionLLM(), retries=0)
    assert len(result.analyses) == 2
    assert result.analyses[0].assessment.stance == "insufficient"
    assert [s.paper_id for s in result.sources] == ([] if all_insufficient else ["synthetic-b"])
    assert result.verdict == ("insufficient" if all_insufficient else "mixed")


def test_scoped_finding_survives_insufficient_paper_stance(chunks):
    class ScopedLLM(FakeLLM):
        def with_structured_output(self, schema, **kwargs):
            parent = super().with_structured_output(schema, **kwargs)

            def invoke(messages):
                result = parent.invoke(messages)
                if schema is PaperAssessment:
                    result["stance"] = "insufficient"
                return result

            return SimpleNamespace(invoke=invoke)

    result = verify_claim("Synthetic claim", chunks, llm=ScopedLLM(), retries=0)
    assert all(a.assessment.stance == "insufficient" for a in result.analyses)
    assert [source.paper_id for source in result.sources] == ["synthetic-a", "synthetic-b"]
    assert result.evidence_summary[0].citation_ids == ["1", "2"]


def test_empty_and_failed_evidence_need_no_aggregate_call(chunks):
    result = verify_claim("claim", [], llm=object())
    assert result.verdict == "insufficient"
    assert result.confidence == "low"
    assert not result.sources
    result = verify_claim("claim", chunks, llm=FakeLLM(bad_paper=True), retries=0)
    assert result.verdict == "insufficient"
    assert all(a.status == "failed" for a in result.analyses)
    with pytest.raises(ValueError, match="blank"):
        verify_claim(" ", [], llm=object())


def test_abstract_adapter_preserves_metadata_and_skips_pdf_only():
    records = abstract_chunks({"data": [
        {"paperId": "p", "title": "T", "abstract": "Evidence", "authors": [{"name": "A"}],
         "year": 2020, "externalIds": {"DOI": "test-doi"}},
        {"paperId": "pdf", "abstract": None, "openAccessPdf": {"url": "https://example.test/paper.pdf"}},
    ]})
    assert len(records) == 1
    assert records[0].doi == "test-doi"
    assert records[0].authors == ["A"]
    assert records[0].page_number is None
    assert records[0].metadata["evidence_type"] == "abstract"


def test_malformed_model_response_retries_then_succeeds(chunks):
    class OnceMalformed(FakeLLM):
        calls = 0

        def with_structured_output(self, schema, **kwargs):
            delegate = super().with_structured_output(schema, **kwargs)
            def invoke(messages):
                if schema is PaperAssessment:
                    self.calls += 1
                    if self.calls == 1:
                        return {"stance": "invalid"}
                return delegate.invoke(messages)
            return SimpleNamespace(invoke=invoke)

    llm = OnceMalformed()
    result = verify_claim("claim", chunks[:1], llm=llm, retries=1)
    assert llm.calls == 2
    assert result.analyses[0].status == "ok"
    assert result.sources
    assert any("SYNTHETIC TEST" in note for note in result.limitations)


def test_model_cannot_return_substantive_stance_without_evidence(chunks):
    class Unsupported(FakeLLM):
        def with_structured_output(self, schema, **kwargs):
            return SimpleNamespace(invoke=lambda messages: {
                "stance": "supports", "explanation": "Unsupported assertion", "findings": []})
    paper = group_chunks(chunks)[0][0]
    with pytest.raises(ValueError, match="no cited evidence"):
        analyze_single_paper("claim", paper, llm=Unsupported())
