import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.llm.client import get_llm
from app.llm.citations import source_registry
from app.llm.prompts import AGGREGATION_SYSTEM_PROMPT
from app.llm.schemas import AggregationAssessment, EvidenceEntry, FinalAssessment


def aggregate_evidence(claim, papers, analyses, *, limitations=(), evidence_cutoff=None, llm=None):
    eligible = [analysis for analysis in analyses
                if analysis.status == "ok" and analysis.assessment.findings]
    sources = source_registry(papers, eligible)
    notes = list(limitations)
    if any(c.metadata.get("synthetic") for p in papers for c in p.chunks):
        notes.append("SYNTHETIC TEST EVIDENCE: this output is not a scientific assessment of real studies.")
    if evidence_cutoff is None:
        notes.append("Evidence cutoff is unknown; retrieved evidence may not be comprehensive.")
    if any(c.metadata.get("evidence_type") == "abstract" for p in papers for c in p.chunks):
        notes.append("Some or all evidence consists of abstracts rather than full papers.")
    failures = [a.paper_id for a in analyses if a.status == "failed"]
    if failures:
        notes.append("Paper analysis failed for: " + ", ".join(failures))
    if not sources:
        assessment = AggregationAssessment(verdict="insufficient", confidence="low",
            confidence_reason="No validated cited findings are available.",
            limitations=["No usable cited evidence was available for synthesis."])
    else:
        model = (llm if llm is not None else get_llm()).with_structured_output(
            AggregationAssessment, method="json_schema")
        payload = {"claim": claim, "analyses": [a.model_dump() for a in eligible],
                   "sources": [s.model_dump() for s in sources],
                   "limitations": notes, "evidence_cutoff": evidence_cutoff}
        known = {s.citation_id for s in sources}
        payload["allowed_citation_ids"] = sorted(known)
        for attempt in range(2):
            assessment = AggregationAssessment.model_validate(model.invoke([
                SystemMessage(content=AGGREGATION_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload)),
            ]))
            if assessment.evidence_summary and all(
                item.statement.strip() and item.citation_ids and set(item.citation_ids) <= known
                for item in assessment.evidence_summary
            ):
                break
            if attempt == 1:
                raise ValueError("Aggregate contains an empty statement, missing citation, or unknown citation")
            payload["correction"] = (
                "The previous response failed citation validation. Return a nonempty "
                "evidence_summary with a nonblank statement and at least one citation ID "
                "per item. Copy IDs exactly from allowed_citation_ids."
            )
    cited_ids = {citation_id for item in assessment.evidence_summary
                 for citation_id in item.citation_ids}
    cited_sources = [source for source in sources if source.citation_id in cited_ids]
    if cited_sources and all(source.metadata.get("evidence_type") == "abstract"
                             for source in cited_sources):
        if assessment.confidence == "high":
            assessment.confidence = "moderate"
        assessment.confidence_reason = (
            "The assessment relies on cited abstracts. Full methods and any overlap "
            "in underlying studies could not be verified."
        )
    return FinalAssessment(
        claim=claim, **assessment.model_dump(exclude={"limitations"}),
        limitations=list(dict.fromkeys(notes + assessment.limitations +
            [lim for a in analyses for lim in a.assessment.limitations])),
        evidence=[EvidenceEntry(paper_id=a.paper_id, stance=a.assessment.stance,
            summary=a.assessment.explanation, study_design=a.assessment.study_design,
            status=a.status) for a in analyses],
        sources=cited_sources,
        analyses=analyses, evidence_cutoff=evidence_cutoff,
    )
