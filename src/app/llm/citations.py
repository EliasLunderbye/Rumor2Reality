"""Citation integrity checks; these establish provenance, not semantic entailment."""
from app.llm.schemas import PaperAnalysis, ScientificPaper, SourceReference


class CitationValidationError(ValueError):
    """A citation validation failure with a fixed, safe diagnostic."""


def validate_analysis(paper: ScientificPaper, analysis: PaperAnalysis) -> None:
    if analysis.paper_id != paper.paper_id:
        raise ValueError("Analysis belongs to a different paper")
    chunks = {chunk.chunk_id: chunk for chunk in paper.chunks}
    if len(chunks) != len(paper.chunks):
        raise ValueError("Duplicate chunk identifiers within paper")
    if any(chunk.paper_id != paper.paper_id for chunk in paper.chunks):
        raise ValueError("Chunk belongs to a different paper")
    for finding in analysis.assessment.findings:
        if finding.chunk_id not in chunks or not finding.statement.strip():
            raise CitationValidationError(
                "Unknown citation or empty finding. Use a nonblank statement and copy an exact chunk_id."
            )
    if analysis.assessment.stance != "insufficient" and not analysis.assessment.findings:
        raise CitationValidationError(
            "Substantive stance has no cited evidence. Add a cited finding or use insufficient."
        )


def source_registry(papers: list[ScientificPaper], analyses: list[PaperAnalysis]) -> list[SourceReference]:
    by_id = {paper.paper_id: paper for paper in papers}
    sources = []
    seen = set()
    for analysis in analyses:
        paper = by_id[analysis.paper_id]
        validate_analysis(paper, analysis)
        if analysis.status != "ok":
            continue
        for finding in analysis.assessment.findings:
            key = (paper.paper_id, finding.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            chunk = next(c for c in paper.chunks if c.chunk_id == finding.chunk_id)
            sources.append(SourceReference(
                citation_id=str(len(sources) + 1), paper_id=paper.paper_id,
                title=paper.title, authors=paper.authors, doi=paper.doi,
                url=paper.url, year=paper.year, chunk_id=chunk.chunk_id,
                page_number=chunk.page_number, excerpt=chunk.text, metadata=chunk.metadata,
            ))
    return sources
