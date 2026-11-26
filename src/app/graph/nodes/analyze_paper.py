from langchain_core.messages import SystemMessage, HumanMessage
from app.llm.client import get_llm
from app.llm.schemas import ScientificPaper, PaperAssessment, PaperAnalysis
from app.llm.prompts import PAPER_ANALYSIS_SYSTEM_PROMPT
from app.llm.citations import validate_analysis


def analyze_single_paper(claim: str, paper: ScientificPaper, *, llm=None,
                         retry_feedback=None, recheck_findings=False) -> PaperAnalysis:
    if not paper.chunks:
        return PaperAnalysis(paper_id=paper.paper_id, title=paper.title,
                             assessment=PaperAssessment(stance="insufficient",
                                                        explanation="No evidence chunks provided."))
    if any(chunk.paper_id != paper.paper_id for chunk in paper.chunks):
        raise ValueError("Chunk belongs to a different paper")
    model = (llm if llm is not None else get_llm()).with_structured_output(
        PaperAssessment, method="json_schema")
    messages = [
        SystemMessage(content=PAPER_ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=f"Scientific claim:\n{claim}\n\nPaper evidence JSON:\n{paper.model_dump_json()}"),
    ]
    if retry_feedback:
        messages.append(HumanMessage(content=(
            "The previous response failed validation. Return a complete structured assessment "
            "with findings containing exact chunk IDs from the supplied paper. "
            f"Retry diagnostic: {retry_feedback}")))
    if recheck_findings:
        messages.append(HumanMessage(content=(
            "Recheck the excerpts for an explicitly reported result measuring the claimed "
            "outcome for the stated intervention. A null result is a finding. A result "
            "in one setting is a finding even for a claim about all settings. A review "
            "reporting a result counts even if it does not list its trials. Extract the "
            "reported observation first and attribute it to the excerpt. If present, "
            "put it in findings with the exact chunk ID even if stance is insufficient. "
            "Discuss causal uncertainty and missing methods in limitations. Keep findings "
            "empty if the excerpts report only an unrelated outcome or intervention.")))
    assessment = model.invoke(messages)
    assessment = PaperAssessment.model_validate(assessment)
    result = PaperAnalysis(paper_id=paper.paper_id, title=paper.title, assessment=assessment)
    validate_analysis(paper, result)
    return result
