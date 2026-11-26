"""Run fixed synthetic evidence through Ollama for manual citation review."""

import json

from app.graph.workflow import verify_claim
from app.llm.schemas import RetrievedChunk


CASES = (
    (
        "Filter A reduces particle counts in every building.",
        (
            "A controlled trial in classrooms found no reduction in particle counts.",
            "A controlled trial in workshops found a reduction in particle counts. "
            "It did not test other building types.",
        ),
        "Limit the reported benefit to the workshops studied. Do not infer that "
        "workshops are the only buildings where a benefit is possible.",
    ),
    (
        "Filter B reduces particle counts in classrooms.",
        (
            "A review of classroom trials found lower particle counts with Filter B. "
            "The abstract does not list the included trials.",
            "A second review of classroom trials reported lower particle counts with "
            "Filter B. The abstract does not list the included trials.",
        ),
        "Report agreement without claiming the reviews used independent trials. "
        "Treat overlap as unknown when explaining confidence.",
    ),
    (
        "Filter C reduces airborne particle counts in classrooms.",
        (
            "A classroom survey found that occupants preferred the appearance of "
            "Filter C. It did not measure airborne particle counts.",
        ),
        "Return insufficient evidence and exclude the appearance survey from "
        "citation and confidence support for particle counts.",
    ),
)


def main():
    for claim, excerpts, rubric in CASES:
        print(json.dumps({"claim": claim, "manual_check": rubric}), flush=True)
        chunks = [
            RetrievedChunk(
                paper_id=f"synthetic-{i}",
                chunk_id=f"synthetic-{i}:abstract",
                title=f"Synthetic study {i}",
                text=excerpt,
                metadata={"synthetic": True, "evidence_type": "abstract"},
            )
            for i, excerpt in enumerate(excerpts, start=1)
        ]
        result = verify_claim(claim, chunks)
        print(result.model_dump_json(), flush=True)


if __name__ == "__main__":
    main()
