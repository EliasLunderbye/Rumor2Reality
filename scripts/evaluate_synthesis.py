"""Run fixed synthetic evidence through Ollama for manual citation review."""

import argparse
import json
from time import perf_counter

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
    (
        "Filter D reduces airborne particle counts in occupied classrooms.",
        (
            "A trial using the North School dataset measured lower airborne particle "
            "counts in occupied classrooms fitted with Filter D.",
            "A separate publication reanalyzed the same North School dataset and "
            "also reported lower airborne particle counts with Filter D in the "
            "same occupied classrooms.",
            "A study at South School found no measurable reduction with Filter D "
            "in occupied classrooms.",
            "A survey at West School measured how easy Filter D was to install "
            "but did not measure airborne particle counts.",
        ),
        "Identify the two North School publications as one underlying dataset. "
        "With all excerpts present, acknowledge the conflicting South School result "
        "and exclude the installation-only survey from citations.",
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", metavar="N",
                        help="Compare the first N excerpts in each fixed case")
    parser.add_argument("--case", type=int, choices=range(1, len(CASES) + 1),
                        help="Run only the selected case (1-based)")
    args = parser.parse_args()
    if args.budgets and any(n < 1 for n in args.budgets):
        parser.error("budgets must be positive")
    cases = CASES if args.case is None else (CASES[args.case - 1],)
    for claim, excerpts, rubric in cases:
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
        for budget in dict.fromkeys(min(n, len(chunks)) for n in (args.budgets or [len(chunks)])):
            print(json.dumps({"claim": claim, "manual_check": rubric, "excerpts": budget}), flush=True)
            started = perf_counter()
            result = verify_claim(claim, chunks[:budget], max_workers=1)
            print(json.dumps({
                "seconds": round(perf_counter() - started, 2),
                "cited_papers": len({source.paper_id for source in result.sources}),
                **result.model_dump(mode="json"),
            }, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
