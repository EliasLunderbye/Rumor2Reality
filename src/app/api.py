"""Local HTTP interface to the same RAG service used by Gradio."""
import logging
import os
from pathlib import Path
from threading import BoundedSemaphore
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app import service
from app.llm.schemas import FinalAssessment

logger = logging.getLogger(__name__)
app = FastAPI(title="Rumor2Reality", version="0.1.0",
              description="Verify scientific claims against retrieved evidence. Research prototype.")
# Limit expensive inference per process. Run locally with one worker by default.
_capacity = BoundedSemaphore(2)


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    top_k: int = Field(default=4, ge=1, le=20, strict=True)


class VerifyResponse(BaseModel):
    assessment: FinalAssessment
    timings: dict[str, float]
    model: str
    retrieved_chunks: int


def get_verifier():
    """Dependency boundary for offline HTTP tests without replacing the graph."""
    return service.run_pipeline


@app.get("/health")
def health():
    """Process liveness only; does not contact Ollama or retrieval services."""
    return {"status": "ok"}


@app.post("/verify", response_model=VerifyResponse, responses={
    503: {"description": "Paper search throttled, verification busy, or local index unavailable."}
})
def verify(request: VerifyRequest, verifier=Depends(get_verifier)):
    if not _capacity.acquire(blocking=False):
        raise HTTPException(status_code=503, detail={
            "code": "busy", "message": "Verification capacity is full. Try again shortly."
        }, headers={"Retry-After": "5"})
    try:
        # Server-owned configuration: HTTP clients cannot choose filesystem paths.
        index_path = os.getenv("RUMOR2REALITY_VECTOR_STORE", "").strip()
        options = {}
        if index_path:
            path = Path(index_path).expanduser()
            if not all((path / name).is_file() for name in ("index.faiss", "metadata.json")):
                raise HTTPException(status_code=503, detail={
                    "code": "local_index_unavailable",
                    "message": "The configured local evidence index is missing. Build it before verifying claims."
                })
            options = {"evidence_source": "Existing vector store", "vector_store_path": str(path)}
        result = verifier(request.claim, top_k=request.top_k, **options)
        return VerifyResponse(assessment=result["assessment"], timings=result["timings"],
                              model=result["metadata"]["model"],
                              retrieved_chunks=result["metadata"]["retrieved_chunks"])
    except HTTPException:
        raise
    except service.PaperDiscoveryRateLimited as exc:
        raise HTTPException(status_code=503, detail={
            "code": "paper_search_rate_limited",
            "message": "Semantic Scholar temporarily throttled paper search. Try again later or use a local evidence index."
        }, headers={"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None) from None
    except service.VerificationUnavailable:
        logger.warning("Verification dependency failed", exc_info=True)
        raise HTTPException(status_code=502, detail={
            "code": "verification_unavailable",
            "message": "Evidence retrieval or model processing failed. The claim has not been assessed."
        }) from None
    except Exception:
        logger.exception("Unexpected verification failure")
        raise HTTPException(status_code=500, detail={
            "code": "internal_error", "message": "Verification failed unexpectedly."
        }) from None
    finally:
        _capacity.release()
