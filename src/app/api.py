"""Local HTTP interface to the same RAG service used by Gradio."""
import logging
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


@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest, verifier=Depends(get_verifier)):
    if not _capacity.acquire(blocking=False):
        raise HTTPException(status_code=503, detail={
            "code": "busy", "message": "Verification capacity is full. Try again shortly."
        }, headers={"Retry-After": "5"})
    try:
        result = verifier(request.claim, top_k=request.top_k)
        return VerifyResponse(assessment=result["assessment"], timings=result["timings"],
                              model=result["metadata"]["model"],
                              retrieved_chunks=result["metadata"]["retrieved_chunks"])
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
