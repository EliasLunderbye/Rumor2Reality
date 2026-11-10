from threading import BoundedSemaphore

from fastapi.testclient import TestClient
import pytest

from app import api, service
from tests.test_pipeline import FakeLLM


@pytest.fixture(scope="session")
def http_client():
    # Start the event loop before the per-test network guard. Windows creates
    # an internal loopback socketpair for its wakeup pipe, not an HTTP request.
    with TestClient(api.app) as client:
        yield client


@pytest.fixture
def client(http_client, monkeypatch):
    monkeypatch.setattr(api, "_capacity", BoundedSemaphore(2))
    try:
        yield http_client
    finally:
        api.app.dependency_overrides.clear()


@pytest.fixture
def empty_discovery(monkeypatch):
    from app.rag import collect_papers
    monkeypatch.setattr(collect_papers, "fetch_papers", lambda *a, **kw: [])
    api.app.dependency_overrides[api.get_verifier] = lambda: (
        lambda claim, **kw: service.run_pipeline(claim, llm=FakeLLM(), **kw)
    )


def test_health_and_openapi(client):
    assert client.get("/health").json() == {"status": "ok"}
    schema = client.get("/openapi.json").json()
    assert "/verify" in schema["paths"]


@pytest.mark.parametrize("payload", [
    {}, {"claim": "   "}, {"claim": "x" * 2001}, {"claim": 1},
    {"claim": "A claim", "top_k": 0}, {"claim": "A claim", "top_k": 21},
    {"claim": "A claim", "top_k": True}, {"claim": "A claim", "top_k": "4"},
    {"claim": "A claim", "vector_store_path": "private/path"},
])
def test_invalid_input_never_calls_verifier(client, payload):
    def unexpected(*a, **kw):
        pytest.fail("Invalid input reached verification")
    api.app.dependency_overrides[api.get_verifier] = lambda: unexpected
    assert client.post("/verify", json=payload).status_code == 422


def test_empty_evidence_uses_real_graph(client, empty_discovery):
    response = client.post("/verify", json={"claim": "  A claim  "})
    assert response.status_code == 200
    result = response.json()
    assert result["assessment"]["claim"] == "A claim"
    assert result["assessment"]["verdict"] == "insufficient"
    assert result["assessment"]["sources"] == []
    assert result["retrieved_chunks"] == 0
    assert result["timings"]["total_seconds"] >= 0


def test_api_preserves_citations_from_real_graph(client, monkeypatch):
    from app.llm.schemas import RetrievedChunk
    from app.rag import collect_papers, ingest, retriever
    chunk = RetrievedChunk(paper_id="paper", chunk_id="paper:abstract", title="Paper",
                           text="Synthetic evidence.", metadata={"evidence_type": "abstract"})
    monkeypatch.setattr(collect_papers, "fetch_papers", lambda *a, **kw: [])
    monkeypatch.setattr(collect_papers, "papers_to_chunks", lambda *a: [chunk])
    monkeypatch.setattr(ingest, "ingest_chunks", lambda *a: None)
    class Retriever:
        def retrieve(self, claim, k):
            assert k == 3
            return [chunk]
    monkeypatch.setattr(retriever, "create_retriever", lambda *a: Retriever())
    api.app.dependency_overrides[api.get_verifier] = lambda: (
        lambda claim, **kw: service.run_pipeline(claim, llm=FakeLLM(), **kw)
    )
    response = client.post("/verify", json={"claim": "A claim", "top_k": 3})
    assert response.status_code == 200
    assessment = response.json()["assessment"]
    assert assessment["sources"][0]["excerpt"] == chunk.text
    assert assessment["sources"][0]["citation_id"] == "1"
    assert assessment["evidence_summary"][0]["citation_ids"] == ["1"]


@pytest.mark.parametrize("error,status,code", [
    (service.VerificationUnavailable("private upstream details"), 502, "verification_unavailable"),
    (RuntimeError("private internal details"), 500, "internal_error"),
])
def test_errors_are_safe_and_release_capacity(client, error, status, code):
    def fail(*a, **kw):
        raise error
    api.app.dependency_overrides[api.get_verifier] = lambda: fail
    for _ in range(3):
        response = client.post("/verify", json={"claim": "A claim"})
        assert response.status_code == status
        assert response.json()["detail"]["code"] == code
        assert "private" not in response.text


def test_capacity_exhaustion_returns_retry_after(client):
    api._capacity.acquire()
    api._capacity.acquire()
    try:
        response = client.post("/verify", json={"claim": "A claim"})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"
    finally:
        api._capacity.release()
        api._capacity.release()


def test_discovery_failure_maps_to_502(client, empty_discovery, monkeypatch):
    from app.rag import collect_papers
    def fail(*a, **kw):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(collect_papers, "fetch_papers", fail)
    assert client.post("/verify", json={"claim": "A claim"}).status_code == 502
