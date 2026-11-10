# Local verification API

The FastAPI adapter uses `app.service`, the same RAG workflow as the Gradio UI.
This is a local research prototype, not a public hosted service.

## Install and start

From the repository root, using the Python 3.12 environment in the development guide:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[api,test]"
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for interactive OpenAPI documentation.
`GET /health` checks process liveness only. It does not check Ollama, installed
models, embeddings or Semantic Scholar availability.

## Verify a claim

Start Ollama with an installed compatible model first. Set `OLLAMA_MODEL`,
`OLLAMA_BASE_URL` and `OLLAMA_TIMEOUT` in the server environment as needed.
An optional `SEMANTIC_SCHOLAR_API_KEY` can help with paper-discovery rate limits.
`EMBEDDING_MODEL` controls retrieval embeddings. Environment files are not loaded
automatically. The first real request may download embedding-model weights.

```powershell
$body = @{ claim = "Vitamin C prevents the common cold"; top_k = 4 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/verify `
  -ContentType 'application/json' -Body $body
```

The request accepts only:

- `claim`: a non-blank string, trimmed, maximum 2,000 characters.
- `top_k`: an integer from 1 to 20 (default 4), counting retrieved evidence chunks.

The API currently uses Semantic Scholar discovery. Clients cannot select models,
change service URLs or supply filesystem paths. Existing-index mode remains
available through the UI and Python service.

The response includes `assessment` (verdict, confidence, evidence summaries,
source citations, analyses and limitations), stage `timings`, `model`, and
`retrieved_chunks`. Confidence is a model-generated category, not a calibrated
probability. Sources may be abstracts rather than full papers.

## Outcomes

| HTTP status | Meaning |
| --- | --- |
| 200 | Assessment completed; no retrieved evidence returns `insufficient` |
| 422 | Invalid input; verification was not started |
| 502 | Paper discovery or model processing failed; no assessment returned |
| 503 | Both per-process verification slots are occupied; retry after 5 seconds |
| 500 | Unexpected internal failure; details stay in server logs |

The synchronous workflow runs outside the ASGI event loop. Two verification
slots per process limit expensive concurrent work. Use one server worker for
this local setup; adding workers multiplies that limit. Ollama has a per-request
timeout and paper discovery has bounded retries, but there is no total job
deadline or cancellation on client disconnect. Requests may take substantially
longer than a single model call. Authentication, deployment, request-body size
limits at the proxy and production monitoring are future work; keep the server
bound to localhost.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

HTTP tests exercise validation, error codes, capacity release, empty evidence,
and citation preservation through the real LangGraph workflow with controlled
retrieval and a fake LLM. They do not establish real-model answer quality.
