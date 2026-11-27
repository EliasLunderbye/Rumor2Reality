# Local verification API

The FastAPI adapter uses `app.service`, the same RAG workflow as the Gradio UI.
This is a local research prototype, not a public hosted service.

## Install and start

To start live Semantic Scholar mode on Windows after installing the dependencies,
run `.\scripts\start-api.ps1` from the repository root. Start Ollama first.
The launcher prompts without echoing the key if it is missing from the current
shell. Open http://127.0.0.1:8000/docs to submit a claim.

From the repository root, using the Python 3.12 environment in the development guide:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[api,test]"
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for interactive OpenAPI documentation.
`GET /health` checks process liveness only. It does not check Ollama, installed
models, embeddings or Semantic Scholar availability.

## Verify a claim

Start Ollama with an installed compatible model first. The app requests an 8,192-token
context window by default. Set `OLLAMA_NUM_CTX` to override it. Set `OLLAMA_MODEL`,
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

The API uses Semantic Scholar discovery unless `RUMOR2REALITY_VECTOR_STORE` is set
in the server environment. In that case it exclusively uses that local index;
it never falls back to live discovery. Clients cannot select models,
change service URLs or supply filesystem paths. Existing-index mode remains
available through the UI and Python service.

## Work without a Semantic Scholar key

Place research PDFs containing selectable text in `data/raw/`. From the repository
root, build a local index once (rebuild when you change the papers):

```powershell
.\.venv\Scripts\python.exe -m app.demo ingest-pdf --input data/raw --output data/processed/local_index
```

The ingestion command uses `all-MiniLM-L6-v2`. Its first run may download weights;
subsequent runs can use the cached model. No Semantic Scholar requests are made.
Use only trusted indexes you built yourself.

Stop the existing API with Ctrl+C. In the same PowerShell window, start local mode:

```powershell
$env:EMBEDDING_MODEL = "all-MiniLM-L6-v2"
$env:RUMOR2REALITY_VECTOR_STORE = (Resolve-Path data/processed/local_index).Path
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Ollama must still be running with the configured `OLLAMA_MODEL` installed. Open
http://127.0.0.1:8000/docs and use the same `/verify` request body, choosing a claim
relevant to your PDFs. Inspect the returned excerpts and citations. Local mode
only searches your supplied collection; it does not represent all published evidence.
Missing index files return 503 with `local_index_unavailable`, without contacting
Semantic Scholar. `/health` remains a liveness check, not an index-readiness check.

To return to live discovery, remove the variable and restart the server:

```powershell
Remove-Item Env:RUMOR2REALITY_VECTOR_STORE
```

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

For a fixed evidence quality check with Ollama running, execute
`./.venv/Scripts/python.exe scripts/evaluate_synthesis.py` from the repository
root in PowerShell. The script uses synthetic excerpts and skips live paper
search. Review each response against the printed criterion.

Use `--budgets 2 4` to compare smaller and larger evidence sets. The runner
holds the input order fixed, reports runtime and distinct cited papers, and
includes two publications derived from the same dataset. This comparison
isolates the evidence supplied to the model from live search ranking. Check
whether citations support the statements and whether additional excerpts add
independent evidence or conflicting findings.

The live vitamin C baseline returned HTTP 200 and excluded the treatment-only
study from the verdict and confidence rationale. Manual review found two
remaining issues: the rationale treated reviews as independent without knowing
whether they shared trials, and the summary overstated a subgroup finding by
claiming the benefit occurred only in that subgroup.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

HTTP tests exercise validation, error codes, capacity release, empty evidence,
and citation preservation through the real LangGraph workflow with controlled
retrieval and a fake LLM. They do not establish real-model answer quality.
