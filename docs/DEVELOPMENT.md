# Independent development

This fork continues the 1RT730 team project. Original authorship and commit history
are preserved. Elias Lunderbye's original contribution focused on the UI and RAG
pipeline integration. Changes after the fork are recorded in commits and pull requests.
Engineering-foundation changes were prepared with Codex assistance.

## Set up on Windows

Install Python 3.12, then run from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ui,test]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux/macOS use `python3.12 -m venv .venv` and `.venv/bin/python`.
Activation is optional. Dependencies have version ranges in pyproject.toml, not a
fully pinned lockfile; future dependency resolution can differ.

## Test boundaries

Default tests use fake LLM responses and controlled retrieval/API responses while
exercising the real workflow. Outbound socket connections are blocked. No Ollama,
API key or model-weight download is needed to run them after dependencies install.
The real embedding-model test is skipped by default. Run it explicitly with:

```powershell
.\.venv\Scripts\python.exe -m pytest --run-model-tests -m model_download -q
```

This may download model weights. Passing offline tests does not establish live
model quality, citation support, retrieval relevance or external-service availability.
GitHub Actions runs dependency checks and the offline suite on pushes and PRs.

## Run the UI

Follow [the UI setup guide](../ui/rumor2reality/README.md) to configure Ollama and
install the selected model, then run:

```powershell
.\.venv\Scripts\python.exe ui/rumor2reality/app.py
```

Keep API keys in local environment variables, never in tracked files. `.env`
files are ignored but are not automatically loaded by this application.

## Next milestones

1. FastAPI endpoint around the existing workflow, with validation and error tests.
2. Docker packaging and documented Ollama connectivity.
3. Reviewed evaluation set measuring retrieval, citation support and latency.
4. Demonstration deployment with logging and resource limits.

Use feature branches and open PRs against **EliasLunderbye/Rumor2Reality:main**.
The `upstream` remote is the original team repository; `origin` is this fork.
No new license is granted here for the inherited team code.
