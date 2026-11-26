"""Fetch papers from Semantic Scholar API."""
import os
import json
import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import Lock
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from app.llm.schemas import RetrievedChunk

_request_lock = Lock()
_next_request_at = 0.0


class PaperSearchRateLimited(RuntimeError):
    """Semantic Scholar throttled the paper search after bounded retries."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _open_search(request):
    global _next_request_at
    # One process shares the key across concurrent verification requests.
    with _request_lock:
        delay = _next_request_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _next_request_at = time.monotonic() + 1.1
        return urlopen(request, timeout=30)


def fetch_papers(query: str, limit: int = 10) -> list[dict]:
    """Fetch papers from Semantic Scholar API.
    
    Args:
        query: Search query (e.g., "vitamin C cold")
        limit: Maximum number of papers to fetch
        
    Returns:
        List of paper dictionaries with metadata
    """
    params = urlencode({
        "query": query,
        "limit": limit,
        "fields": "paperId,title,abstract,authors,year,url,externalIds,isOpenAccess,openAccessPdf"
    })
    headers = {"User-Agent": "scientific-claim-project/0.1"}
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    
    request = Request(f"https://api.semanticscholar.org/graph/v1/paper/search?{params}", headers=headers)
    for attempt in range(3):
        try:
            with _open_search(request) as response:
                payload = json.load(response)
            break
        except HTTPError as exc:
            if exc.code != 429:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            exc.close()
            delay = 5 * (2 ** attempt)
            if retry_after:
                try:
                    delay = max(0, float(retry_after))
                except ValueError:
                    try:
                        delay = max(0, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                    except (ValueError, TypeError, OverflowError):
                        pass
            # Do not retry earlier than the server allows or block the UI indefinitely.
            if attempt == 2 or delay > 30:
                raise PaperSearchRateLimited(
                    "Semantic Scholar is rate-limiting paper searches (HTTP 429). "
                    "Try again later, set SEMANTIC_SCHOLAR_API_KEY before starting the app, "
                    "or choose Existing vector store to use previously ingested papers/PDFs. "
                    "The claim has not been assessed.",
                    retry_after=math.ceil(delay) if retry_after and math.isfinite(delay) else None,
                ) from exc
            time.sleep(delay)
    
    papers = payload.get("data", [])
    return papers if isinstance(papers, list) else []


def papers_to_chunks(papers: list[dict]) -> list[RetrievedChunk]:
    """Convert papers to RetrievedChunk format.
    
    Uses abstract as the evidence chunk for each paper.
    """
    chunks = []
    for paper in papers:
        abstract = paper.get("abstract")
        paper_id = paper.get("paperId")
        
        if not abstract or not abstract.strip() or not paper_id:
            continue
        
        authors = [
            a["name"] if isinstance(a, dict) else a 
            for a in paper.get("authors", [])
        ]
        
        chunk = RetrievedChunk(
            paper_id=paper_id,
            chunk_id=f"{paper_id}:abstract",
            text=abstract,
            title=paper.get("title", "Untitled"),
            authors=authors,
            doi=(paper.get("externalIds") or {}).get("DOI"),
            url=paper.get("url"),
            year=paper.get("year"),
            score=paper.get("relevance_score"),
            metadata={
                "evidence_type": "abstract",
                "provider": "Semantic Scholar",
                "is_open_access": paper.get("isOpenAccess", False),
                "open_access_pdf": (paper.get("openAccessPdf") or {}).get("url")
            }
        )
        chunks.append(chunk)
    
    return chunks
