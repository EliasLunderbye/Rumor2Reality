from __future__ import annotations

import html
import json
import re
from typing import Any

import gradio as gr

from pipeline import DEFAULT_MODEL, pipeline_events
from urllib.parse import urlsplit
from app.service import PaperDiscoveryRateLimited


DEFAULT_CLAIM = "Vitamin C prevents the common cold"
READY_ANSWER = """
### Your assessment will appear here

Check a claim to see a verdict, confidence, and a summary linked to the evidence below.
"""
READY_TIMING = {
    "status": "ready",
    "next_action": "enter a claim and run the check",
}


def _empty_sources_html() -> str:
    return """
    <div class="r2r-empty-evidence">
      <strong>No evidence retrieved yet.</strong>
      <span>Run the check to see the papers selected by the retrieval step.</span>
    </div>
    """


def _study_type_label(value: str) -> str:
    """Keep the first descriptive phrase without inferring a study design."""
    label = re.split(r"[(;\n]|\.\s", value.strip(), maxsplit=1)[0].strip()
    if not label or label.lower() == "unknown":
        return "Study type unspecified"
    if len(label) > 48:
        label = label[:45].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
    return label[0].upper() + label[1:]


def _sources_cards_html(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return _empty_sources_html()

    cards = []
    for source in sources:
        rank = html.escape(str(source["rank"]))
        title = html.escape(str(source["title"]))
        year = html.escape(str(source["year"]))
        score = source.get("relevance")
        relevance = f"Similarity: {float(score):.3f}" if score is not None else "Similarity unavailable"
        reason = html.escape(str(source["why_selected"]))
        raw_url = str(source.get("url") or "")
        url = html.escape(raw_url, quote=True) if urlsplit(raw_url).scheme in {"http", "https"} else ""
        stance = html.escape(str(source.get("stance", "Background")))
        study_detail = str(source.get("study_type") or "Unknown")
        study_type = html.escape(_study_type_label(study_detail))
        study_title = html.escape(study_detail, quote=True)
        page = str(source.get("page") or "").strip()
        page_html = (
            f"<span>p. {html.escape(page)}</span>"
            if page.lower() not in {"", "?", "unknown", "none", "n/a"} else ""
        )
        section = html.escape(str(source.get("section", "Unknown section")))
        snippet = html.escape(str(source.get("snippet", "")))
        source_link = (f'<a href="{url}" target="_blank" rel="noopener noreferrer">Open paper ↗</a>'
                       if url else "")
        cards.append(
            f"""
            <article class="r2r-source-card">
              <div class="r2r-source-topline">
                <span class="r2r-source-rank">#{rank}</span>
                <span class="r2r-source-score">{relevance}</span>
              </div>
              <h3>{title} <span>({year})</span></h3>
              <div class="r2r-source-meta">
                <span>{stance}</span>
                <span title="{study_title}">{study_type}</span>
                {page_html}
              </div>
              <p>{reason}</p>
              <details class="r2r-evidence-excerpt" id="r2r-citation-{rank}">
                <summary>Show evidence excerpt <span>· {section}</span></summary>
                <blockquote>{snippet}</blockquote>
              </details>
              {source_link}
            </article>
            """
        )
    return f"""<div class="r2r-sources-grid">{''.join(cards)}</div>"""


def _progress_html(active_step: str) -> str:
    labels = {
        "scrape": "Finding research papers",
        "retrieve": "Selecting relevant excerpts",
        "generate": "Comparing the evidence",
        "done": "Live check complete",
        "demo": "Recorded demo · Saved result, no live search",
        "limited": "Live search temporarily unavailable",
        "failed": "Check stopped",
    }
    return (f'<div class="r2r-status" role="status">{labels[active_step]}</div>'
            if active_step in labels else "")


def _debug_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, ensure_ascii=True)


def _inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def _linked_finding(item: str, source_ids: set[str]) -> str:
    match = re.fullmatch(r"(.*) \[([\d, ]+)\]", item, flags=re.DOTALL)
    if not match:
        return html.escape(item)
    statement, citations = match.groups()
    links = []
    for citation in citations.split(","):
        citation = citation.strip()
        if citation in source_ids:
            links.append(f'<a class="r2r-citation" href="#r2r-citation-{citation}" '
                         f'aria-label="Read cited excerpt {citation}">{citation}</a>')
        else:
            links.append(html.escape(citation))
    return f"{html.escape(statement)} {' '.join(links)}"


def _answer_html(
    markdown_text: str,
    tone: str = "ready",
    verdict: dict[str, str] | None = None,
    confidence: dict[str, str] | None = None,
    evidence_summary: list[str] | None = None,
    limitations: list[str] | None = None,
    source_ids: set[str] | None = None,
    analyses: list[dict[str, Any]] | None = None,
) -> str:
    blocks: list[str] = []
    for raw_block in markdown_text.strip().split("\n\n"):
        block = raw_block.strip()
        if not block:
            continue
        if block.startswith("### "):
            blocks.append(f"<h3>{_inline_markdown(block[4:])}</h3>")
        else:
            lines = [_inline_markdown(line) for line in block.splitlines()]
            blocks.append(f"<p>{'<br>'.join(lines)}</p>")

    findings = evidence_summary or []
    takeaway = (
        f'<p class="r2r-takeaway">{_linked_finding(findings[0], source_ids or set())}</p>'
        if findings else ""
    )
    verdict_html = ""
    if verdict:
        label = html.escape(verdict.get("label", "Not enough evidence"))
        rationale = html.escape(verdict.get("rationale", ""))
        rationale_html = f"<small>{rationale}</small>" if rationale else ""
        verdict_tone = html.escape(verdict.get("tone", "caution"))
        verdict_html = f"""
        <div class="r2r-verdict r2r-verdict-{verdict_tone}">
          <span>{label}</span>
          <small>Based on the evidence reviewed</small>
          {takeaway}
          {rationale_html}
        </div>
        """

    confidence_html = ""
    if confidence:
        label = html.escape(confidence.get("label", "Low"))
        rationale = html.escape(confidence.get("rationale", ""))
        confidence_html = f"""
        <div class="r2r-confidence">
          <strong>Confidence: {label}</strong>
          <span>{rationale}</span>
        </div>
        """

    summary_html = ""
    if findings[1:]:
        items = "".join(f"<li>{_linked_finding(item, source_ids or set())}</li>"
                        for item in findings[1:])
        summary_html = f"""
        <div class="r2r-output-list">
          <strong>Other findings</strong>
          <ul>{items}</ul>
        </div>
        """

    limitations_html = ""
    paper_notes = [
        (analysis["title"], analysis["assessment"]["limitations"])
        for analysis in (analyses or [])
        if analysis["assessment"]["limitations"]
    ]
    attributed = {note for _, notes in paper_notes for note in notes}
    general_notes = [note for note in (limitations or []) if note not in attributed]
    if general_notes or paper_notes:
        sections = []
        if general_notes:
            items = "".join(f"<li>{html.escape(note)}</li>" for note in general_notes)
            sections.append(f"<strong>Overall assessment</strong><ul>{items}</ul>")
        for title, notes in paper_notes:
            items = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
            sections.append(f"<strong>{html.escape(title)}</strong><ul>{items}</ul>")
        limitations_html = f"""
        <details class="r2r-output-list r2r-limitations">
          <summary>Limitations by assessment and paper</summary>
          {''.join(sections)}
        </details>
        """

    content = "".join(blocks)
    return (
        f"""<div class="r2r-answer-card r2r-answer-{tone}">"""
        f"{verdict_html}{confidence_html}{summary_html}{content}{limitations_html}</div>"
    )


def analyse_claim(claim: str, top_k: int, model_name: str, evidence_source="Semantic Scholar",
                  vector_store_path="", progress=gr.Progress()):
    claim = (claim or "").strip()
    if not claim:
        yield (
            _progress_html("ready"),
            _answer_html("### Add a rumor or claim\n\nPaste a challenge prompt, then run the check.", "warning"),
            _empty_sources_html(),
            READY_TIMING,
            "",
        )
        return

    messages = {"scrape": "Loading candidate evidence...", "retrieve": "Embedding and retrieving evidence...",
                "generate": "Analyzing papers and synthesizing cited findings..."}
    try:
        def report_paper(index, total):
            progress((index - 1, total), desc=f"Analyzing paper {index} of {total}")

        for stage, result in pipeline_events(
            claim, int(top_k), model_name, evidence_source, vector_store_path,
            paper_progress=report_paper, reuse_paper_search=True,
        ):
            if stage != "done":
                yield (_progress_html(stage), _answer_html("### Checking evidence\n\n" + messages[stage], "working"),
                       _empty_sources_html(), {"status": messages[stage]}, "")
            else:
                metadata = result["metadata"]
                excerpts = metadata["retrieved_chunks"]
                papers = metadata["cited_papers"]
                counts = (f"{excerpts} {'excerpt' if excerpts == 1 else 'excerpts'} retrieved · "
                          f"{papers} {'paper' if papers == 1 else 'papers'} cited")
                yield (_progress_html("done"), _answer_html(counts, "done",
                       verdict=result["verdict"], confidence=result["confidence"],
                       evidence_summary=result["evidence_summary"], limitations=result["limitations"],
                       source_ids={str(source["rank"]) for source in result["sources"]},
                       analyses=result["assessment"]["analyses"]),
                       _sources_cards_html(result["sources"]),
                       {"timings": result["timings"], "metadata": result["metadata"]}, _debug_json(result))
    except PaperDiscoveryRateLimited as exc:
        delay = (f"Wait at least {exc.retry_after} seconds before trying again."
                 if exc.retry_after else "Wait a few minutes before trying again.")
        yield (_progress_html("limited"), _answer_html(
            "### Paper search is temporarily limited\n\n"
            f"Semantic Scholar received too many requests. {delay} "
            "You can explore the recorded demo while you wait.", "warning"),
            _empty_sources_html(), {"status": "rate_limited", "retry_after": exc.retry_after}, "")
    except Exception as exc:
        yield (_progress_html("failed"), _answer_html("### Check failed\n\n" + str(exc), "warning"),
               _empty_sources_html(), {"status": "failed", "error_type": type(exc).__name__}, "")


def clear_outputs():
    return _progress_html("ready"), _answer_html(READY_ANSWER), _empty_sources_html(), READY_TIMING, ""


def show_recorded_demo():
    # A condensed replay of the verified live run, with short excerpt selections.
    sources = [
        {"rank": "1", "title": "The vitamin C and the common cold: a narrative review",
         "year": 2025, "stance": "contradicts", "study_type": "narrative review",
         "section": "abstract", "why_selected": "The review addresses prevention in the general population.",
         "snippet": "Current evidence does not show that vitamin C reduces the incidence of the common cold in the general population.",
         "url": "https://doi.org/10.12775/jehs.2025.85.66914"},
        {"rank": "3", "title": "Vitamin C for the common cold and pneumonia",
         "year": 2025, "stance": "neutral", "study_type": "overview",
         "section": "abstract", "why_selected": "The overview reports different findings by population.",
         "snippet": "In controlled trials in the general human population, >1 g/day vitamin C did not prevent common colds. However, in 5 trials with participants undertaking heavy physical activity, vitamin C halved the incidence of colds.",
         "url": "https://doi.org/10.20452/pamw.16926"},
        {"rank": "4", "title": "Effectiveness of vitamin C in the management of viral infections in adults",
         "year": 2024, "stance": "insufficient", "study_type": "unknown",
         "section": "abstract", "why_selected": "The paper reports uncertainty about prevention.",
         "snippet": "The current evidence is insufficient to recommend the use of vitamin C for preventing or treating viral infections in the general population.",
         "url": "https://doi.org/10.12775/qs.2024.20.53985"},
    ]
    excluded = {
        "title": "Vitamin C as a Supplementary Therapy in Relieving Symptoms of the Common Cold",
        "reason": "The trials studied treatment after infection alongside antivirals, so the paper was excluded from the final prevention citations.",
        "doi": "10.1155/2020/8573742",
    }
    verdict = {"label": "Evidence weighs against the broad claim", "tone": "refuted"}
    confidence = {"label": "Moderate", "rationale": (
        "The assessment relies on cited abstracts. Full methods and any overlap "
        "in underlying studies could not be verified.")}
    findings = [
        "The cited evidence did not establish a prevention benefit in the general population. [1, 3, 4]",
        "The overview reported fewer colds in trials involving people undertaking heavy physical activity. [3]",
    ]
    answer = _answer_html(
        "Choose a citation to inspect the excerpt, or select **Check claim** to run this example live.",
        "done", verdict=verdict, confidence=confidence, evidence_summary=findings,
        source_ids={source["rank"] for source in sources}, limitations=[
            "The demo shows selected text from abstracts. The retrieved papers may not cover all relevant research."
        ],
    )
    excluded_html = (
        '<details class="r2r-output-list r2r-excluded"><summary>Why one paper was excluded</summary>'
        f'<p><a href="https://doi.org/{excluded["doi"]}" target="_blank" rel="noopener noreferrer">'
        f'{html.escape(excluded["title"])}</a></p><p>{html.escape(excluded["reason"])}</p></details>'
    )
    return (DEFAULT_CLAIM, _progress_html("demo"), answer,
            _sources_cards_html(sources) + excluded_html,
            {"status": "recorded_demo"}, _debug_json({
                "mode": "recorded_demo", "claim": DEFAULT_CLAIM, "verdict": verdict,
                "confidence": confidence, "findings": findings, "sources": sources,
                "excluded_paper": excluded,
            }))


JS = """
() => {
    document.addEventListener("click", (event) => {
        const link = event.target.closest?.(".r2r-citation");
        if (!link) return;
        const excerpt = document.getElementById(link.hash.slice(1));
        if (!excerpt) return;
        event.preventDefault();
        excerpt.open = true;
        excerpt.scrollIntoView({block: "center"});
        excerpt.querySelector("summary")?.focus({preventScroll: true});
    });
}
"""


CSS = """
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(249, 115, 22, 0.14), transparent 28rem),
        linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%) !important;
    color: #172033;
}
.r2r-shell {
    max-width: 1160px;
    margin: 0 auto;
    padding: 2rem 0 2.5rem;
}
.r2r-title h1 {
    margin: 0 0 0.25rem;
    color: #172033;
    font-size: clamp(2rem, 4.2vw, 3.6rem);
    line-height: 1.08;
    letter-spacing: 0;
}
.r2r-title p {
    max-width: 760px;
    margin: 0.4rem 0 1rem;
    color: #475569;
    font-size: 1rem;
}
.r2r-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.8rem 0 1.2rem;
}
.r2r-badge {
    border: 1px solid #cbd5e1;
    background: rgba(255, 255, 255, 0.72);
    border-radius: 999px;
    color: #334155;
    font-size: 0.84rem;
    padding: 0.35rem 0.7rem;
}
.r2r-panel {
    border: 1px solid #d8dee8;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.8);
    box-shadow: 0 16px 42px rgba(15, 23, 42, 0.08);
}
.r2r-progress {
    display: grid;
    gap: 0.65rem;
}
.r2r-step {
    align-items: center;
    border: 1px solid #d8dee8;
    border-radius: 8px;
    display: grid;
    gap: 0.75rem;
    grid-template-columns: auto 1fr;
    min-height: 62px;
    padding: 0.8rem;
    background: #ffffff;
}
.r2r-step strong {
    display: block;
    color: #172033;
    font-size: 0.92rem;
}
.r2r-step small {
    color: #64748b;
}
.r2r-step-dot {
    width: 0.8rem;
    height: 0.8rem;
    border-radius: 50%;
    background: #cbd5e1;
}
.r2r-step-done .r2r-step-dot {
    background: #16a34a;
}
.r2r-step-active {
    border-color: #fb923c;
    background: #fff7ed;
}
.r2r-step-active .r2r-step-dot {
    background: #f97316;
    box-shadow: 0 0 0 6px rgba(249, 115, 22, 0.18);
}
.r2r-step-waiting {
    color: #64748b;
}
.r2r-answer-card {
    min-height: 236px;
    border: 1px solid #d8dee8;
    border-radius: 8px;
    background: #ffffff;
    padding: 1.05rem 1.1rem;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
}
.r2r-answer-card h3 {
    margin: 0 0 0.75rem;
    color: #172033;
    font-size: 1.15rem;
}
.r2r-answer-card p {
    margin: 0 0 0.8rem;
    color: #334155;
    font-size: 0.98rem;
    line-height: 1.55;
}
.r2r-answer-card p:last-child {
    margin-bottom: 0;
}
.r2r-answer-card strong {
    color: #0f172a;
}
.r2r-answer-card code {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    color: #0f172a;
    padding: 0.08rem 0.28rem;
}
.r2r-verdict {
    border-radius: 8px;
    display: grid;
    gap: 0.2rem;
    margin: 0 0 0.9rem;
    padding: 0.75rem 0.85rem;
}
.r2r-verdict span {
    color: #0f172a;
    font-size: 1rem;
    font-weight: 800;
}
.r2r-verdict small {
    color: #475569;
    font-size: 0.86rem;
    line-height: 1.35;
}
.r2r-verdict-caution {
    background: #fff7ed;
    border: 1px solid #fed7aa;
}
.r2r-confidence {
    align-items: start;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    display: grid;
    gap: 0.2rem;
    margin: 0 0 0.85rem;
    padding: 0.65rem 0.8rem;
}
.r2r-confidence strong {
    color: #172033;
}
.r2r-confidence span {
    color: #475569;
    font-size: 0.88rem;
}
.r2r-output-list {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    margin: 0 0 0.85rem;
    padding: 0.75rem 0.85rem;
}
.r2r-output-list strong {
    color: #0f172a !important;
    display: block;
    font-size: 0.98rem;
    margin-bottom: 0.4rem;
    opacity: 1 !important;
}
.r2r-output-list ul {
    color: #172033 !important;
    margin: 0;
    padding-left: 1.1rem;
    opacity: 1 !important;
}
.r2r-output-list li {
    color: #172033 !important;
    font-size: 0.92rem;
    line-height: 1.45;
    margin: 0.25rem 0;
    opacity: 1 !important;
}
.r2r-output-list li::marker {
    color: #ea580c !important;
}
.r2r-limitations {
    background: #fff7ed;
    border-color: #fdba74;
}
.r2r-limitations li,
.r2r-limitations ul {
    color: #3f2a12 !important;
}
.r2r-limitations strong {
    color: #7c2d12 !important;
}
.r2r-verdict-supported {
    background: #ecfdf5;
    border: 1px solid #bbf7d0;
}
.r2r-verdict-refuted {
    background: #fef2f2;
    border: 1px solid #fecaca;
}
.r2r-verdict-mixed {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
}
.r2r-answer-ready {
    background:
        linear-gradient(135deg, rgba(255, 247, 237, 0.82), rgba(255, 255, 255, 0.96)),
        #ffffff;
}
.r2r-answer-working {
    border-color: #fb923c;
    background: #fff7ed;
}
.r2r-answer-warning {
    border-color: #f97316;
    background: #fff7ed;
}
.r2r-section-title {
    color: #334155;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0;
    margin: 0 0 0.5rem;
}
.r2r-note {
    border-left: 4px solid #f97316;
    color: #475569;
    background: #fff7ed;
    padding: 0.75rem 0.9rem;
    border-radius: 6px;
    margin-bottom: 1rem;
}
.r2r-note strong {
    color: #172033;
}
.r2r-empty-evidence {
    align-items: center;
    background: rgba(255, 255, 255, 0.78);
    border: 1px dashed #cbd5e1;
    border-radius: 8px;
    color: #64748b;
    display: flex;
    gap: 0.55rem;
    min-height: 64px;
    padding: 1rem;
}
.r2r-empty-evidence strong {
    color: #334155;
}
.r2r-sources-grid {
    display: grid;
    gap: 0.75rem;
    grid-template-columns: repeat(2, minmax(0, 1fr));
}
.r2r-source-card {
    background: #ffffff;
    border: 1px solid #d8dee8;
    border-radius: 8px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    min-height: 150px;
    padding: 0.95rem;
}
.r2r-source-topline {
    align-items: center;
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.7rem;
}
.r2r-source-rank,
.r2r-source-score {
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.22rem 0.5rem;
}
.r2r-source-rank {
    background: #172033;
    color: #ffffff;
}
.r2r-source-score {
    background: #ecfdf5;
    color: #047857;
}
.r2r-source-card h3 {
    color: #172033;
    font-size: 0.98rem;
    line-height: 1.3;
    margin: 0 0 0.55rem;
}
.r2r-source-card h3 span {
    color: #64748b;
    font-weight: 600;
}
.r2r-source-card p {
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.45;
    margin: 0 0 0.75rem;
}
.r2r-source-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin: 0 0 0.65rem;
}
.r2r-source-meta span {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 999px;
    color: #334155;
    font-size: 0.76rem;
    font-weight: 700;
    padding: 0.18rem 0.48rem;
}
.r2r-evidence-excerpt {
    margin: 0 0 0.85rem;
}
.r2r-evidence-excerpt summary {
    color: #334155;
    cursor: pointer;
    font-size: 0.88rem;
    font-weight: 600;
    padding: 0.25rem 0;
}
.r2r-evidence-excerpt summary:focus-visible {
    outline: 2px solid #ea580c;
    outline-offset: 3px;
    border-radius: 3px;
}
.r2r-evidence-excerpt[open] summary {
    margin-bottom: 0.6rem;
}
.r2r-source-card blockquote {
    border-left: 3px solid #fed7aa;
    color: #475569;
    font-size: 0.84rem;
    line-height: 1.4;
    margin: 0 0 0.75rem;
    padding-left: 0.65rem;
}
.r2r-source-card a {
    color: #ea580c;
    font-weight: 700;
    text-decoration: none;
}
@media (max-width: 820px) {
    .r2r-sources-grid {
        grid-template-columns: 1fr;
    }
}
button.primary {
    min-height: 46px !important;
}
footer {
    display: none !important;
}
.gradio-container {
    background: #f3f6fa !important;
}
.r2r-shell { max-width: 1120px; gap: 1.15rem !important; }
.r2r-title {
    background: linear-gradient(115deg, #101b31, #193d4b);
    border-radius: 18px;
    padding: clamp(1.5rem, 5vw, 3.2rem);
    margin-bottom: .35rem;
}
.r2r-title h1 { color: #fff; letter-spacing: -.045em; }
.r2r-title p { color: #d0e3e9; margin-bottom: 0; }
.r2r-title .r2r-kicker {
    color: #8ee1c9; font-size: .8rem; letter-spacing: .11em;
    text-transform: uppercase; font-weight: 700; margin-bottom: .7rem;
}
.r2r-panel { border-radius: 14px; padding: 1rem !important; background: #fff; }
.r2r-section-title { font-size: 1rem; color: #172033; }
.r2r-progress { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.r2r-step { min-height: 66px; }
.r2r-step-active, .r2r-answer-working { background: #ecfdf5; border-color: #6ee7b7; }
.r2r-step-active .r2r-step-dot { background: #047857; box-shadow: 0 0 0 4px #a7f3d0; }
.r2r-answer-card { min-height: 124px; border-radius: 14px; }
.r2r-answer-ready { background: #fff; }
.r2r-verdict span { font-size: 1.35rem; }
.r2r-source-card { border-radius: 12px; }
.r2r-source-card a { color: #047857; }
.r2r-title { padding: clamp(1.25rem, 4vw, 2rem); }
.r2r-title h1 { font-size: clamp(2rem, 5vw, 3rem); }
.r2r-shell textarea { background: #f8fafc !important; color: #172033 !important; }
.r2r-example { color: #475569 !important; font-size: .9rem; }
.r2r-status {
    padding: .65rem 1rem; border-radius: 8px; background: #e5f5ee;
    color: #17483d; font-weight: 650;
}
.r2r-citation {
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 1.5rem; min-height: 1.5rem; border-radius: 5px;
    background: #e5f5ee; color: #075d48 !important; font-weight: 700;
}
.r2r-citation:focus-visible, .r2r-excluded summary:focus-visible {
    outline: 2px solid #047857; outline-offset: 3px;
}
.r2r-excluded { margin-top: .9rem; }
.r2r-excluded summary, .r2r-limitations summary { font-weight: 700; cursor: pointer; }
.r2r-excluded p { margin: .5rem 0; }
.r2r-verdict .r2r-takeaway { margin-top: .45rem; }
.r2r-limitations strong { margin-top: .85rem; }
@media (max-width: 640px) {
    .r2r-progress { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .r2r-empty-evidence { align-items: start; flex-direction: column; }
}
"""


with gr.Blocks(
    title="Rumor2Reality",
    css=CSS,
    js=JS,
    theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate").set(
        body_background_fill_dark="#f3f6fa",
        body_text_color_dark="#172033",
        block_background_fill_dark="#ffffff",
        block_label_background_fill_dark="#e5f5ee",
        block_label_text_color_dark="#17483d",
        block_title_text_color_dark="#172033",
        input_background_fill_dark="#f8fafc",
        input_border_color_dark="#cbd5e1",
        button_secondary_background_fill_dark="#e2e8f0",
        button_secondary_text_color_dark="#172033",
    ),
) as demo:
    with gr.Column(elem_classes=["r2r-shell"]):
        gr.Markdown(
            """
            <div class="r2r-title">
              <p class="r2r-kicker">Research claim checker</p>
              <h1>Rumor2Reality</h1>
              <p>Check a claim against research papers and inspect the excerpts behind the assessment.</p>
            </div>
            """
        )

        with gr.Column(elem_classes=["r2r-panel"]):
            claim_input = gr.Textbox(label="Claim to check", lines=3,
                                     max_lines=6, placeholder="Enter a claim about a research topic")
            gr.Markdown('<p class="r2r-example">Example: Vitamin C prevents the common cold.</p>')
            with gr.Row():
                run_button = gr.Button("Check claim", variant="primary", scale=3)
                demo_button = gr.Button("Explore the demo", scale=2)
                clear_button = gr.Button("Clear results", scale=1)
            with gr.Accordion("Search and model settings", open=False):
                model_name = gr.Textbox(label="Ollama model", value=DEFAULT_MODEL)
                evidence_source = gr.Dropdown(label="Evidence source",
                    choices=["Semantic Scholar", "Existing vector store"], value="Semantic Scholar")
                vector_store_path = gr.Textbox(label="Existing vector store directory",
                    placeholder="Absolute path to an ingested PDF or paper index")
                top_k = gr.Slider(
                    label="Evidence chunks to retrieve",
                    minimum=1,
                    maximum=20,
                    value=4,
                    step=1,
                )
        progress = gr.HTML(value=_progress_html("ready"))
        gr.Markdown('<h2 class="r2r-section-title">Assessment</h2>')
        answer = gr.HTML(value=_answer_html(READY_ANSWER))

        gr.Markdown('<h2 class="r2r-section-title">Cited evidence</h2>')
        sources = gr.HTML(value=_empty_sources_html())

        with gr.Accordion("Full pipeline result", open=False):
            debug = gr.Code(label="Pipeline output", language="json")

        with gr.Accordion("Timing and metadata", open=False):
            timing = gr.JSON(label="Run details", value=READY_TIMING)

        with gr.Accordion("Example challenge prompts", open=False):
            gr.Examples(
                examples=[
                    [DEFAULT_CLAIM, 4, DEFAULT_MODEL],
                    ["A thread claims that AI-generated misinformation spreads faster than human-written rumors.", 5, DEFAULT_MODEL],
                    ["People say retrieval augmented generation always prevents hallucinations.", 3, DEFAULT_MODEL],
                ],
                inputs=[claim_input, top_k, model_name],
            )

    run_event = run_button.click(
        analyse_claim,
        inputs=[claim_input, top_k, model_name, evidence_source, vector_store_path],
        outputs=[progress, answer, sources, timing, debug],
        show_progress_on=[progress],
    )
    submit_event = claim_input.submit(
        analyse_claim,
        inputs=[claim_input, top_k, model_name, evidence_source, vector_store_path],
        outputs=[progress, answer, sources, timing, debug],
        show_progress_on=[progress],
    )
    clear_button.click(
        clear_outputs,
        outputs=[progress, answer, sources, timing, debug],
        cancels=[run_event, submit_event],
    )
    demo_button.click(
        show_recorded_demo,
        outputs=[claim_input, progress, answer, sources, timing, debug],
        cancels=[run_event, submit_event],
        queue=False,
    )


if __name__ == "__main__":
    demo.queue().launch()
