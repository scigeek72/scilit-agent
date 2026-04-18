"""
tools/llm_tools.py — LLM utility functions for all agents.

Every function accepts an optional `llm` parameter so it can be tested
without a real LLM (pass a MagicMock).  When llm=None, get_llm() is called.

Functions used by Ingest Agent (Phase 6):
  summarize_paper        — 2–4 sentence plain-language summary
  extract_tags           — topic tags from title + abstract
  fill_paper_wiki_page   — complete the paper wiki page stubs
  update_concept_page    — merge new paper into an existing concept page
  flag_contradictions    — detect if paper contradicts existing claims

Functions used by Query Agent (Phase 7):
  decompose_query        — split complex query into sub-queries
  generate_hyde_doc      — hypothetical document for HyDE retrieval
  synthesize_answer      — generate grounded answer from context
  self_critique          — verify answer is grounded in sources
  is_worth_filing        — decide if answer merits wiki write-back
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ingest Agent tools
# ---------------------------------------------------------------------------

def summarize_paper(parsed_paper: dict, llm=None) -> str:
    """
    Generate a 2–4 sentence plain-language summary of a paper.

    Used by the Ingest Agent to populate wiki/papers/{id}.md ## Summary.
    Returns a best-effort summary even on LLM failure (falls back to
    truncated abstract).
    """
    title    = parsed_paper.get("title", "")
    abstract = parsed_paper.get("abstract", "")
    sections = parsed_paper.get("sections", [])

    # Build a compact context (title + abstract + first section text)
    intro_text = ""
    for s in sections[:2]:
        if s.get("text"):
            intro_text = s["text"][:500]
            break

    context = f"Title: {title}\n\nAbstract: {abstract[:800]}"
    if intro_text:
        context += f"\n\nIntroduction excerpt: {intro_text}"

    prompt = (
        "Write a 2–4 sentence plain-language summary of this scientific paper. "
        "Focus on what was done, the key finding, and why it matters. "
        "Do not use jargon. Do not repeat the title.\n\n"
        f"{context}"
    )

    response = _call_llm(prompt, llm, max_tokens=200)
    if response:
        return response.strip()

    # Fallback: truncated abstract
    return abstract[:400] + ("…" if len(abstract) > 400 else "")


def extract_tags(parsed_paper: dict, llm=None) -> list[str]:
    """
    Extract 3–6 topic tags from a paper's title and abstract.

    Returns lowercase hyphenated tags like ['transformers', 'self-attention'].
    Used to populate the `tags:` frontmatter in wiki/papers/{id}.md.
    """
    title    = parsed_paper.get("title", "")
    abstract = parsed_paper.get("abstract", "")

    prompt = (
        "List 3–6 lowercase topic tags for this scientific paper. "
        "Use hyphens for multi-word tags (e.g. 'self-attention'). "
        "Return ONLY a comma-separated list of tags, nothing else.\n\n"
        f"Title: {title}\nAbstract: {abstract[:600]}"
    )

    response = _call_llm(prompt, llm, max_tokens=80)
    if response:
        raw = response.strip().lower()
        tags = [t.strip().strip('"').strip("'") for t in raw.split(",")]
        return [t for t in tags if t and len(t) <= 50][:6]

    # Fallback: extract noun phrases from title
    return _fallback_tags(title)


def fill_paper_wiki_page(page_stub: str, parsed_paper: dict, llm=None) -> str:
    """
    Fill in the stub sections of a paper wiki page.

    Takes the skeleton produced by make_paper_page() and uses the LLM to
    write the Key Contributions, Methods, Results & Claims, Limitations,
    and Open Questions Raised sections.

    Returns the completed page content.
    """
    title    = parsed_paper.get("title", "")
    abstract = parsed_paper.get("abstract", "")
    sections = parsed_paper.get("sections", [])
    refs     = parsed_paper.get("references", [])

    # Build a compact paper digest for the LLM
    section_digest = "\n".join(
        f"[{s.get('heading', '')}] {s.get('text', '')[:300]}"
        for s in sections[:6]
    )
    ref_titles = "; ".join(r.get("title", "")[:60] for r in refs[:5])

    prompt = f"""You are filling in a scientific wiki page for the paper titled: "{title}"

Abstract: {abstract[:600]}

Section excerpts:
{section_digest}

Key references cited: {ref_titles}

Fill in the following sections of the wiki page. Be concise and factual.
Use [[wiki-links]] for concepts that deserve their own page.
Use confidence labels (high/medium/low) on claims in Results & Claims.

Sections to fill (return ONLY these sections, in order):

## Key Contributions
(2–4 bullet points on what this paper specifically contributed)

## Methods
(1–3 sentences describing the approach; reference [[concepts/]] inline)

## Results & Claims
(2–4 bullet points with confidence labels)

## Limitations
(1–2 sentences on stated or obvious limitations)

## Related Papers
- Builds on:
- Contradicts:
- Extended by:

## Key Concepts
(comma-separated [[wiki-links]] to concept pages)

## Open Questions Raised
(1–3 bullet points on unresolved questions this paper raises)"""

    response = _call_llm(prompt, llm, max_tokens=600)
    if not response:
        return page_stub  # return stub unchanged if LLM fails

    # Replace the stub placeholders with LLM output
    filled = _replace_stubs(page_stub, response)
    return filled


def update_concept_page(
    existing_page: str,
    parsed_paper: dict,
    concept_name: str,
    llm=None,
) -> str:
    """
    Update an existing concept wiki page to incorporate a new paper.

    If the concept page is empty/stub, generates it from scratch.
    Adds a row to the Key Papers table and updates Cross-domain Notes
    if the paper is from a different domain than existing papers.

    Returns the updated page content.
    """
    paper_id = parsed_paper.get("paper_id", "")
    title    = parsed_paper.get("title", "")
    abstract = parsed_paper.get("abstract", "")
    source   = parsed_paper.get("source", "")
    year     = parsed_paper.get("year", 0)

    wiki_filename = paper_id.replace(":", "-").replace("/", "-")

    prompt = f"""You are updating a scientific wiki page for the concept: "{concept_name}"

Existing page:
{existing_page[:1200]}

New paper to incorporate:
- Paper ID: {paper_id}
- Title: {title}
- Source: {source} ({year})
- Abstract excerpt: {abstract[:400]}

Update the wiki page to:
1. Add a row for [[papers/{wiki_filename}]] to the Key Papers table
   (format: | [[papers/{wiki_filename}]] | one-line contribution | {year} | {source} |)
2. Update or add to Cross-domain Notes if this paper is from a new domain
3. Keep all existing content intact
4. Do not change the frontmatter (--- block)

Return the COMPLETE updated page."""

    response = _call_llm(prompt, llm, max_tokens=800)
    if response:
        return response.strip()
    return existing_page  # return unchanged on failure


def flag_contradictions(
    existing_claims: list[str],
    new_paper: dict,
    llm=None,
) -> list[str]:
    """
    Detect if any claims in the new paper contradict existing wiki claims.

    Returns a list of contradiction descriptions (empty list = no contradictions).
    Used by the Ingest Agent to decide whether to update wiki/debates/.
    """
    if not existing_claims or not new_paper.get("sections"):
        return []

    new_text = " ".join(
        s.get("text", "")[:200] for s in new_paper.get("sections", [])[:3]
    )
    claims_text = "\n".join(f"- {c}" for c in existing_claims[:10])

    prompt = (
        "You are checking for contradictions between established claims and a new paper.\n\n"
        f"Established claims from wiki:\n{claims_text}\n\n"
        f"New paper excerpt: {new_text[:600]}\n\n"
        "List any direct contradictions (claims the new paper disputes). "
        "If none, reply with exactly: NONE\n"
        "Format: one contradiction per line."
    )

    response = _call_llm(prompt, llm, max_tokens=200)
    if not response or "NONE" in response.upper():
        return []
    lines = [l.strip().lstrip("-").strip() for l in response.splitlines() if l.strip()]
    return [l for l in lines if l and "NONE" not in l.upper()]


# ---------------------------------------------------------------------------
# Query Agent tools (Phase 7)
# ---------------------------------------------------------------------------

def decompose_query(query: str, llm=None) -> list[str]:
    """
    Decompose a complex query into 2–4 simpler sub-queries.

    Use when query contains: compare, difference, vs, survey, review, explain.
    Returns list of sub-query strings. Falls back to [query] if LLM fails.
    """
    prompt = (
        "Decompose this research query into 2–4 simpler, focused sub-queries "
        "that together cover the original question. "
        "Return ONLY a numbered list, one sub-query per line.\n\n"
        f"Query: {query}"
    )
    response = _call_llm(prompt, llm, max_tokens=150)
    if not response:
        return [query]

    lines = re.findall(r"^\d+\.\s*(.+)$", response, re.MULTILINE)
    return lines if lines else [query]


def synthesize_answer(
    query: str,
    wiki_context: str,
    chunks: list[dict],
    llm=None,
) -> str:
    """
    Generate a grounded answer to a query using wiki context + retrieved chunks.

    Returns the answer string. Falls back to a best-effort answer if LLM fails.
    """
    chunk_text = "\n\n".join(
        f"[{c.get('paper_id', '?')} — {c.get('section_heading', '')}]\n{c.get('text', '')[:300]}"
        for c in chunks[:5]
    )

    prompt = (
        f"Answer the following research question based ONLY on the provided sources.\n\n"
        f"Question: {query}\n\n"
        f"Wiki knowledge base:\n{wiki_context[:1500]}\n\n"
        f"Retrieved paper chunks:\n{chunk_text}\n\n"
        "Write a clear, concise answer (3–6 sentences). "
        "Cite specific papers where relevant using (Author, Year) style. "
        "If the sources do not contain enough information, say so explicitly."
    )
    response = _call_llm(prompt, llm, max_tokens=400)
    return response.strip() if response else "Insufficient information in knowledge base to answer this query."


def self_critique(
    query: str,
    answer: str,
    sources: list[dict],
    llm=None,
) -> dict:
    """
    Evaluate whether the answer is grounded in the provided sources.

    Returns {is_grounded: bool, confidence: float, issues: list[str]}.
    Used by the Query Agent before deciding whether to file back.
    """
    source_titles = "; ".join(
        c.get("title", c.get("paper_id", "?"))[:60] for c in sources[:5]
    )

    prompt = (
        f"Evaluate whether this answer is grounded in the provided sources.\n\n"
        f"Question: {query}\n"
        f"Answer: {answer}\n"
        f"Sources available: {source_titles}\n\n"
        "Reply in this exact format:\n"
        "GROUNDED: yes|no\n"
        "CONFIDENCE: 0.0-1.0\n"
        "ISSUES: issue1; issue2 (or NONE)"
    )
    response = _call_llm(prompt, llm, max_tokens=100)
    return _parse_critique(response)


def is_worth_filing(query: str, answer: str, llm=None) -> bool:
    """
    Decide whether this query-answer pair should be written back to the wiki.

    Returns True for novel synthesis, comparisons, cross-paper analyses.
    Returns False for simple factual lookups or trivial answers.
    """
    prompt = (
        "Should this query-answer pair be saved to the wiki knowledge base?\n\n"
        f"Query: {query}\n"
        f"Answer: {answer[:300]}\n\n"
        "Save to wiki if: novel synthesis, comparison of multiple papers, "
        "cross-domain insight, or complex analysis.\n"
        "Do NOT save if: trivial factual lookup, single-paper summary, very short answer.\n"
        "Reply with ONLY: YES or NO"
    )
    response = _call_llm(prompt, llm, max_tokens=5)
    return bool(response and "YES" in response.upper())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, llm, max_tokens: int = 300) -> str | None:
    """
    Call the LLM with a prompt. Returns the text response or None on failure.
    If llm is None, loads it from llm_provider.get_llm().
    """
    if llm is None:
        try:
            from llm_provider import get_llm
            llm = get_llm(temperature=0.1)
        except Exception as exc:
            logger.warning("Could not load LLM: %s", exc)
            return None

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke(
            [HumanMessage(content=prompt)],
            config={"max_tokens": max_tokens},
        )
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return None


def _replace_stubs(page: str, llm_sections: str) -> str:
    """
    Replace '_To be filled by Ingest Agent_' stubs in the page with LLM output.
    Merges the LLM-generated sections into the existing page structure.
    """
    # Parse LLM output into sections
    section_re = re.compile(r"^## (.+)$", re.MULTILINE)
    parts = section_re.split(llm_sections)

    # parts is: [pre, heading1, body1, heading2, body2, ...]
    replacements: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        replacements[heading] = body

    # Apply replacements to the page
    def _replace_section(m: re.Match) -> str:
        heading = m.group(1).strip()
        if heading in replacements:
            return f"## {heading}\n{replacements[heading]}"
        return m.group(0)

    result = re.sub(
        r"^## (.+)\n_To be filled by Ingest Agent_",
        _replace_section,
        page,
        flags=re.MULTILINE,
    )
    return result


def _parse_critique(response: str | None) -> dict:
    """Parse self_critique LLM response into structured dict."""
    if not response:
        return {"is_grounded": True, "confidence": 0.5, "issues": []}

    is_grounded = "yes" in response.lower().split("grounded:")[-1][:10] if "grounded:" in response.lower() else True

    conf_m = re.search(r"confidence:\s*([\d.]+)", response, re.IGNORECASE)
    confidence = float(conf_m.group(1)) if conf_m else 0.5
    confidence = max(0.0, min(1.0, confidence))

    issues_m = re.search(r"issues:\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
    issues_raw = issues_m.group(1).strip() if issues_m else "NONE"
    issues = [] if "NONE" in issues_raw.upper() else [i.strip() for i in issues_raw.split(";") if i.strip()]

    return {"is_grounded": is_grounded, "confidence": confidence, "issues": issues}


def _fallback_tags(title: str) -> list[str]:
    """Extract simple tags from a title when LLM is unavailable."""
    stop = {"a", "an", "the", "of", "in", "for", "and", "or", "with", "via",
            "using", "based", "on", "to", "is", "are", "by"}
    words = re.findall(r"[a-z][a-z-]+", title.lower())
    return [w for w in words if w not in stop and len(w) > 3][:5]
