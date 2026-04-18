"""
retrieval/hyde.py — Hypothetical Document Embedding (HyDE).

Instead of embedding the raw query, we ask the LLM to write a short
hypothetical paper excerpt that would answer the query, then embed that.
The idea (Gao et al., 2022) is that a document-like vector is closer to
real relevant passages in embedding space than a short query vector.

Usage:
    from retrieval.hyde import generate_hyde_document
    excerpt = generate_hyde_document("What is self-attention?")
    # use excerpt as query text for embedding
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_HYDE_PROMPT = """\
Write a short (3-5 sentence) scientific paper excerpt that directly answers the following research question. Write in the style of a methods or results section. Do not add a title or references.

Question: {query}

Excerpt:"""

_MAX_TOKENS = 200


def generate_hyde_document(query: str, llm=None) -> str:
    """
    Generate a hypothetical document excerpt for HyDE retrieval.

    llm: optional LangChain-compatible chat model.  If None, uses get_llm().
    Returns the generated text, or the original query on failure.
    """
    if llm is None:
        try:
            from llm_provider import get_llm
            llm = get_llm(temperature=0.3)
        except Exception as exc:
            logger.warning("Could not load LLM for HyDE: %s — using raw query", exc)
            return query

    prompt = _HYDE_PROMPT.format(query=query)

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke(
            [HumanMessage(content=prompt)],
            config={"max_tokens": _MAX_TOKENS},
        )
        text = response.content.strip()
        if not text:
            return query
        logger.debug("HyDE generated %d chars for query: %s", len(text), query[:60])
        return text
    except Exception as exc:
        logger.warning("HyDE generation failed: %s — falling back to raw query", exc)
        return query
