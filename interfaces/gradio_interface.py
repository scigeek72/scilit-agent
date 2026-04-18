"""
interfaces/gradio_interface.py — Gradio web UI for scilit-agent.

Tabs:
  Query   — ask questions, see answers with citations
  Ingest  — trigger paper search and ingest
  Lint    — run wiki health check, view report
  Status  — service and wiki stats

Launch:
  python -m interfaces.gradio_interface
  # or from Python:
  from interfaces.gradio_interface import build_app
  build_app().launch()
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from agents.ingest_agent import run_ingest
from agents.query_agent import run_query
from agents.lint_agent import run_lint
from agents.frontier_agent import run_frontier
import services


# ---------------------------------------------------------------------------
# Tab: Query
# ---------------------------------------------------------------------------

def _do_query(question: str, year_filter: str, source_filter: str) -> tuple[str, str]:
    """Run the query agent. Returns (answer_markdown, sources_markdown)."""
    if not question.strip():
        return "Please enter a question.", ""

    year = int(year_filter) if year_filter.strip().isdigit() else None
    src  = source_filter.strip() or None

    result = run_query(question.strip(), year_filter=year, source_filter=src)

    answer     = result.get("answer", "No answer generated.")
    confidence = result.get("confidence", 0.0)
    cache_hit  = result.get("cache_hit", False)
    grounded   = result.get("is_grounded", True)
    filed      = result.get("filed_page_path")

    # Build answer block
    badges: list[str] = [f"**Confidence**: {confidence*100:.0f}%"]
    if cache_hit:
        badges.append("📦 *cache hit*")
    if not grounded:
        badges.append("⚠️ *low confidence — treat with caution*")
    if filed:
        badges.append(f"📝 *filed to* `{filed}`")

    answer_md = answer + "\n\n" + " · ".join(badges)

    # Build sources block
    sources = result.get("sources", [])
    seen:    set[str] = set()
    rows:    list[str] = []
    for s in sources:
        pid = s.get("paper_id") or s.get("page_path", "")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        if s.get("type") == "wiki":
            rows.append(f"- 📄 `{pid}`")
        else:
            title = s.get("title", pid)
            year  = s.get("year", "")
            rows.append(f"- **{title}**" + (f" ({year})" if year else ""))

    sources_md = "\n".join(rows) if rows else "*No sources cited.*"
    return answer_md, sources_md


# ---------------------------------------------------------------------------
# Tab: Ingest
# ---------------------------------------------------------------------------

def _do_ingest(query: str) -> str:
    """Run the ingest pipeline. Returns a status markdown string."""
    if not query.strip():
        return "Please enter a search query."

    result = run_ingest(query.strip())

    n_papers = len(result.get("papers_processed", []))
    n_chunks = result.get("chunks_indexed", 0)
    errors   = result.get("errors", [])

    lines = [
        f"✅ **Papers processed**: {n_papers}",
        f"✅ **Chunks indexed**: {n_chunks}",
    ]
    if errors:
        lines.append(f"\n⚠️ **Non-fatal errors** ({len(errors)}):")
        for e in errors[:5]:
            lines.append(f"- {e}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab: Lint
# ---------------------------------------------------------------------------

def _do_lint() -> str:
    """Run the lint agent. Returns a markdown summary."""
    result = run_lint()

    orphans  = result.get("orphans", [])
    debates  = result.get("contradictions", [])
    stale    = result.get("stale_claims", [])
    missing  = result.get("missing_concept_pages", [])
    gaps     = result.get("gaps", [])
    report   = result.get("report_path", "")

    def _list(items: list[str], max_n: int = 5) -> str:
        if not items:
            return "*None.*"
        shown = "\n".join(f"- {i}" for i in items[:max_n])
        extra = f"\n*…and {len(items)-max_n} more.*" if len(items) > max_n else ""
        return shown + extra

    md = f"""## Wiki Lint Report

| Check | Count |
|---|---|
| Orphaned pages | {len(orphans)} |
| Unresolved debates | {len(debates)} |
| Potentially stale claims | {len(stale)} |
| Missing concept pages | {len(missing)} |
| Gap suggestions | {len(gaps)} |

### Suggested Next Searches
{_list(gaps)}

### Orphaned Pages
{_list(orphans)}

### Missing Concept Pages
{_list(missing)}
"""
    if report:
        md += f"\n*Full report written to `{report}`*"
    return md


# ---------------------------------------------------------------------------
# Tab: Frontier
# ---------------------------------------------------------------------------

def _do_frontier(query: str) -> tuple[str, str]:
    """Run the Frontier Agent. Returns (summary_markdown, report_markdown)."""
    if not query.strip():
        return "Please enter a gap question.", ""

    result = run_frontier(query.strip())

    focus    = result.get("query_focus", "both")
    oqs      = result.get("open_questions", [])
    gaps     = result.get("method_domain_gaps", [])
    temporal = result.get("temporal_dropouts", [])
    cross    = result.get("cross_domain_opportunities", [])
    filed    = result.get("filed_page_path")
    report   = result.get("report", "")

    summary_lines = [
        f"**Focus**: {focus}",
        f"**Open question clusters**: {len(oqs)}",
        f"**Method-domain gaps**: {len(gaps)}",
        f"**Temporal dropouts**: {len(temporal)}",
        f"**Cross-domain opportunities**: {len(cross)}",
    ]
    if filed:
        summary_lines.append(f"\n📝 *Report filed to* `{filed}`")

    return "\n\n".join(summary_lines), report or "_No report generated._"


# ---------------------------------------------------------------------------
# Tab: Status
# ---------------------------------------------------------------------------

def _do_status() -> str:
    """Return a markdown string showing service and wiki status."""
    lines: list[str] = ["## Service Status\n"]

    # Grobid
    g = services.grobid_status()
    grobid_ok = "✅" if g.get("running") else "❌"
    lines.append(f"- **Grobid**    {grobid_ok}  {g.get('url', '')}")

    # marker
    m = services.marker_status()
    marker_ok = "✅" if m.get("installed") else "❌"
    lines.append(f"- **marker**    {marker_ok}")

    # PyMuPDF (fallback)
    try:
        import fitz  # noqa: F401
        lines.append("- **PyMuPDF**   ✅  (fallback parser)")
    except ImportError:
        lines.append("- **PyMuPDF**   ❌")

    # Wiki stats
    wiki_dir = Config.wiki_dir()
    lines.append(f"\n## Wiki — {Config.TOPIC_NAME}\n")
    if wiki_dir.exists():
        for subdir in ("papers", "concepts", "methods", "debates"):
            d = wiki_dir / subdir
            count = len(list(d.glob("*.md"))) if d.exists() else 0
            lines.append(f"- **{subdir.capitalize()}**: {count}")
    else:
        lines.append("*Wiki not initialised — run Ingest first.*")

    # Config summary
    lines.append(f"\n## Config\n")
    lines.append(f"- **Topic**: {Config.TOPIC_NAME}")
    lines.append(f"- **LLM**: {Config.LLM_PROVIDER} / "
                 f"{Config.OPENAI_CHAT_MODEL if Config.LLM_PROVIDER == 'openai' else Config.ANTHROPIC_CHAT_MODEL}")
    lines.append(f"- **Embeddings**: {Config.EMBEDDING_PROVIDER}")
    lines.append(f"- **Active sources**: {', '.join(Config.ACTIVE_SOURCES)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app():
    """Build and return the Gradio Blocks app."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("gradio is not installed. Run: pip install gradio")

    with gr.Blocks(title="scilit-agent", theme=gr.themes.Soft()) as app:
        gr.Markdown(f"# 🔬 scilit-agent\n*Source-agnostic scientific literature assistant · Topic: **{Config.TOPIC_NAME}***")

        with gr.Tabs():

            # ── Query tab ────────────────────────────────────────────────
            with gr.TabItem("🔍 Query"):
                with gr.Row():
                    q_input = gr.Textbox(
                        label="Question",
                        placeholder="What are the main challenges in CRISPR delivery?",
                        lines=2, scale=4,
                    )
                with gr.Row():
                    year_input   = gr.Textbox(label="Year filter (optional)", placeholder="2023", scale=1)
                    source_input = gr.Textbox(label="Source filter (optional)", placeholder="arxiv", scale=1)
                q_btn    = gr.Button("Ask", variant="primary")
                q_answer = gr.Markdown(label="Answer")
                q_sources = gr.Markdown(label="Sources")

                q_btn.click(
                    fn=_do_query,
                    inputs=[q_input, year_input, source_input],
                    outputs=[q_answer, q_sources],
                )

            # ── Ingest tab ────────────────────────────────────────────────
            with gr.TabItem("📥 Ingest"):
                i_input  = gr.Textbox(
                    label="Search query",
                    placeholder="attention mechanisms transformer models",
                    lines=2,
                )
                i_btn    = gr.Button("Ingest Papers", variant="primary")
                i_output = gr.Markdown(label="Result")

                i_btn.click(fn=_do_ingest, inputs=[i_input], outputs=[i_output])

            # ── Frontier tab ──────────────────────────────────────────────
            with gr.TabItem("🔭 Frontier"):
                f_input  = gr.Textbox(
                    label="Gap question",
                    placeholder="What methodological gaps exist in this corpus?",
                    lines=2,
                )
                f_btn     = gr.Button("Analyse Gaps", variant="primary")
                f_summary = gr.Markdown(label="Summary")
                f_report  = gr.Markdown(label="Full Report")

                f_btn.click(fn=_do_frontier, inputs=[f_input], outputs=[f_summary, f_report])

            # ── Lint tab ──────────────────────────────────────────────────
            with gr.TabItem("🔧 Lint"):
                lint_btn    = gr.Button("Run Lint Check", variant="secondary")
                lint_output = gr.Markdown(label="Lint Report")

                lint_btn.click(fn=_do_lint, inputs=[], outputs=[lint_output])

            # ── Status tab ────────────────────────────────────────────────
            with gr.TabItem("📊 Status"):
                status_btn    = gr.Button("Refresh Status", variant="secondary")
                status_output = gr.Markdown()

                status_btn.click(fn=_do_status, inputs=[], outputs=[status_output])
                # Show status on load
                app.load(fn=_do_status, outputs=[status_output])

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
