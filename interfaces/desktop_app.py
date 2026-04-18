"""
interfaces/desktop_app.py — Native desktop UI for scilit-agent.

Tries PyQt6 first; falls back to Tkinter if PyQt6 is not installed.
Both UIs provide the same four panels: Query, Ingest, Lint, Status.

Launch:
  python -m interfaces.desktop_app
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from agents.ingest_agent import run_ingest
from agents.query_agent import run_query
from agents.lint_agent import run_lint
import services


# ---------------------------------------------------------------------------
# Shared backend calls (same logic as gradio_interface, no UI dependency)
# ---------------------------------------------------------------------------

def _run_query(question: str, year: int | None, source: str | None) -> tuple[str, str]:
    result  = run_query(question, year_filter=year, source_filter=source)
    answer  = result.get("answer", "No answer generated.")
    conf    = result.get("confidence", 0.0)
    filed   = result.get("filed_page_path")
    sources = result.get("sources", [])

    meta = f"\nConfidence: {conf*100:.0f}%"
    if result.get("cache_hit"):
        meta += " · cache hit"
    if not result.get("is_grounded", True):
        meta += " · ⚠ low confidence"
    if filed:
        meta += f"\nFiled: {filed}"

    seen: set[str] = set()
    src_lines: list[str] = []
    for s in sources:
        pid = s.get("paper_id") or s.get("page_path", "")
        if pid and pid not in seen:
            seen.add(pid)
            title = s.get("title", pid)
            year_ = s.get("year", "")
            src_lines.append(f"• {title}" + (f" ({year_})" if year_ else ""))

    return answer + meta, "\n".join(src_lines) or "No sources."


def _run_ingest(query: str) -> str:
    result   = run_ingest(query)
    n_papers = len(result.get("papers_processed", []))
    n_chunks = result.get("chunks_indexed", 0)
    errors   = result.get("errors", [])
    lines    = [f"Papers processed: {n_papers}", f"Chunks indexed:   {n_chunks}"]
    if errors:
        lines.append(f"Non-fatal errors: {len(errors)}")
        lines += [f"  • {e}" for e in errors[:5]]
    return "\n".join(lines)


def _run_lint() -> str:
    result   = run_lint()
    orphans  = result.get("orphans", [])
    debates  = result.get("contradictions", [])
    stale    = result.get("stale_claims", [])
    missing  = result.get("missing_concept_pages", [])
    gaps     = result.get("gaps", [])
    report   = result.get("report_path", "")
    lines = [
        f"Orphaned pages       : {len(orphans)}",
        f"Unresolved debates   : {len(debates)}",
        f"Potentially stale    : {len(stale)}",
        f"Missing concept pages: {len(missing)}",
        f"Gap suggestions      : {len(gaps)}",
        "",
    ]
    if gaps:
        lines.append("Suggested next searches:")
        lines += [f"  • {g}" for g in gaps[:5]]
    if report:
        lines.append(f"\nReport: {report}")
    return "\n".join(lines)


def _run_status() -> str:
    lines: list[str] = ["=== Service Status ==="]
    g = services.grobid_status()
    lines.append(f"Grobid : {'OK' if g.get('running') else 'offline'} ({g.get('url','')})")
    m = services.marker_status()
    lines.append(f"marker : {'installed' if m.get('installed') else 'not found'}")
    try:
        import fitz  # noqa
        lines.append("PyMuPDF: installed (fallback parser)")
    except ImportError:
        lines.append("PyMuPDF: not installed")

    wiki_dir = Config.wiki_dir()
    lines.append(f"\n=== Wiki — {Config.TOPIC_NAME} ===")
    if wiki_dir.exists():
        for subdir in ("papers", "concepts", "methods", "debates"):
            d = wiki_dir / subdir
            count = len(list(d.glob("*.md"))) if d.exists() else 0
            lines.append(f"  {subdir.capitalize():<10}: {count}")
    else:
        lines.append("  Not initialised.")

    lines.append(f"\n=== Config ===")
    lines.append(f"  Topic    : {Config.TOPIC_NAME}")
    lines.append(f"  LLM      : {Config.LLM_PROVIDER}")
    lines.append(f"  Embeddings: {Config.EMBEDDING_PROVIDER}")
    lines.append(f"  Sources  : {', '.join(Config.ACTIVE_SOURCES)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PyQt6 UI
# ---------------------------------------------------------------------------

def _launch_pyqt() -> None:
    from PyQt6.QtCore import QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
        QPlainTextEdit, QPushButton, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget,
    )

    class Worker(QThread):
        """Run a callable in a background thread; emit result when done."""
        done = pyqtSignal(str)

        def __init__(self, fn, *args):
            super().__init__()
            self._fn   = fn
            self._args = args

        def run(self):
            try:
                result = self._fn(*self._args)
            except Exception as exc:
                result = f"Error: {exc}"
            self.done.emit(str(result))

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"scilit-agent — {Config.TOPIC_NAME}")
            self.resize(900, 650)
            self._workers: list[Worker] = []

            tabs = QTabWidget()
            tabs.addTab(self._query_tab(), "🔍 Query")
            tabs.addTab(self._ingest_tab(), "📥 Ingest")
            tabs.addTab(self._lint_tab(), "🔧 Lint")
            tabs.addTab(self._status_tab(), "📊 Status")
            self.setCentralWidget(tabs)

        # ── Query tab ────────────────────────────────────────────────────
        def _query_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            self._q_input = QPlainTextEdit()
            self._q_input.setPlaceholderText("Enter your question…")
            self._q_input.setMaximumHeight(80)
            layout.addWidget(QLabel("Question:"))
            layout.addWidget(self._q_input)

            row = QHBoxLayout()
            self._q_year   = QLineEdit(); self._q_year.setPlaceholderText("Year filter (optional)")
            self._q_source = QLineEdit(); self._q_source.setPlaceholderText("Source filter (optional)")
            row.addWidget(self._q_year)
            row.addWidget(self._q_source)
            layout.addLayout(row)

            self._q_btn = QPushButton("Ask")
            self._q_btn.clicked.connect(self._on_query)
            layout.addWidget(self._q_btn)

            layout.addWidget(QLabel("Answer:"))
            self._q_answer = QTextEdit(); self._q_answer.setReadOnly(True)
            layout.addWidget(self._q_answer)

            layout.addWidget(QLabel("Sources:"))
            self._q_sources = QTextEdit(); self._q_sources.setReadOnly(True)
            self._q_sources.setMaximumHeight(120)
            layout.addWidget(self._q_sources)
            return w

        def _on_query(self):
            question = self._q_input.toPlainText().strip()
            if not question:
                return
            year_s = self._q_year.text().strip()
            year   = int(year_s) if year_s.isdigit() else None
            source = self._q_source.text().strip() or None
            self._q_btn.setEnabled(False)
            self._q_answer.setPlainText("Thinking…")

            # Pack tuple into delimited string since Worker.done is a str signal
            def _pack():
                ans, srcs = _run_query(question, year, source)
                return ans + "|||" + srcs

            w = Worker(_pack)
            w.done.connect(self._on_query_done)
            self._workers.append(w)
            w.start()

        def _on_query_done(self, result: str):
            parts = result.split("|||", 1)
            self._q_answer.setPlainText(parts[0])
            self._q_sources.setPlainText(parts[1] if len(parts) > 1 else "")
            self._q_btn.setEnabled(True)

        # ── Ingest tab ───────────────────────────────────────────────────
        def _ingest_tab(self) -> QWidget:
            w = QWidget(); layout = QVBoxLayout(w)
            layout.addWidget(QLabel("Search query:"))
            self._i_input = QLineEdit()
            self._i_input.setPlaceholderText("attention mechanisms transformers")
            layout.addWidget(self._i_input)
            self._i_btn = QPushButton("Ingest Papers")
            self._i_btn.clicked.connect(self._on_ingest)
            layout.addWidget(self._i_btn)
            layout.addWidget(QLabel("Result:"))
            self._i_output = QTextEdit(); self._i_output.setReadOnly(True)
            layout.addWidget(self._i_output)
            return w

        def _on_ingest(self):
            query = self._i_input.text().strip()
            if not query:
                return
            self._i_btn.setEnabled(False)
            self._i_output.setPlainText("Ingesting…")
            w = Worker(_run_ingest, query)
            w.done.connect(lambda r: (self._i_output.setPlainText(r), self._i_btn.setEnabled(True)))
            self._workers.append(w); w.start()

        # ── Lint tab ─────────────────────────────────────────────────────
        def _lint_tab(self) -> QWidget:
            w = QWidget(); layout = QVBoxLayout(w)
            self._lint_btn = QPushButton("Run Lint Check")
            self._lint_btn.clicked.connect(self._on_lint)
            layout.addWidget(self._lint_btn)
            self._lint_output = QTextEdit(); self._lint_output.setReadOnly(True)
            layout.addWidget(self._lint_output)
            return w

        def _on_lint(self):
            self._lint_btn.setEnabled(False)
            self._lint_output.setPlainText("Running lint…")
            w = Worker(_run_lint)
            w.done.connect(lambda r: (self._lint_output.setPlainText(r), self._lint_btn.setEnabled(True)))
            self._workers.append(w); w.start()

        # ── Status tab ───────────────────────────────────────────────────
        def _status_tab(self) -> QWidget:
            w = QWidget(); layout = QVBoxLayout(w)
            self._status_btn = QPushButton("Refresh")
            self._status_btn.clicked.connect(self._on_status)
            layout.addWidget(self._status_btn)
            self._status_output = QTextEdit(); self._status_output.setReadOnly(True)
            layout.addWidget(self._status_output)
            return w

        def _on_status(self):
            self._status_btn.setEnabled(False)
            w = Worker(_run_status)
            w.done.connect(lambda r: (self._status_output.setPlainText(r), self._status_btn.setEnabled(True)))
            self._workers.append(w); w.start()

    app = QApplication(sys.argv)
    win = MainWindow()
    win._on_status()  # load status on start
    win.show()
    sys.exit(app.exec())


# ---------------------------------------------------------------------------
# Tkinter fallback UI
# ---------------------------------------------------------------------------

def _launch_tkinter() -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title(f"scilit-agent — {Config.TOPIC_NAME}")
    root.geometry("850x620")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    def _async(fn, *args, output_widget: tk.Text, btn: tk.Button | None = None):
        """Run fn(*args) in a thread; write result to output_widget."""
        if btn:
            btn.config(state="disabled")
        output_widget.config(state="normal")
        output_widget.delete("1.0", "end")
        output_widget.insert("end", "Working…")
        output_widget.config(state="disabled")

        def _run():
            try:
                result = fn(*args)
            except Exception as exc:
                result = f"Error: {exc}"
            root.after(0, _set_output, result)

        def _set_output(text: str):
            output_widget.config(state="normal")
            output_widget.delete("1.0", "end")
            output_widget.insert("end", text)
            output_widget.config(state="disabled")
            if btn:
                btn.config(state="normal")

        threading.Thread(target=_run, daemon=True).start()

    def _text_widget(parent, height=20) -> tk.Text:
        t = tk.Text(parent, height=height, wrap="word", state="disabled",
                    font=("Helvetica", 11))
        sb = ttk.Scrollbar(parent, command=t.yview)
        t.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        t.pack(fill="both", expand=True)
        return t

    # ── Query tab ────────────────────────────────────────────────────────
    qf = ttk.Frame(notebook); notebook.add(qf, text="🔍 Query")
    ttk.Label(qf, text="Question:").pack(anchor="w", padx=6, pady=(6, 0))
    q_entry = tk.Text(qf, height=3, font=("Helvetica", 11))
    q_entry.pack(fill="x", padx=6)

    q_row = ttk.Frame(qf); q_row.pack(fill="x", padx=6, pady=4)
    ttk.Label(q_row, text="Year:").grid(row=0, column=0)
    q_year = ttk.Entry(q_row, width=8); q_year.grid(row=0, column=1, padx=4)
    ttk.Label(q_row, text="Source:").grid(row=0, column=2)
    q_src  = ttk.Entry(q_row, width=12); q_src.grid(row=0, column=3, padx=4)

    q_out = _text_widget(qf, height=14)

    def _on_q():
        question = q_entry.get("1.0", "end").strip()
        yr_s = q_year.get().strip()
        year = int(yr_s) if yr_s.isdigit() else None
        src  = q_src.get().strip() or None

        def _combined():
            ans, srcs = _run_query(question, year, src)
            return ans + "\n\n--- Sources ---\n" + srcs

        _async(_combined, output_widget=q_out, btn=q_btn)

    q_btn = ttk.Button(qf, text="Ask", command=_on_q)
    q_btn.pack(padx=6, pady=2)

    # ── Ingest tab ────────────────────────────────────────────────────────
    inf = ttk.Frame(notebook); notebook.add(inf, text="📥 Ingest")
    ttk.Label(inf, text="Search query:").pack(anchor="w", padx=6, pady=(6, 0))
    i_entry = ttk.Entry(inf, font=("Helvetica", 11)); i_entry.pack(fill="x", padx=6)
    i_out   = _text_widget(inf)

    def _on_i():
        _async(_run_ingest, i_entry.get().strip(), output_widget=i_out, btn=i_btn)

    i_btn = ttk.Button(inf, text="Ingest Papers", command=_on_i)
    i_btn.pack(padx=6, pady=4)

    # ── Lint tab ──────────────────────────────────────────────────────────
    lf    = ttk.Frame(notebook); notebook.add(lf, text="🔧 Lint")
    l_out = _text_widget(lf)

    def _on_l():
        _async(_run_lint, output_widget=l_out, btn=l_btn)

    l_btn = ttk.Button(lf, text="Run Lint Check", command=_on_l)
    l_btn.pack(padx=6, pady=6)

    # ── Status tab ────────────────────────────────────────────────────────
    sf    = ttk.Frame(notebook); notebook.add(sf, text="📊 Status")
    s_out = _text_widget(sf)

    def _on_s():
        _async(_run_status, output_widget=s_out, btn=s_btn)

    s_btn = ttk.Button(sf, text="Refresh", command=_on_s)
    s_btn.pack(padx=6, pady=6)

    # Load status on start
    _on_s()
    root.mainloop()


# ---------------------------------------------------------------------------
# Entry point — try PyQt6, fall back to Tkinter
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        import PyQt6  # noqa: F401
        _launch_pyqt()
    except ImportError:
        print("PyQt6 not found — using Tkinter fallback.")
        _launch_tkinter()


if __name__ == "__main__":
    main()
