"""Shared naive-compare pane: Kimi buckets, phrase table, private-only-by-label bar."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

_VERDICT_COLORS = {
    "private-only": QColor("#f8d7da"),
    "overlap": QColor("#d4edda"),
    "unscored": QColor("#fff3cd"),
}


class _CompareWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            result = self._fn()
            self.done.emit(result)
        except Exception as exc:
            import traceback

            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class NaiveCompareWidget(QGroupBox):
    """Run and display a Kimi compare of approved phrases vs a public pack."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        exploration_getter: Optional[Callable[[], str]] = None,
    ):
        super().__init__("Naive corpus compare (Kimi)", parent)
        self._exploration_getter = exploration_getter
        self._projects = None
        self._worker: Optional[_CompareWorker] = None
        self._init_ui()

    def bind_project(self, controller) -> None:
        if self._projects is controller:
            self.reload()
            return
        self._projects = controller
        controller.changed.connect(self._on_project_changed)
        self._on_project_changed(controller.current)

    def reload(self) -> None:
        project = self._current_project()
        if project is None:
            self._set_status("Select a project to compare corpora.")
            self._clear_result()
            return
        from moyo.compare.naive import load_result

        saved = load_result(project)
        if saved is None:
            self._set_status(
                "Approved phrases vs extracted.json (else exploration.md / sources.json). "
                "Kimi judgment — not cosine distance."
            )
            return
        self._render(saved)

    def showEvent(self, event):
        super().showEvent(event)
        if self._projects is not None and self._worker is None:
            self.reload()

    def _init_ui(self) -> None:
        layout = QVBoxLayout()

        warn = QLabel(
            "Sends approved private phrases and the extracted public corpus to Moonshot Kimi. "
            "Uses public_sources/extracted.json when present. "
            "This is a qualitative judgment, not Barrier Probe distance. "
            "Requires MOONSHOT_API_KEY. Rejected phrases are not sent."
        )
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #555;")
        layout.addWidget(warn)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Compare private vs public")
        self.run_btn.clicked.connect(self._run)
        btn_row.addWidget(self.run_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #555;")
        layout.addWidget(self.status)

        self.headline = QLabel("")
        self.headline.setWordWrap(True)
        self.headline.setFont(QFont("Arial", 11))
        self.headline.setVisible(False)
        layout.addWidget(self.headline)

        self.caveats = QLabel("")
        self.caveats.setWordWrap(True)
        self.caveats.setStyleSheet("color: #555;")
        self.caveats.setVisible(False)
        layout.addWidget(self.caveats)

        buckets = QHBoxLayout()
        self._private_box, self._private_list = self._make_bucket("Only in private")
        self._overlap_box, self._overlap_list = self._make_bucket("Already public")
        self._public_box, self._public_list = self._make_bucket("Only in public")
        buckets.addWidget(self._private_box)
        buckets.addWidget(self._overlap_box)
        buckets.addWidget(self._public_box)
        layout.addLayout(buckets)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Phrase", "Label", "Kimi verdict", "Public quote"]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMinimumHeight(160)
        layout.addWidget(self.table)

        if MATPLOTLIB_AVAILABLE:
            chart_box = QGroupBox("Private-only phrases by label")
            chart_layout = QVBoxLayout()
            self.figure = Figure(figsize=(7, 2.4))
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumHeight(180)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            chart_layout.addWidget(self.canvas)
            chart_box.setLayout(chart_layout)
            layout.addWidget(chart_box)
        else:
            self.figure = None
            self.canvas = None

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Monospace"))
        self.log.setMinimumHeight(60)
        self.log.setMaximumHeight(90)
        layout.addWidget(self.log)

        self.setLayout(layout)
        self._set_status(
            "Approved phrases vs extracted.json (else exploration.md / sources.json). "
            "Kimi judgment — not cosine distance."
        )

    def _make_bucket(self, title: str) -> tuple[QGroupBox, QListWidget]:
        box = QGroupBox(f"{title} — 0")
        inner = QVBoxLayout()
        listing = QListWidget()
        listing.setMaximumHeight(110)
        inner.addWidget(listing)
        box.setLayout(inner)
        return box, listing

    def _on_project_changed(self, project) -> None:
        self.reload()

    def _current_project(self):
        if self._projects is None:
            return None
        return self._projects.current

    def _exploration_path(self) -> Optional[str]:
        if self._exploration_getter is None:
            return None
        try:
            value = (self._exploration_getter() or "").strip()
        except Exception:
            return None
        return value or None

    def _run(self) -> None:
        if self._worker is not None:
            QMessageBox.information(self, "Busy", "A compare is already running.")
            return
        project = self._current_project()
        if project is None:
            QMessageBox.warning(
                self,
                "No project",
                "Select or create a project in the toolbar first.",
            )
            return

        exploration = self._exploration_path()
        holder = {"worker": None}

        def job():
            from moyo.compare.naive import run_naive_compare

            def progress(msg: str) -> None:
                worker = holder["worker"]
                if worker is not None:
                    worker.log.emit(msg)
                else:
                    print(msg)

            return run_naive_compare(
                project=project,
                exploration_path=exploration,
                progress=progress,
            )

        self.log.clear()
        self.log.append("Comparing approved phrases to the public pack with Kimi…")
        self.progress_bar.setVisible(True)
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Compare private vs public…")
        self._worker = _CompareWorker(job)
        holder["worker"] = self._worker
        self._worker.log.connect(self.log.append)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, result) -> None:
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Compare private vs public")
        self._worker = None
        self.log.append("Done.")
        self._render(result)

    def _on_failed(self, message: str) -> None:
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Compare private vs public")
        self._worker = None
        self.log.append(f"Failed: {message}")
        QMessageBox.critical(self, "Compare failed", message.split("\n", 1)[0])

    def _clear_result(self) -> None:
        self.headline.setVisible(False)
        self.headline.setText("")
        self.caveats.setVisible(False)
        self.caveats.setText("")
        self._fill_bucket(self._private_box, self._private_list, "Only in private", [])
        self._fill_bucket(self._overlap_box, self._overlap_list, "Already public", [])
        self._fill_bucket(self._public_box, self._public_list, "Only in public", [])
        self.table.setRowCount(0)
        self._draw_chart({})

    def _render(self, result) -> None:
        packing = result.packing or {}
        priv = packing.get("private") or {}
        pub = packing.get("public") or {}
        used = packing.get("used") or (priv.get("chars", 0) + pub.get("chars", 0))
        budget = packing.get("budget") or 0
        trunc = []
        if priv.get("truncated"):
            trunc.append(f"{priv.get('omitted_items', 0)} phrases omitted")
        if pub.get("truncated"):
            trunc.append(f"public truncated ({pub.get('omitted_chars', 0)} chars)")
        trunc_s = f"  Truncated: {'; '.join(trunc)}." if trunc else ""
        pub_kind = pub.get("kind") or "public"
        pub_path = pub.get("path") or ""
        loc = Path(pub_path).name if pub_path else pub_kind
        self._set_status(
            f"Packed {priv.get('items', 0)} phrases ({priv.get('chars', 0)} chars) + "
            f"{pub_kind} {loc} ({pub.get('chars', 0)} chars). "
            f"Prompt {used}/{budget} chars.{trunc_s}"
        )

        if result.headline:
            self.headline.setText(result.headline)
            self.headline.setVisible(True)
        else:
            self.headline.setVisible(False)

        if result.caveats:
            self.caveats.setText("Caveats: " + "; ".join(result.caveats))
            self.caveats.setVisible(True)
        else:
            self.caveats.setVisible(False)

        self._fill_bucket(
            self._private_box,
            self._private_list,
            "Only in private",
            [i.text for i in result.only_private],
        )
        self._fill_bucket(
            self._overlap_box,
            self._overlap_list,
            "Already public",
            [i.text for i in result.overlap],
        )
        self._fill_bucket(
            self._public_box,
            self._public_list,
            "Only in public",
            [i.text for i in result.only_public],
        )

        rows = result.phrase_rows or []
        self.table.setRowCount(0)
        for item in rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [item.text, item.label, item.verdict, item.quote]
            color = _VERDICT_COLORS.get(item.verdict)
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if color is not None:
                    cell.setBackground(color)
                self.table.setItem(row, col, cell)

        from moyo.compare.naive import private_only_by_label

        self._draw_chart(private_only_by_label(result))

    def _fill_bucket(
        self, box: QGroupBox, listing: QListWidget, title: str, texts: list[str]
    ) -> None:
        box.setTitle(f"{title} — {len(texts)}")
        listing.clear()
        shown = texts[:8]
        for text in shown:
            listing.addItem(QListWidgetItem(text))
        extra = len(texts) - len(shown)
        if extra > 0:
            listing.addItem(QListWidgetItem(f"… {extra} more"))

    def _draw_chart(self, counts: dict) -> None:
        if self.figure is None:
            return
        from moyo.privateside.phrases.schema import LABELS

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        labels = list(LABELS)
        values = [int(counts.get(name, 0) or 0) for name in labels]
        ax.barh(labels, values, color="#C0392B")
        ax.set_xlabel("Private-only phrases")
        ax.set_title("Private-only phrases by label")
        ax.invert_yaxis()
        xmax = max(values) if any(values) else 1
        ax.set_xlim(0, xmax + 0.5)
        self.figure.tight_layout()
        self.canvas.draw()

    def _set_status(self, text: str) -> None:
        self.status.setText(text)
