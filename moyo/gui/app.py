#!/usr/bin/env python3
"""
moyo GUI - A comprehensive GUI for the moyo project.
Provides tabs for data input, FAISS index creation, and 2D visualization.
"""

import sys
import json
import re
from pathlib import Path
from typing import Optional, Dict

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QTextEdit, QFileDialog, QProgressBar,
    QGroupBox, QScrollArea, QComboBox, QLineEdit, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QCheckBox,
    QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox, QFormLayout,
    QListWidget, QListWidgetItem, QPlainTextEdit, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor

# When running without an editable install, add the repo root so that
# "moyo" and "shared_utils" are importable.  With `pip install -e .` this
# block is a no-op because the packages are already on sys.path.
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import numpy as np
    from sklearn.manifold import MDS
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: visualization dependencies unavailable ({e}). "
          "Install with: pip install moyo[gui]")
    MATPLOTLIB_AVAILABLE = False


class BackgroundWorker(QThread):
    """Reusable worker that runs a callable on a background thread.

    The callable can return any pickleable result; it is delivered via the
    ``done`` signal.  Stdout/stderr captured during the call is streamed via
    ``log``.  Exceptions are caught and surfaced via ``failed``.
    """

    log = pyqtSignal(str)
    done = pyqtSignal(object)        # arbitrary result payload
    failed = pyqtSignal(str)         # error message

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr

        buf = io.StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                result = self._fn(*self._args, **self._kwargs)
            text = buf.getvalue()
            if text:
                self.log.emit(text)
            self.done.emit(result)
        except Exception as exc:
            text = buf.getvalue()
            if text:
                self.log.emit(text)
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class DataInputTab(QWidget):
    """Tab for data input functionality — text, file, or folder to corpus."""

    def __init__(self):
        super().__init__()
        self.corpus_data = []
        self.default_corpus_path = _repo_root / "data" / "private" / "corpus.txt"
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Data Input - Text, File, or Folder to Corpus")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        # Input method selection — QButtonGroup ensures mutual exclusion.
        method_group = QGroupBox("Input Method")
        method_layout = QVBoxLayout()

        self._input_method_group = QButtonGroup(self)
        self.text_radio = QRadioButton("Direct Text Input")
        self.file_radio = QRadioButton("Single File")
        self.folder_radio = QRadioButton("Folder")

        self._input_method_group.addButton(self.text_radio)
        self._input_method_group.addButton(self.file_radio)
        self._input_method_group.addButton(self.folder_radio)

        self.text_radio.setChecked(True)
        self.text_radio.toggled.connect(self.on_input_method_changed)
        self.file_radio.toggled.connect(self.on_input_method_changed)
        self.folder_radio.toggled.connect(self.on_input_method_changed)

        method_layout.addWidget(self.text_radio)
        method_layout.addWidget(self.file_radio)
        method_layout.addWidget(self.folder_radio)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Text input section
        self.text_group = QGroupBox("Direct Text Input")
        text_layout = QVBoxLayout()
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text directly here...")
        text_layout.addWidget(self.text_input)
        self.text_group.setLayout(text_layout)
        layout.addWidget(self.text_group)

        # File input section
        self.file_group = QGroupBox("File Input")
        file_layout = QVBoxLayout()

        file_btn_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        select_file_btn = QPushButton("Select .txt File")
        select_file_btn.clicked.connect(self.select_file)
        file_btn_layout.addWidget(self.file_path_label)
        file_btn_layout.addWidget(select_file_btn)
        file_layout.addLayout(file_btn_layout)

        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("File Type:"))
        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems(["txt", "text", "pdf", "docx", "csv", "json"])
        self.file_type_combo.setCurrentText("txt")
        type_layout.addWidget(self.file_type_combo)
        file_layout.addLayout(type_layout)

        self.file_group.setLayout(file_layout)
        self.file_group.setVisible(False)
        layout.addWidget(self.file_group)

        # Folder input section
        self.folder_group = QGroupBox("Folder Input")
        folder_layout = QVBoxLayout()

        folder_btn_layout = QHBoxLayout()
        self.folder_path_label = QLabel("No folder selected")
        select_folder_btn = QPushButton("Select Folder")
        select_folder_btn.clicked.connect(self.select_folder)
        folder_btn_layout.addWidget(self.folder_path_label)
        folder_btn_layout.addWidget(select_folder_btn)
        folder_layout.addLayout(folder_btn_layout)

        ext_layout = QHBoxLayout()
        ext_layout.addWidget(QLabel("File Extensions:"))
        self.file_extensions = QLineEdit("*.txt")
        ext_layout.addWidget(self.file_extensions)
        folder_layout.addLayout(ext_layout)

        self.folder_group.setLayout(folder_layout)
        self.folder_group.setVisible(False)
        layout.addWidget(self.folder_group)

        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout()

        self.use_default_checkbox = QCheckBox(
            "Use default location (data/private/corpus.txt)"
        )
        self.use_default_checkbox.setChecked(True)
        self.use_default_checkbox.toggled.connect(self.on_output_method_changed)
        output_layout.addWidget(self.use_default_checkbox)

        custom_layout = QHBoxLayout()
        self.custom_path_label = QLabel("No custom path selected")
        select_output_btn = QPushButton("Select Custom Output File")
        select_output_btn.clicked.connect(self.select_output_file)
        custom_layout.addWidget(self.custom_path_label)
        custom_layout.addWidget(select_output_btn)
        output_layout.addLayout(custom_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        self.process_btn = QPushButton("Process Data and Save to Corpus")
        self.process_btn.clicked.connect(self.process_data)
        layout.addWidget(self.process_btn)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        self.setLayout(layout)

    def on_input_method_changed(self):
        self.text_group.setVisible(self.text_radio.isChecked())
        self.file_group.setVisible(self.file_radio.isChecked())
        self.folder_group.setVisible(self.folder_radio.isChecked())

    def on_output_method_changed(self):
        use_default = self.use_default_checkbox.isChecked()
        self.custom_path_label.setEnabled(not use_default)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", "",
            "Text Files (*.txt);;All Files (*);;PDF Files (*.pdf);;"
            "Word Files (*.docx);;CSV Files (*.csv);;JSON Files (*.json)",
        )
        if file_path:
            self.file_path_label.setText(file_path)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder_path:
            self.folder_path_label.setText(folder_path)

    def select_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", "", "Text Files (*.txt)"
        )
        if file_path:
            self.custom_path_label.setText(file_path)

    def process_data(self):
        self.output_text.append("Processing data input...")
        self.progress_bar.setValue(0)

        try:
            if self.text_radio.isChecked():
                data = self.process_text_input()
            elif self.file_radio.isChecked():
                data = self.process_file_input()
            elif self.folder_radio.isChecked():
                data = self.process_folder_input()
            else:
                self.output_text.append("Please select an input method")
                return

            if not data:
                self.output_text.append("No data to process")
                return

            self.progress_bar.setValue(50)

            if self.use_default_checkbox.isChecked():
                output_path = self.default_corpus_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_path = Path(self.custom_path_label.text())

            self.save_corpus(data, output_path)

            self.progress_bar.setValue(100)
            self.output_text.append(f"Corpus saved to: {output_path}")
            self.output_text.append(f"Processed {len(data)} items")

        except Exception as e:
            self.output_text.append(f"Error processing data: {e}")
            self.progress_bar.setValue(0)

    def process_text_input(self):
        text_content = self.text_input.toPlainText()
        if not text_content.strip():
            return []
        lines = text_content.strip().split("\n")
        return [
            {"id": f"text_line_{i}", "text": line.strip(), "source": "direct_text", "chunk_id": i}
            for i, line in enumerate(lines)
            if line.strip()
        ]

    def process_file_input(self):
        file_path = self.file_path_label.text()
        if file_path == "No file selected":
            return []
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
            return [
                {"id": f"file_line_{i}", "text": line.strip(), "source": Path(file_path).name, "chunk_id": i}
                for i, line in enumerate(lines)
                if line.strip()
            ]
        except Exception as e:
            self.output_text.append(f"Error reading file: {e}")
            return []

    def process_folder_input(self):
        folder_path = self.folder_path_label.text()
        if folder_path == "No folder selected":
            return []
        try:
            folder = Path(folder_path)
            extensions = [ext.strip() for ext in self.file_extensions.text().split(",")]
            all_items = []
            counter = 0
            for ext in extensions:
                for fp in folder.glob(ext):
                    try:
                        with open(fp, encoding="utf-8") as f:
                            lines = f.readlines()
                        for line in lines:
                            if line.strip():
                                all_items.append({
                                    "id": f"{fp.stem}_line_{counter}",
                                    "text": line.strip(),
                                    "source": fp.name,
                                    "chunk_id": counter,
                                })
                                counter += 1
                    except Exception as e:
                        self.output_text.append(f"Error reading {fp}: {e}")
            return all_items
        except Exception as e:
            self.output_text.append(f"Error processing folder: {e}")
            return []

    def save_corpus(self, data, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(f"ID: {item['id']}\n")
                f.write(f"Source: {item['source']}\n")
                f.write(f"Text: {item['text']}\n")
                f.write("-" * 50 + "\n")


def _populate_embedding_model_combo(combo: QComboBox, default_key: str = "mini") -> None:
    """Fill a combo with catalog labels; itemData stores the model key."""
    from shared_utils.model_config import list_embedding_choices

    combo.clear()
    default_index = 0
    for i, entry in enumerate(list_embedding_choices()):
        combo.addItem(entry["label"], entry["key"])
        combo.setItemData(i, entry["description"], Qt.ToolTipRole)
        if entry["key"] == default_key:
            default_index = i
    combo.setCurrentIndex(default_index)


def _device_status_text() -> str:
    from shared_utils.embeddings import get_device_info

    info = get_device_info()
    if info["cuda_available"]:
        name = info.get("gpu_name") or "GPU"
        mem = info.get("gpu_memory_gb")
        mem_s = f", {mem} GB" if mem is not None else ""
        return f"CUDA available ({name}{mem_s}) — auto will use GPU"
    return (
        "CUDA not available — embeddings will run on CPU. "
        "On WSL2 check nvidia-smi / /dev/nvidia* and restart WSL if needed."
    )


class FAISSIndexTab(QWidget):
    """Tab for creating a FAISS index from a corpus."""

    def __init__(self):
        super().__init__()
        self.default_corpus_path = _repo_root / "data" / "private" / "corpus.txt"
        self.default_private_index_path = _repo_root / "indexes" / "private"
        self.default_public_index_path = _repo_root / "indexes" / "public"
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Create FAISS Index from Corpus")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        corpus_group = QGroupBox("Corpus Selection")
        corpus_layout = QVBoxLayout()

        self.use_default_corpus_checkbox = QCheckBox(
            "Use default corpus (data/private/corpus.txt)"
        )
        self.use_default_corpus_checkbox.setChecked(True)
        self.use_default_corpus_checkbox.toggled.connect(self.on_corpus_method_changed)
        corpus_layout.addWidget(self.use_default_corpus_checkbox)

        corpus_btn_layout = QHBoxLayout()
        self.corpus_path_label = QLabel("No custom corpus selected")
        select_corpus_btn = QPushButton("Select Custom Corpus File")
        select_corpus_btn.clicked.connect(self.select_corpus)
        corpus_btn_layout.addWidget(self.corpus_path_label)
        corpus_btn_layout.addWidget(select_corpus_btn)
        corpus_layout.addLayout(corpus_btn_layout)
        corpus_group.setLayout(corpus_layout)
        layout.addWidget(corpus_group)

        index_group = QGroupBox("Index Options")
        index_layout = QVBoxLayout()

        index_type_layout = QHBoxLayout()
        index_type_layout.addWidget(QLabel("Index Type:"))
        self.index_type_combo = QComboBox()
        self.index_type_combo.addItems(["IndexFlatL2", "IndexIVFFlat", "IndexHNSW"])
        index_type_layout.addWidget(self.index_type_combo)
        index_layout.addLayout(index_type_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Embedding Model:"))
        self.embedding_model_combo = QComboBox()
        _populate_embedding_model_combo(self.embedding_model_combo, default_key="mini")
        self.embedding_model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.embedding_model_combo)
        index_layout.addLayout(model_layout)

        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (CUDA if available)", "auto")
        self.device_combo.addItem("CUDA (GPU)", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.setCurrentIndex(0)
        device_layout.addWidget(self.device_combo)
        index_layout.addLayout(device_layout)

        self.device_status_label = QLabel(_device_status_text())
        self.device_status_label.setWordWrap(True)
        self.device_status_label.setStyleSheet("color: #555;")
        index_layout.addWidget(self.device_status_label)

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Dimension:"))
        self.dimension_label = QLabel("384")
        params_layout.addWidget(self.dimension_label)
        self.model_hint_label = QLabel("")
        self.model_hint_label.setWordWrap(True)
        self.model_hint_label.setStyleSheet("color: #555;")
        params_layout.addWidget(self.model_hint_label, stretch=1)
        index_layout.addLayout(params_layout)

        index_group.setLayout(index_layout)
        layout.addWidget(index_group)
        self.on_model_changed()

        output_group = QGroupBox("Index Output")
        output_layout = QVBoxLayout()

        dest_layout = QHBoxLayout()
        dest_layout.addWidget(QLabel("Create Index For:"))
        self.index_type_radio = QComboBox()
        self.index_type_radio.addItems(["Private", "Public"])
        self.index_type_radio.currentTextChanged.connect(self.on_index_type_changed)
        dest_layout.addWidget(self.index_type_radio)
        output_layout.addLayout(dest_layout)

        self.use_default_index_checkbox = QCheckBox("Use default location")
        self.use_default_index_checkbox.setChecked(True)
        self.use_default_index_checkbox.toggled.connect(self.on_index_output_changed)
        output_layout.addWidget(self.use_default_index_checkbox)

        index_output_layout = QHBoxLayout()
        self.index_output_label = QLabel("No custom index location selected")
        select_index_output_btn = QPushButton("Select Custom Index Location")
        select_index_output_btn.clicked.connect(self.select_index_output)
        index_output_layout.addWidget(self.index_output_label)
        index_output_layout.addWidget(select_index_output_btn)
        output_layout.addLayout(index_output_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        self.create_index_btn = QPushButton("Create FAISS Index")
        self.create_index_btn.clicked.connect(self.create_index)
        layout.addWidget(self.create_index_btn)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        self.setLayout(layout)

    def on_corpus_method_changed(self):
        self.corpus_path_label.setEnabled(not self.use_default_corpus_checkbox.isChecked())

    def on_index_output_changed(self):
        self.index_output_label.setEnabled(not self.use_default_index_checkbox.isChecked())

    def on_model_changed(self):
        from shared_utils.model_config import get_catalog_entry

        model_key = self.embedding_model_combo.currentData() or "mini"
        entry = get_catalog_entry(model_key) or {}
        self.dimension_label.setText(str(entry.get("dimensions", 384)))
        hint = entry.get("description", "")
        if entry.get("backend") == "openai":
            hint = f"{hint} (device ignored — API)"
            self.device_combo.setEnabled(False)
        else:
            self.device_combo.setEnabled(True)
        self.model_hint_label.setText(hint)

    def on_index_type_changed(self):
        index_type = self.index_type_radio.currentText()
        loc = "indexes/private" if index_type == "Private" else "indexes/public"
        self.use_default_index_checkbox.setText(f"Use default location ({loc})")

    def select_corpus(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Corpus File", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.corpus_path_label.setText(file_path)

    def select_index_output(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Index Output Directory")
        if folder_path:
            self.index_output_label.setText(folder_path)

    def create_index(self):
        self.output_text.append("Creating FAISS index...")
        self.progress_bar.setValue(0)

        try:
            corpus_path = (
                self.default_corpus_path
                if self.use_default_corpus_checkbox.isChecked()
                else Path(self.corpus_path_label.text())
            )

            if not corpus_path.exists():
                self.output_text.append(f"Corpus file not found: {corpus_path}")
                return

            self.progress_bar.setValue(10)

            index_type = self.index_type_radio.currentText()
            if self.use_default_index_checkbox.isChecked():
                index_path = (
                    self.default_private_index_path
                    if index_type == "Private"
                    else self.default_public_index_path
                )
            else:
                index_path = Path(self.index_output_label.text())

            index_path.mkdir(parents=True, exist_ok=True)
            self.progress_bar.setValue(20)

            corpus_data = []
            with open(corpus_path, encoding="utf-8") as f:
                content = f.read()
                for section in content.split("-" * 50):
                    if section.strip():
                        item = {}
                        for line in section.strip().split("\n"):
                            if line.startswith("ID: "):
                                item["id"] = line[4:].strip()
                            elif line.startswith("Source: "):
                                item["source"] = line[8:].strip()
                            elif line.startswith("Text: "):
                                item["text"] = line[6:].strip()
                        if item.get("text", "").strip():
                            corpus_data.append(item)

            self.output_text.append(f"Loaded corpus with {len(corpus_data)} items")
            self.progress_bar.setValue(30)

            texts = [item["text"] for item in corpus_data if item.get("text", "").strip()]
            if not texts:
                self.output_text.append("No valid text data found in corpus")
                return

            self.output_text.append(f"Processing {len(texts)} text items...")
            self.progress_bar.setValue(40)

            model_key = self.embedding_model_combo.currentData() or "mini"
            from shared_utils.model_config import get_catalog_entry, resolve_model_name
            entry = get_catalog_entry(model_key) or {}
            model_name = resolve_model_name(model_key)
            device = self.device_combo.currentData() or "auto"

            try:
                from shared_utils.embeddings import embed, resolve_device
                from shared_utils.faiss_index import FAISSIndex
                import faiss
                resolved = resolve_device(device) if entry.get("backend") != "openai" else "api"
                gpu_msg = "GPU" if hasattr(faiss, "GpuIndexFlatIP") else "CPU"
                self.output_text.append(f"Using FAISS {gpu_msg} backend")
                self.output_text.append(f"Embedding device: {resolved}")
            except ImportError as e:
                self.output_text.append(f"Error importing shared_utils: {e}")
                return

            self.output_text.append(f"Generating embeddings using {model_name}...")
            self.progress_bar.setValue(50)
            embeddings = embed(
                texts,
                model_name=model_name,
                batch_size=32,
                normalize=True,
                device=device,
            )

            if not embeddings:
                self.output_text.append("Failed to generate embeddings")
                return

            self.output_text.append(f"Generated {len(embeddings)} embeddings")
            self.progress_bar.setValue(70)

            raw_type = self.index_type_combo.currentText().lower()
            if "ivf" in raw_type:
                faiss_type = "ivf"
            elif "hnsw" in raw_type:
                faiss_type = "hnsw"
            else:
                faiss_type = "flat"

            dimension = entry.get("dimensions", 384)
            self.output_text.append(f"Creating {faiss_type.upper()} index (dim={dimension})...")
            self.progress_bar.setValue(80)

            faiss_index = FAISSIndex(dimension=dimension, index_type=faiss_type)
            metadata = [
                {
                    "id": item.get("id", f"item_{i}"),
                    "source": item.get("source", "unknown"),
                    "chunk_id": item.get("chunk_id", i),
                    "text": item["text"][:100] + "..." if len(item["text"]) > 100 else item["text"],
                }
                for i, item in enumerate(corpus_data)
                if item.get("text", "").strip()
            ]

            faiss_index.add_vectors(embeddings, metadata)
            self.progress_bar.setValue(90)

            corpus_name = re.sub(r"[^A-Za-z0-9._-]+", "_", corpus_path.stem).strip("._-") or "corpus"
            index_dir = index_path / corpus_name
            index_dir.mkdir(parents=True, exist_ok=True)
            self.output_text.append(f"Saving index to {index_dir}...")
            saved_path = faiss_index.save(index_dir, name=corpus_name)

            self.output_text.append("FAISS index created successfully!")
            self.output_text.append(f"Index saved to: {saved_path}")
            self.output_text.append(f"Total vectors: {faiss_index.get_vector_count()}")
            self.output_text.append(f"Index type: {faiss_type.upper()}, dimension: {dimension}")
            self.progress_bar.setValue(100)

        except Exception as e:
            import traceback
            self.output_text.append(f"Error creating index: {e}")
            self.output_text.append(traceback.format_exc())
            self.progress_bar.setValue(0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_SOURCE_TYPES = [
    "patent",
    "press_release",
    "git_commit",
    "conference_talk",
    "leaked_code",
]


def _make_log_pane(min_height: int = 120) -> QTextEdit:
    """Build a read-only monospace text pane used as a log view across tabs."""
    log = QTextEdit()
    log.setReadOnly(True)
    log.setFont(QFont("Monospace"))
    log.setMinimumHeight(min_height)
    return log


def _busy(button: QPushButton, busy: bool, idle_label: str) -> None:
    """Toggle a button between idle and busy states."""
    if busy:
        button.setText(f"⏳ {idle_label}…")
        button.setEnabled(False)
    else:
        button.setText(idle_label)
        button.setEnabled(True)


class GatherPublicSourcesTab(QWidget):
    """Tab for gathering public sources via the PublicSourcesCrawler."""

    def __init__(self):
        super().__init__()
        self._worker: Optional[BackgroundWorker] = None
        self._last_output_dir: Optional[Path] = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Gather Public Sources")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        desc = QLabel(
            "Crawl public sources (patents, press releases, git commits, conferences, leaks) "
            "by topic or token list. Output JSON is written under the chosen directory "
            "and can be fed into the Build Public Corpus tab."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # --- Mode: topic vs tokens
        mode_group = QGroupBox("Query Mode")
        mode_layout = QHBoxLayout()
        self._mode_buttons = QButtonGroup(self)
        self.topic_radio = QRadioButton("Single topic")
        self.tokens_radio = QRadioButton("Token list")
        self.prompt_radio = QRadioButton("Naive prompt (AI explore)")
        self.topic_radio.setChecked(True)
        self._mode_buttons.addButton(self.topic_radio)
        self._mode_buttons.addButton(self.tokens_radio)
        self._mode_buttons.addButton(self.prompt_radio)
        self.topic_radio.toggled.connect(self._refresh_mode)
        self.prompt_radio.toggled.connect(self._refresh_mode)
        mode_layout.addWidget(self.topic_radio)
        mode_layout.addWidget(self.tokens_radio)
        mode_layout.addWidget(self.prompt_radio)
        mode_layout.addStretch(1)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # --- Topic / tokens / naive-prompt input
        self.topic_input = QLineEdit()
        self.topic_input.setPlaceholderText("e.g. 'artificial intelligence safety'")
        self.tokens_input = QLineEdit()
        self.tokens_input.setPlaceholderText("Comma-separated: neural networks, transformers, LLM")
        self.tokens_input.setEnabled(False)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "Ask in plain language, e.g. 'give me all the info you can on the recipe for Coca-Cola'"
        )
        self.prompt_input.setFixedHeight(60)
        self.prompt_input.setEnabled(False)

        topic_row = QFormLayout()
        topic_row.addRow("Topic:", self.topic_input)
        topic_row.addRow("Tokens:", self.tokens_input)
        topic_row.addRow("Naive prompt:", self.prompt_input)
        layout.addLayout(topic_row)

        self.explore_note = QLabel(
            "Explore mode rewords your prompt via the local Ollama fuzzer "
            "(llama3.1:8b, black-box — no target concept). Mode basic paraphrases "
            "only; full rotates translate / abstract / summarize / typo. Asks every "
            "configured retrieval LLM (see config/retrieval_llms.json), then writes "
            "exploration.md plus a corroborated claims summary.md. Foreign-language "
            "results are translated to English with a language annotation. Optional "
            "impact criteria below refine what counts as high-impact (base definition "
            "covers classified, proprietary, and personal/sensitive information)."
        )
        self.explore_note.setWordWrap(True)
        self.explore_note.setVisible(False)
        layout.addWidget(self.explore_note)

        self.explore_fuzz_mode_combo = QComboBox()
        self.explore_fuzz_mode_combo.addItem("basic (paraphrase only)", "basic")
        self.explore_fuzz_mode_combo.addItem(
            "full (translate / abstract / summarize / typo)", "full"
        )
        self.explore_fuzz_row = QWidget()
        explore_fuzz_layout = QFormLayout(self.explore_fuzz_row)
        explore_fuzz_layout.setContentsMargins(0, 0, 0, 0)
        explore_fuzz_layout.addRow("Fuzz mode:", self.explore_fuzz_mode_combo)
        self.explore_fuzz_row.setVisible(False)
        layout.addWidget(self.explore_fuzz_row)

        self.impact_definition_input = QTextEdit()
        self.impact_definition_input.setPlaceholderText(
            "Optional: additional high-impact criteria for summary.md "
            "(appended to the built-in definition)"
        )
        self.impact_definition_input.setFixedHeight(50)
        self.impact_definition_input.setVisible(False)
        layout.addWidget(self.impact_definition_input)

        # --- Source type filter
        types_group = QGroupBox("Source Types (leave all unchecked = use defaults)")
        types_layout = QHBoxLayout()
        self._type_checks: Dict[str, QCheckBox] = {}
        for src in _SOURCE_TYPES:
            cb = QCheckBox(src.replace("_", " "))
            types_layout.addWidget(cb)
            self._type_checks[src] = cb
        types_group.setLayout(types_layout)
        layout.addWidget(types_group)

        # --- Crawl parameters
        params_group = QGroupBox("Crawl Parameters")
        params_layout = QFormLayout()
        self.max_per_source_spin = QSpinBox()
        self.max_per_source_spin.setRange(1, 5000)
        self.max_per_source_spin.setValue(100)
        params_layout.addRow("Max results per source:", self.max_per_source_spin)

        self.max_total_spin = QSpinBox()
        self.max_total_spin.setRange(1, 50000)
        self.max_total_spin.setValue(1000)
        params_layout.addRow("Max total results:", self.max_total_spin)

        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 60.0)
        self.delay_spin.setSingleStep(0.5)
        self.delay_spin.setValue(1.0)
        params_layout.addRow("Delay between requests (s):", self.delay_spin)

        self.output_dir_input = QLineEdit("data/public_sources")
        out_pick_btn = QPushButton("Browse…")
        out_pick_btn.clicked.connect(self._pick_output_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_dir_input)
        out_row.addWidget(out_pick_btn)
        params_layout.addRow("Output directory:", out_row)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # --- Actions
        action_row = QHBoxLayout()
        self.run_btn = QPushButton("Start Crawl")
        self.run_btn.clicked.connect(self._start_crawl)
        action_row.addWidget(self.run_btn)
        self.open_results_btn = QPushButton("Open Last Output Dir")
        self.open_results_btn.setEnabled(False)
        self.open_results_btn.clicked.connect(self._open_last_output)
        action_row.addWidget(self.open_results_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.log = _make_log_pane(160)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def _refresh_mode(self):
        is_topic = self.topic_radio.isChecked()
        is_prompt = self.prompt_radio.isChecked()
        is_tokens = self.tokens_radio.isChecked()
        self.topic_input.setEnabled(is_topic)
        self.tokens_input.setEnabled(is_tokens)
        self.prompt_input.setEnabled(is_prompt)
        self.explore_note.setVisible(is_prompt)
        self.explore_fuzz_row.setVisible(is_prompt)
        self.impact_definition_input.setVisible(is_prompt)
        self.run_btn.setText("Explore" if is_prompt else "Start Crawl")

    def _pick_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_input.setText(path)

    def _selected_source_types(self):
        from moyo.publicside.gatherpublicsources.schema import SourceType
        picks = [s for s, cb in self._type_checks.items() if cb.isChecked()]
        if not picks:
            return None
        return [SourceType(s) for s in picks]

    def _start_crawl(self):
        if self._worker is not None:
            QMessageBox.information(self, "Busy", "A crawl is already running.")
            return

        if self.prompt_radio.isChecked():
            self._start_explore()
            return

        topic_mode = self.topic_radio.isChecked()
        if topic_mode and not self.topic_input.text().strip():
            QMessageBox.warning(self, "Missing input", "Enter a topic.")
            return
        if not topic_mode and not self.tokens_input.text().strip():
            QMessageBox.warning(self, "Missing input", "Enter at least one token.")
            return

        try:
            from moyo.publicside.gatherpublicsources.schema import CrawlConfig
            from moyo.publicside.gatherpublicsources.crawler import PublicSourcesCrawler
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return

        output_dir = self.output_dir_input.text().strip() or "data/public_sources"
        source_types = self._selected_source_types()

        if topic_mode:
            topic = self.topic_input.text().strip()
            config = CrawlConfig(
                topic=topic,
                source_types=source_types or [],
                max_results_per_source=self.max_per_source_spin.value(),
                max_total_results=self.max_total_spin.value(),
                delay_between_requests=self.delay_spin.value(),
                output_directory=output_dir,
            )

            def job():
                crawler = PublicSourcesCrawler(config)
                return crawler.crawl(topic, source_types)
        else:
            tokens = [t.strip() for t in self.tokens_input.text().split(",") if t.strip()]
            config = CrawlConfig(
                topic=", ".join(tokens),
                source_types=source_types or [],
                max_results_per_source=self.max_per_source_spin.value(),
                max_total_results=self.max_total_spin.value(),
                delay_between_requests=self.delay_spin.value(),
                output_directory=output_dir,
            )

            def job():
                crawler = PublicSourcesCrawler(config)
                return crawler.crawl_with_tokens(tokens, source_types)

        self._last_output_dir = Path(output_dir)
        self.log.clear()
        self.log.append("Starting crawl…")
        self.progress_bar.setVisible(True)
        _busy(self.run_btn, True, "Start Crawl")

        self._worker = BackgroundWorker(job)
        self._worker.log.connect(self.log.append)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _start_explore(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing input", "Enter a naive prompt to explore.")
            return

        try:
            from moyo.publicside.gatherpublicsources.explorer import explore_and_save
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return

        output_dir = self.output_dir_input.text().strip() or "data/public_sources"
        impact_extra = self.impact_definition_input.toPlainText().strip() or None
        fuzz_mode = self.explore_fuzz_mode_combo.currentData() or "basic"

        def job():
            return explore_and_save(
                prompt,
                output_directory=output_dir,
                impact_definition=impact_extra,
                fuzz_mode=fuzz_mode,
                progress=print,
            )

        self._last_output_dir = Path(output_dir)
        self.log.clear()
        self.log.append(f"Exploring prompt across configured LLMs (fuzz_mode={fuzz_mode})…")
        self.progress_bar.setVisible(True)
        _busy(self.run_btn, True, "Explore")

        self._worker = BackgroundWorker(job)
        self._worker.log.connect(self.log.append)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, result):
        try:
            # Explore result (naive-prompt mode) vs crawl result.
            if hasattr(result, "seeds") and hasattr(result, "markdown"):
                from moyo.publicside.gatherpublicsources.explorer import (
                    format_retrieval_table,
                )

                ok = sum(1 for r in result.results if getattr(r, "ok", False))
                self.log.append(
                    f"✅ Done. seeds={len(result.seeds)} "
                    f"llms={len(result.llm_labels)} successful={ok}/{len(result.results)}"
                )
                self.log.append(format_retrieval_table(result))
                if result.output_path:
                    self.log.append(f"Report: {result.output_path}")
                    self._last_output_dir = Path(result.output_path)
                    self.open_results_btn.setEnabled(True)
                if getattr(result, "summary_path", None):
                    self.log.append(f"Summary: {result.summary_path}")
                return

            sources_found = getattr(result, "sources_found", 0)
            sources_processed = getattr(result, "sources_processed", 0)
            sources_failed = getattr(result, "sources_failed", 0)
            output_path = getattr(result, "output_path", None)

            self.log.append(
                f"✅ Done. found={sources_found} processed={sources_processed} "
                f"failed={sources_failed}"
            )
            if output_path:
                self.log.append(f"Output: {output_path}")
                self._last_output_dir = Path(output_path)
                self.open_results_btn.setEnabled(True)
        finally:
            self._cleanup_worker()

    def _on_failed(self, msg: str):
        self.log.append(f"❌ {msg}")
        QMessageBox.critical(self, "Task failed", msg)
        self._cleanup_worker()

    def _cleanup_worker(self):
        self._worker = None
        self.progress_bar.setVisible(False)
        label = "Explore" if self.prompt_radio.isChecked() else "Start Crawl"
        _busy(self.run_btn, False, label)

    def _open_last_output(self):
        if self._last_output_dir and self._last_output_dir.exists():
            QMessageBox.information(
                self, "Output directory", str(self._last_output_dir.resolve())
            )


class BuildPublicCorpusTab(QWidget):
    """Tab for building a public FAISS index from gathered public sources."""

    def __init__(self):
        super().__init__()
        self._worker: Optional[BackgroundWorker] = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        title = QLabel("Build Public Corpus Index")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        desc = QLabel(
            "Embed crawled public sources (sources.json files) into a FAISS index. "
            "Inputs are typically produced by the Gather Public Sources tab."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # --- Input
        input_group = QGroupBox("Input Sources")
        input_layout = QFormLayout()

        self.sources_dir_input = QLineEdit("data/public_sources")
        sources_btn = QPushButton("Browse…")
        sources_btn.clicked.connect(self._pick_sources_dir)
        sources_row = QHBoxLayout()
        sources_row.addWidget(self.sources_dir_input)
        sources_row.addWidget(sources_btn)
        input_layout.addRow("Sources directory:", sources_row)

        self.name_input = QLineEdit("public_index")
        input_layout.addRow("Index name:", self.name_input)

        self.description_input = QLineEdit("")
        self.description_input.setPlaceholderText("Optional description for the index")
        input_layout.addRow("Description:", self.description_input)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # --- Embedding / chunking
        embed_group = QGroupBox("Embedding & Chunking")
        embed_layout = QFormLayout()

        self.model_combo = QComboBox()
        _populate_embedding_model_combo(self.model_combo, default_key="mini")
        self.model_combo.currentIndexChanged.connect(self._on_public_model_changed)
        embed_layout.addRow("Embedding model:", self.model_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (CUDA if available)", "auto")
        self.device_combo.addItem("CUDA (GPU)", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        embed_layout.addRow("Device:", self.device_combo)

        self.device_status_label = QLabel(_device_status_text())
        self.device_status_label.setWordWrap(True)
        self.device_status_label.setStyleSheet("color: #555;")
        embed_layout.addRow("", self.device_status_label)

        self.model_hint_label = QLabel("")
        self.model_hint_label.setWordWrap(True)
        self.model_hint_label.setStyleSheet("color: #555;")
        embed_layout.addRow("", self.model_hint_label)

        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(64, 8192)
        self.chunk_size_spin.setValue(512)
        embed_layout.addRow("Chunk size:", self.chunk_size_spin)

        self.chunk_overlap_spin = QSpinBox()
        self.chunk_overlap_spin.setRange(0, 4096)
        self.chunk_overlap_spin.setValue(50)
        embed_layout.addRow("Chunk overlap:", self.chunk_overlap_spin)

        self.min_len_spin = QSpinBox()
        self.min_len_spin.setRange(1, 10000)
        self.min_len_spin.setValue(50)
        embed_layout.addRow("Min chunk length:", self.min_len_spin)

        self.max_len_spin = QSpinBox()
        self.max_len_spin.setRange(50, 20000)
        self.max_len_spin.setValue(2000)
        embed_layout.addRow("Max chunk length:", self.max_len_spin)

        self.index_type_combo = QComboBox()
        self.index_type_combo.addItems(["flat", "ivf", "hnsw", "pq"])
        embed_layout.addRow("Index type:", self.index_type_combo)

        embed_group.setLayout(embed_layout)
        layout.addWidget(embed_group)
        self._on_public_model_changed()

        # --- Source filters
        filter_group = QGroupBox("Source Filters")
        filter_layout = QVBoxLayout()

        types_row = QHBoxLayout()
        types_row.addWidget(QLabel("Include source types:"))
        self._type_checks: Dict[str, QCheckBox] = {}
        for src in _SOURCE_TYPES:
            cb = QCheckBox(src.replace("_", " "))
            types_row.addWidget(cb)
            self._type_checks[src] = cb
        filter_layout.addLayout(types_row)

        score_row = QHBoxLayout()
        score_row.addWidget(QLabel("Min relevance:"))
        self.min_relevance_spin = QDoubleSpinBox()
        self.min_relevance_spin.setRange(0.0, 1.0)
        self.min_relevance_spin.setSingleStep(0.05)
        self.min_relevance_spin.setValue(0.0)
        score_row.addWidget(self.min_relevance_spin)
        score_row.addWidget(QLabel("  Min confidence:"))
        self.min_confidence_spin = QDoubleSpinBox()
        self.min_confidence_spin.setRange(0.0, 1.0)
        self.min_confidence_spin.setSingleStep(0.05)
        self.min_confidence_spin.setValue(0.0)
        score_row.addWidget(self.min_confidence_spin)
        score_row.addStretch(1)
        filter_layout.addLayout(score_row)

        self.dedupe_check = QCheckBox("Deduplicate chunks")
        self.dedupe_check.setChecked(True)
        self.normalize_check = QCheckBox("Normalize text")
        self.normalize_check.setChecked(True)
        filter_layout.addWidget(self.dedupe_check)
        filter_layout.addWidget(self.normalize_check)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # --- Output
        out_group = QGroupBox("Output")
        out_layout = QFormLayout()
        self.output_dir_input = QLineEdit("indexes/public")
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._pick_output_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_dir_input)
        out_row.addWidget(out_btn)
        out_layout.addRow("Output directory:", out_row)
        out_group.setLayout(out_layout)
        layout.addWidget(out_group)

        # --- Actions
        action_row = QHBoxLayout()
        self.build_btn = QPushButton("Build Index")
        self.build_btn.clicked.connect(self._start_build)
        action_row.addWidget(self.build_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.log = _make_log_pane(160)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def _on_public_model_changed(self):
        from shared_utils.model_config import get_catalog_entry

        model_key = self.model_combo.currentData() or "mini"
        entry = get_catalog_entry(model_key) or {}
        hint = entry.get("description", "")
        dims = entry.get("dimensions")
        if dims:
            hint = f"{dims}d — {hint}" if hint else f"{dims}d"
        if entry.get("backend") == "openai":
            hint = f"{hint} (device ignored — API)"
            self.device_combo.setEnabled(False)
        else:
            self.device_combo.setEnabled(True)
        self.model_hint_label.setText(hint)

    def _pick_sources_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Sources Directory")
        if path:
            self.sources_dir_input.setText(path)

    def _pick_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_input.setText(path)

    def _selected_source_types(self):
        from moyo.publicside.gatherpublicsources.schema import SourceType
        picks = [s for s, cb in self._type_checks.items() if cb.isChecked()]
        return [SourceType(s) for s in picks]

    def _load_sources(self, sources_dir: Path):
        """Recursively load sources.json files into PublicSource objects."""
        from moyo.publicside.gatherpublicsources.schema import PublicSource

        sources = []
        if not sources_dir.exists():
            raise FileNotFoundError(f"Sources directory not found: {sources_dir}")

        files = list(sources_dir.rglob("sources.json"))
        if not files:
            # Treat top-level JSON files as raw arrays
            files = [p for p in sources_dir.rglob("*.json") if p.name != "summary.json"]

        for path in files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data = [data]
                for raw in data:
                    try:
                        sources.append(PublicSource(**raw))
                    except Exception:
                        continue
            except Exception:
                continue
        return sources

    def _start_build(self):
        if self._worker is not None:
            QMessageBox.information(self, "Busy", "A build is already running.")
            return

        sources_dir = Path(self.sources_dir_input.text().strip() or "data/public_sources")
        name = self.name_input.text().strip() or "public_index"
        description = self.description_input.text().strip()
        output_dir = self.output_dir_input.text().strip() or "indexes/public"

        try:
            from moyo.publicside.barrierprobe.schema import IndexConfig, IndexType
            from moyo.publicside.barrierprobe.public_index_builder import PublicIndexBuilder
            from shared_utils.model_config import resolve_model_name
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return

        model_key = self.model_combo.currentData() or "mini"
        model_name = resolve_model_name(model_key)
        device = self.device_combo.currentData() or "auto"

        cfg = IndexConfig(
            index_type=IndexType(self.index_type_combo.currentText()),
            embedding_model=model_name,
            embedding_device=device,
            chunk_size=self.chunk_size_spin.value(),
            chunk_overlap=self.chunk_overlap_spin.value(),
            min_chunk_length=self.min_len_spin.value(),
            max_chunk_length=self.max_len_spin.value(),
            output_directory=output_dir,
            source_types=self._selected_source_types(),
            min_relevance_score=self.min_relevance_spin.value(),
            min_confidence_score=self.min_confidence_spin.value(),
            deduplication_enabled=self.dedupe_check.isChecked(),
            normalization_enabled=self.normalize_check.isChecked(),
        )

        load_sources = self._load_sources

        def job():
            print(f"Loading sources from {sources_dir}…")
            sources = load_sources(sources_dir)
            print(f"Loaded {len(sources)} sources")
            if not sources:
                raise RuntimeError(
                    "No sources found. Run the Gather Public Sources tab first."
                )

            builder = PublicIndexBuilder(cfg)
            n = builder.add_sources(sources)
            print(f"Added {n} sources, {len(builder.chunks)} chunks")
            if cfg.normalization_enabled:
                changed = builder.normalize_chunks()
                print(f"Normalized {changed} chunks")
            if cfg.deduplication_enabled:
                removed = builder.deduplicate_chunks()
                print(f"Removed {removed} duplicate chunks")
            return builder.build_index(name=name, description=description)

        self.log.clear()
        self.log.append("Starting build…")
        self.progress_bar.setVisible(True)
        _busy(self.build_btn, True, "Build Index")

        self._worker = BackgroundWorker(job)
        self._worker.log.connect(self.log.append)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, result):
        try:
            if getattr(result, "success", False):
                self.log.append(
                    f"✅ Built index: {result.chunks_created} chunks, "
                    f"{result.vectors_created} vectors in "
                    f"{result.processing_time:.2f}s"
                )
                if result.index_path:
                    self.log.append(f"Index path: {result.index_path}")
            else:
                self.log.append(f"❌ Build failed: {getattr(result, 'message', '')}")
        finally:
            self._cleanup_worker()

    def _on_failed(self, msg: str):
        self.log.append(f"❌ {msg}")
        QMessageBox.critical(self, "Build failed", msg)
        self._cleanup_worker()

    def _cleanup_worker(self):
        self._worker = None
        self.progress_bar.setVisible(False)
        _busy(self.build_btn, False, "Build Index")


class BarrierProbeTab(QWidget):
    """Tab for running barrier analysis between public and private indices."""

    def __init__(self):
        super().__init__()
        self._worker: Optional[BackgroundWorker] = None
        self._last_result = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Barrier Probe — Public vs Private Distance Analysis")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        desc = QLabel(
            "For each private phrase, find its nearest public neighbour and rank "
            "potential information-barrier breaches by cosine distance. "
            "Both indices must share the same chunking + embedding model."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # --- Index paths
        idx_group = QGroupBox("Indices")
        idx_layout = QFormLayout()
        self.public_path = QLineEdit("indexes/public")
        pub_btn = QPushButton("Browse…")
        pub_btn.clicked.connect(lambda: self._pick(self.public_path, "Public Index"))
        pub_row = QHBoxLayout()
        pub_row.addWidget(self.public_path)
        pub_row.addWidget(pub_btn)
        idx_layout.addRow("Public index:", pub_row)

        self.private_path = QLineEdit("indexes/private")
        priv_btn = QPushButton("Browse…")
        priv_btn.clicked.connect(lambda: self._pick(self.private_path, "Private Index"))
        priv_row = QHBoxLayout()
        priv_row.addWidget(self.private_path)
        priv_row.addWidget(priv_btn)
        idx_layout.addRow("Private index:", priv_row)
        idx_group.setLayout(idx_layout)
        layout.addWidget(idx_group)

        # --- Analysis params
        param_group = QGroupBox("Analysis Parameters")
        param_layout = QFormLayout()

        self.similarity_spin = QDoubleSpinBox()
        self.similarity_spin.setRange(0.0, 2.0)
        self.similarity_spin.setSingleStep(0.05)
        self.similarity_spin.setValue(0.8)
        param_layout.addRow("Cosine-distance threshold:", self.similarity_spin)

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 5000)
        self.top_k_spin.setValue(10)
        param_layout.addRow("Top K breaches:", self.top_k_spin)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # --- Actions
        action_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Barrier Analysis")
        self.run_btn.clicked.connect(self._start)
        action_row.addWidget(self.run_btn)

        self.save_json_btn = QPushButton("Save JSON Report")
        self.save_json_btn.setEnabled(False)
        self.save_json_btn.clicked.connect(self._save_json)
        action_row.addWidget(self.save_json_btn)

        self.save_html_btn = QPushButton("Save HTML Report")
        self.save_html_btn.setEnabled(False)
        self.save_html_btn.clicked.connect(self._save_html)
        action_row.addWidget(self.save_html_btn)

        action_row.addStretch(1)
        layout.addLayout(action_row)

        # --- Risk summary
        self.summary_label = QLabel("No analysis run yet.")
        self.summary_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.summary_label)

        # --- Breach table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Rank", "Risk", "Distance", "Private phrase", "Public phrase"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setSortingEnabled(True)
        self.table.setMinimumHeight(220)
        layout.addWidget(self.table)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.log = _make_log_pane(120)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def _pick(self, line_edit: QLineEdit, label: str):
        path = QFileDialog.getExistingDirectory(self, f"Select {label}")
        if path:
            line_edit.setText(path)

    def _start(self):
        if self._worker is not None:
            QMessageBox.information(self, "Busy", "Analysis already running.")
            return

        public_path = self.public_path.text().strip()
        private_path = self.private_path.text().strip()
        if not (Path(public_path).exists() and Path(private_path).exists()):
            QMessageBox.warning(
                self, "Path missing",
                "Both the public and private index directories must exist."
            )
            return

        try:
            from moyo.publicside.barrierprobe.schema import BarrierProbeConfig
            from moyo.publicside.barrierprobe.barrier_analyzer import BarrierAnalyzer
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return

        cfg = BarrierProbeConfig(
            public_index_path=public_path,
            private_index_path=private_path,
            similarity_threshold=self.similarity_spin.value(),
        )
        top_k = self.top_k_spin.value()

        def job():
            analyzer = BarrierAnalyzer(cfg)
            return analyzer.analyze_barriers(top_k=top_k)

        self.log.clear()
        self.log.append("Running barrier analysis…")
        self.progress_bar.setVisible(True)
        _busy(self.run_btn, True, "Run Barrier Analysis")
        self.save_json_btn.setEnabled(False)
        self.save_html_btn.setEnabled(False)

        self._worker = BackgroundWorker(job)
        self._worker.log.connect(self.log.append)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, result):
        try:
            self._last_result = result
            high = getattr(result, "high_risk_breaches", 0)
            medium = getattr(result, "medium_risk_breaches", 0)
            low = getattr(result, "low_risk_breaches", 0)
            total = getattr(result, "breach_count", 0)
            self.summary_label.setText(
                f"Breaches: {total}   High: {high}   Medium: {medium}   Low: {low}"
            )

            breaches = getattr(result, "potential_breaches", []) or []
            self.table.setSortingEnabled(False)
            self.table.setRowCount(len(breaches))
            for row, b in enumerate(breaches):
                rank = b.get("rank", row + 1)
                risk = b.get("risk_level", "")
                dist = b.get("distance", 0.0)
                pub = (b.get("public_content") or "")[:200]
                priv = (b.get("private_content") or "")[:200]
                self._set(row, 0, str(rank))
                self._set(row, 1, risk, _risk_color(risk))
                self._set(row, 2, f"{dist:.4f}")
                self._set(row, 3, priv)
                self._set(row, 4, pub)
            self.table.setSortingEnabled(True)

            for rec in getattr(result, "recommendations", []) or []:
                self.log.append(f"• {rec}")

            self.save_json_btn.setEnabled(True)
            self.save_html_btn.setEnabled(True)
        finally:
            self._cleanup_worker()

    def _set(self, row, col, text, color=None):
        item = QTableWidgetItem(text)
        if color is not None:
            item.setBackground(color)
        self.table.setItem(row, col, item)

    def _on_failed(self, msg: str):
        self.log.append(f"❌ {msg}")
        QMessageBox.critical(self, "Analysis failed", msg)
        self._cleanup_worker()

    def _cleanup_worker(self):
        self._worker = None
        self.progress_bar.setVisible(False)
        _busy(self.run_btn, False, "Run Barrier Analysis")

    def _save_json(self):
        if self._last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON Report", "barrier_report.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            data = self._last_result.dict() if hasattr(self._last_result, "dict") else dict(self._last_result)
            Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            self.log.append(f"Saved JSON report to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _save_html(self):
        if self._last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HTML Report", "barrier_report.html", "HTML (*.html)"
        )
        if not path:
            return
        try:
            res = self._last_result
            lines = [
                "<html><head><meta charset='utf-8'><title>Barrier Probe Report</title>",
                "<style>body{font-family:sans-serif;}th,td{padding:4px 8px;border:1px solid #ccc;}"
                "tr.high{background:#fdd;}tr.medium{background:#fed;}tr.low{background:#dfd;}</style>",
                "</head><body>",
                f"<h1>Barrier Probe Report</h1>",
                f"<p>Total breaches: {res.breach_count} "
                f"(high {res.high_risk_breaches}, "
                f"medium {res.medium_risk_breaches}, "
                f"low {res.low_risk_breaches})</p>",
                "<table><tr><th>Rank</th><th>Risk</th><th>Distance</th>"
                "<th>Private</th><th>Public</th></tr>",
            ]
            for b in res.potential_breaches:
                lines.append(
                    f"<tr class='{b.get('risk_level','')}'>"
                    f"<td>{b.get('rank','')}</td>"
                    f"<td>{b.get('risk_level','')}</td>"
                    f"<td>{b.get('distance', 0):.4f}</td>"
                    f"<td>{(b.get('private_content') or '')[:400]}</td>"
                    f"<td>{(b.get('public_content') or '')[:400]}</td>"
                    "</tr>"
                )
            lines.append("</table></body></html>")
            Path(path).write_text("\n".join(lines), encoding="utf-8")
            self.log.append(f"Saved HTML report to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))


def _risk_color(risk: str) -> QColor:
    if risk == "high":
        return QColor(255, 200, 200)
    if risk == "medium":
        return QColor(255, 230, 180)
    if risk == "low":
        return QColor(210, 240, 210)
    return QColor(255, 255, 255)


class FuzzerTab(QWidget):
    """LLM-assisted fuzzer for semantic barrier probing."""

    def __init__(self):
        super().__init__()
        self._worker: Optional[BackgroundWorker] = None
        self._last_results: list = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("LLM Fuzzer")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        desc = QLabel(
            "Use an LLM (default: local Ollama llama3.1:8b) to iteratively transform input "
            "phrases toward a target concept. Mode basic paraphrases only; full rotates "
            "translate, abstract, summarize, and typo. Foreign-language outputs are "
            "translated to English with a language annotation in saved reports."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # --- Corpus
        corpus_group = QGroupBox("Corpus")
        corpus_layout = QFormLayout()
        self.corpus_path = QLineEdit("indexes/private")
        corp_btn = QPushButton("Browse…")
        corp_btn.clicked.connect(lambda: self._pick(self.corpus_path, "Corpus Index"))
        corp_row = QHBoxLayout()
        corp_row.addWidget(self.corpus_path)
        corp_row.addWidget(corp_btn)
        corpus_layout.addRow("Corpus FAISS index:", corp_row)
        corpus_group.setLayout(corpus_layout)
        layout.addWidget(corpus_group)

        # --- Inputs
        input_group = QGroupBox("Phrases & Target")
        input_layout = QFormLayout()
        self.target_input = QLineEdit()
        self.target_input.setPlaceholderText("Target concept, e.g. 'confidential information'")
        input_layout.addRow("Target concept:", self.target_input)

        self.phrases_input = QPlainTextEdit()
        self.phrases_input.setPlaceholderText("One phrase per line, e.g.\ndata breach\nsecurity incident")
        self.phrases_input.setMinimumHeight(90)
        input_layout.addRow("Phrases:", self.phrases_input)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # --- LLM
        llm_group = QGroupBox("LLM Configuration")
        llm_layout = QFormLayout()
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["ollama", "openai", "anthropic", "custom", "local"])
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        llm_layout.addRow("Provider:", self.provider_combo)

        self.model_input = QLineEdit("llama3.1:8b")
        llm_layout.addRow("Model:", self.model_input)

        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Leave blank to use OPENAI_API_KEY / ANTHROPIC_API_KEY")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        llm_layout.addRow("API key:", self.api_key_input)

        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText(
            "Ollama http://localhost:11434  |  custom OpenAI-compatible http://localhost:8000/v1"
        )
        self.base_url_input.setText("http://localhost:11434")
        self.base_url_input.setEnabled(True)
        llm_layout.addRow("Base URL:", self.base_url_input)
        llm_group.setLayout(llm_layout)
        layout.addWidget(llm_group)

        # --- Fuzzing
        fuzz_group = QGroupBox("Fuzzing Parameters")
        fuzz_layout = QFormLayout()
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 100)
        self.max_iter_spin.setValue(5)
        fuzz_layout.addRow("Max iterations:", self.max_iter_spin)

        self.target_sim_spin = QDoubleSpinBox()
        self.target_sim_spin.setRange(0.0, 1.0)
        self.target_sim_spin.setSingleStep(0.05)
        self.target_sim_spin.setValue(0.95)
        fuzz_layout.addRow("Target similarity:", self.target_sim_spin)

        self.search_k_spin = QSpinBox()
        self.search_k_spin.setRange(1, 1000)
        self.search_k_spin.setValue(10)
        fuzz_layout.addRow("Search K (neighbours):", self.search_k_spin)

        self.sim_threshold_spin = QDoubleSpinBox()
        self.sim_threshold_spin.setRange(0.0, 1.0)
        self.sim_threshold_spin.setSingleStep(0.05)
        self.sim_threshold_spin.setValue(0.8)
        fuzz_layout.addRow("Similarity threshold:", self.sim_threshold_spin)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setValue(0.7)
        fuzz_layout.addRow("Temperature:", self.temperature_spin)

        self.fuzz_mode_combo = QComboBox()
        self.fuzz_mode_combo.addItem("basic (paraphrase only)", "basic")
        self.fuzz_mode_combo.addItem(
            "full (translate / abstract / summarize / typo)", "full"
        )
        fuzz_layout.addRow("Fuzz mode:", self.fuzz_mode_combo)

        fuzz_group.setLayout(fuzz_layout)
        layout.addWidget(fuzz_group)

        # --- Actions
        action_row = QHBoxLayout()
        self.test_btn = QPushButton("Test LLM Connection")
        self.test_btn.clicked.connect(self._test_llm)
        action_row.addWidget(self.test_btn)

        self.fuzz_btn = QPushButton("Run Fuzzer")
        self.fuzz_btn.clicked.connect(self._run_fuzzer)
        action_row.addWidget(self.fuzz_btn)

        self.save_btn = QPushButton("Save Results JSON")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_results)
        action_row.addWidget(self.save_btn)

        action_row.addStretch(1)
        layout.addLayout(action_row)

        # --- Results
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Original", "Fuzzed", "Final similarity", "Iterations"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(180)
        layout.addWidget(self.table)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.log = _make_log_pane(120)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def _pick(self, line_edit: QLineEdit, label: str):
        path = QFileDialog.getExistingDirectory(self, f"Select {label}")
        if path:
            line_edit.setText(path)

    def _on_provider_changed(self, provider: str):
        suggested = {
            "local": "all-MiniLM-L6-v2",
            "ollama": "llama3.1:8b",
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-6",
            "custom": "",
        }
        self.model_input.setText(suggested.get(provider, ""))
        # Ollama and custom OpenAI-compatible endpoints need a base URL.
        self.base_url_input.setEnabled(provider in ("ollama", "custom"))
        # Hosted providers and custom endpoints may need a key (custom is
        # optional; self-hosted servers ignore it).
        self.api_key_input.setEnabled(provider in ("openai", "anthropic", "custom"))

    def _resolve_api_key(self) -> Optional[str]:
        key = self.api_key_input.text().strip()
        if key:
            return key
        provider = self.provider_combo.currentText()
        import os
        if provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        if provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        # 'custom' endpoints often need no key; leave blank if none supplied.
        return None

    def _make_config(self):
        from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzerConfig
        base_url = self.base_url_input.text().strip() or None
        return LLMFuzzerConfig(
            llm_provider=self.provider_combo.currentText(),
            model_name=self.model_input.text().strip(),
            api_key=self._resolve_api_key(),
            base_url=base_url,
            temperature=self.temperature_spin.value(),
            search_k=self.search_k_spin.value(),
            similarity_threshold=self.sim_threshold_spin.value(),
            max_iterations=self.max_iter_spin.value(),
            target_similarity=self.target_sim_spin.value(),
            fuzz_mode=self.fuzz_mode_combo.currentData() or "basic",
        )

    def _test_llm(self):
        from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer
        cfg = self._make_config()
        self.log.append(
            f"Testing LLM ({cfg.llm_provider} / {cfg.model_name})…"
        )
        try:
            fuzzer = LLMFuzzer(cfg)
            if not fuzzer.llm_client:
                self.log.append("❌ LLM client not initialised (missing dependency or API key)")
                return
            response = fuzzer.query_llm(
                "Please respond with 'LLM test successful' if you can read this message."
            )
            if response:
                self.log.append(f"✅ {response.strip()[:200]}")
            else:
                self.log.append("❌ LLM returned no response (check API key / model name)")
        except Exception as exc:
            self.log.append(f"❌ {exc}")

    def _run_fuzzer(self):
        if self._worker is not None:
            QMessageBox.information(self, "Busy", "Fuzzer is already running.")
            return

        target = self.target_input.text().strip()
        if not target:
            QMessageBox.warning(self, "Missing input", "Enter a target concept.")
            return
        phrases = [p.strip() for p in self.phrases_input.toPlainText().splitlines() if p.strip()]
        if not phrases:
            QMessageBox.warning(self, "Missing input", "Enter at least one phrase.")
            return
        corpus = Path(self.corpus_path.text().strip())
        if not corpus.exists():
            QMessageBox.warning(self, "Path missing", f"Corpus not found: {corpus}")
            return

        try:
            from shared_utils import FAISSIndex
            from moyo.publicside.barrierprobe.llm_fuzzer import fuzz_phrases_for_barrier_analysis
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return

        cfg = self._make_config()

        def job():
            print(f"Loading corpus index from {corpus}…")
            index = FAISSIndex.load(str(corpus))
            print(f"Loaded {index.get_vector_count()} vectors")
            print(f"Fuzzing {len(phrases)} phrases toward '{target}'…")
            return fuzz_phrases_for_barrier_analysis(phrases, target, index, cfg)

        self.log.clear()
        self.log.append(f"Starting fuzzer with {len(phrases)} phrases…")
        self.progress_bar.setVisible(True)
        _busy(self.fuzz_btn, True, "Run Fuzzer")

        self._worker = BackgroundWorker(job)
        self._worker.log.connect(self.log.append)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, results):
        try:
            self._last_results = results or []
            self.table.setRowCount(len(self._last_results))
            for row, r in enumerate(self._last_results):
                self.table.setItem(row, 0, QTableWidgetItem(str(r.get("original_phrase", ""))))
                fuzzed = r.get("fuzzed_phrase_for_report") or r.get("fuzzed_phrase", "")
                self.table.setItem(row, 1, QTableWidgetItem(str(fuzzed)))
                self.table.setItem(row, 2, QTableWidgetItem(f"{r.get('final_similarity', 0.0):.3f}"))
                self.table.setItem(row, 3, QTableWidgetItem(str(r.get("iterations", 0))))
            if self._last_results:
                avg_sim = sum(r.get("final_similarity", 0.0) for r in self._last_results) / len(self._last_results)
                mode = self._last_results[0].get("fuzz_mode", "basic")
                self.log.append(
                    f"✅ Done (mode={mode}). Average final similarity: {avg_sim:.3f}"
                )
                self.save_btn.setEnabled(True)
            else:
                self.log.append("⚠️  Fuzzer returned no results.")
        finally:
            self._cleanup_worker()

    def _on_failed(self, msg: str):
        self.log.append(f"❌ {msg}")
        QMessageBox.critical(self, "Fuzzer failed", msg)
        self._cleanup_worker()

    def _cleanup_worker(self):
        self._worker = None
        self.progress_bar.setVisible(False)
        _busy(self.fuzz_btn, False, "Run Fuzzer")

    def _save_results(self):
        if not self._last_results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Fuzzer Results", "fuzzer_results.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self._last_results, indent=2, default=str),
                encoding="utf-8",
            )
            self.log.append(f"Saved results to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))


class VisualizationTab(QWidget):
    """Tab for 2D FAISS index visualization with multiple chart types."""

    REDUCERS = ["MDS", "PCA", "t-SNE", "UMAP"]
    PLOT_TYPES = [
        "Scatter (2D projection)",
        "Distance histogram",
        "Nearest-neighbor CDF",
        "Density contours (KDE)",
        "Cross-distance heatmap",
        "Pairwise distance matrix",
    ]

    def __init__(self):
        super().__init__()
        self.default_private_index_path = _repo_root / "indexes" / "private"
        self.default_public_index_path = _repo_root / "indexes" / "public"
        self.private_index = None
        self.public_index = None
        # Cached arrays: avoid recomputing on every plot change
        self._private_vectors = None
        self._public_vectors = None
        self._distance_matrix = None
        self._coords_2d = None
        self._coords_reducer = None  # which reducer the coords came from
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("FAISS Index Visualization")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        # --- Index selection -------------------------------------------------
        data_group = QGroupBox("Load FAISS Indices")
        data_layout = QVBoxLayout()

        private_layout = QHBoxLayout()
        self.private_index_label = QLabel(str(self.default_private_index_path))
        select_private_btn = QPushButton("Load Private Index")
        select_private_btn.clicked.connect(self.load_private_index)
        private_layout.addWidget(QLabel("Private:"))
        private_layout.addWidget(self.private_index_label, stretch=1)
        private_layout.addWidget(select_private_btn)
        data_layout.addLayout(private_layout)

        self.use_default_private_checkbox = QCheckBox(
            "Use default private index (indexes/private)"
        )
        self.use_default_private_checkbox.setChecked(True)
        self.use_default_private_checkbox.toggled.connect(self.on_private_method_changed)
        data_layout.addWidget(self.use_default_private_checkbox)

        public_layout = QHBoxLayout()
        self.public_index_label = QLabel(str(self.default_public_index_path))
        select_public_btn = QPushButton("Load Public Index")
        select_public_btn.clicked.connect(self.load_public_index)
        public_layout.addWidget(QLabel("Public:"))
        public_layout.addWidget(self.public_index_label, stretch=1)
        public_layout.addWidget(select_public_btn)
        data_layout.addLayout(public_layout)

        self.use_default_public_checkbox = QCheckBox(
            "Use default public index (indexes/public)"
        )
        self.use_default_public_checkbox.setChecked(True)
        self.use_default_public_checkbox.toggled.connect(self.on_public_method_changed)
        data_layout.addWidget(self.use_default_public_checkbox)

        self.load_btn = QPushButton("Load Indices")
        self.load_btn.clicked.connect(self.load_indices)
        data_layout.addWidget(self.load_btn)

        self.status_label = QLabel("No indices loaded yet.")
        self.status_label.setStyleSheet("color: gray;")
        data_layout.addWidget(self.status_label)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # --- Visualization controls -----------------------------------------
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(self.PLOT_TYPES)
        viz_layout.addRow("Plot type:", self.plot_type_combo)

        self.reducer_combo = QComboBox()
        self.reducer_combo.addItems(self.REDUCERS)
        viz_layout.addRow("Dim. reduction (scatter/contours):", self.reducer_combo)

        self.viz_random_state_input = QLineEdit("42")
        viz_layout.addRow("Random state:", self.viz_random_state_input)

        # Scatter overlay options
        self.show_distance_lines_checkbox = QCheckBox(
            "Overlay nearest-neighbour lines (scatter only)"
        )
        self.show_distance_lines_checkbox.setChecked(True)
        viz_layout.addRow(self.show_distance_lines_checkbox)

        self.distance_threshold_input = QLineEdit("0.5")
        viz_layout.addRow("Line distance threshold:", self.distance_threshold_input)

        self.show_clustering_checkbox = QCheckBox(
            "Colour by KMeans cluster (scatter only)"
        )
        self.show_clustering_checkbox.setChecked(False)
        viz_layout.addRow(self.show_clustering_checkbox)

        self.num_clusters_input = QLineEdit("5")
        viz_layout.addRow("Number of clusters:", self.num_clusters_input)

        self.bins_input = QLineEdit("40")
        viz_layout.addRow("Histogram/heatmap bins:", self.bins_input)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # --- Actions --------------------------------------------------------
        action_row = QHBoxLayout()
        self.viz_btn = QPushButton("Generate Plot")
        self.viz_btn.setEnabled(False)
        self.viz_btn.clicked.connect(self.generate_visualization)
        action_row.addWidget(self.viz_btn)

        self.viz_export_btn = QPushButton("Export Current Plot")
        self.viz_export_btn.setEnabled(False)
        self.viz_export_btn.clicked.connect(self.export_visualization)
        action_row.addWidget(self.viz_export_btn)

        action_row.addStretch(1)
        layout.addLayout(action_row)

        # --- Canvas ---------------------------------------------------------
        if MATPLOTLIB_AVAILABLE:
            viz_canvas_group = QGroupBox("Plot")
            viz_canvas_layout = QVBoxLayout()

            self.figure = Figure(figsize=(10, 7))
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumHeight(500)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            viz_canvas_layout.addWidget(self.canvas)
            viz_canvas_group.setLayout(viz_canvas_layout)
            layout.addWidget(viz_canvas_group, stretch=1)
        else:
            no_viz_label = QLabel(
                "Visualization dependencies not installed. "
                "Run: pip install moyo[gui]"
            )
            no_viz_label.setStyleSheet("color: red;")
            layout.addWidget(no_viz_label)

        self.log = _make_log_pane(120)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def on_private_method_changed(self):
        self.private_index_label.setEnabled(not self.use_default_private_checkbox.isChecked())

    def on_public_method_changed(self):
        self.public_index_label.setEnabled(not self.use_default_public_checkbox.isChecked())

    def load_private_index(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Load Private FAISS Index")
        if folder_path:
            self.private_index_label.setText(folder_path)
            self.use_default_private_checkbox.setChecked(False)

    def load_public_index(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Load Public FAISS Index")
        if folder_path:
            self.public_index_label.setText(folder_path)
            self.use_default_public_checkbox.setChecked(False)

    # -----------------------------------------------------------------------
    # Loading + caching
    # -----------------------------------------------------------------------

    def _resolve_paths(self):
        priv = (
            self.default_private_index_path
            if self.use_default_private_checkbox.isChecked()
            else Path(self.private_index_label.text())
        )
        pub = (
            self.default_public_index_path
            if self.use_default_public_checkbox.isChecked()
            else Path(self.public_index_label.text())
        )
        return priv, pub

    def load_indices(self):
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Visualization deps not installed")
            return

        priv_path, pub_path = self._resolve_paths()
        if not priv_path.exists():
            QMessageBox.warning(self, "Path missing", f"Private index not found: {priv_path}")
            return
        if not pub_path.exists():
            QMessageBox.warning(self, "Path missing", f"Public index not found: {pub_path}")
            return

        try:
            from shared_utils.faiss_index import FAISSIndex
        except ImportError as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return

        try:
            self.log.append(f"Loading private index from {priv_path}…")
            self.private_index = FAISSIndex.load(priv_path)
            self._private_vectors = self.private_index.index.reconstruct_n(
                0, self.private_index.get_vector_count()
            )

            self.log.append(f"Loading public index from {pub_path}…")
            self.public_index = FAISSIndex.load(pub_path)
            self._public_vectors = self.public_index.index.reconstruct_n(
                0, self.public_index.get_vector_count()
            )

            n_priv = len(self._private_vectors)
            n_pub = len(self._public_vectors)
            self.log.append(f"Loaded {n_priv} private + {n_pub} public vectors")

            from sklearn.metrics.pairwise import cosine_distances
            all_vectors = np.vstack([self._private_vectors, self._public_vectors])
            self._distance_matrix = cosine_distances(all_vectors)
            self._coords_2d = None
            self._coords_reducer = None

            self.status_label.setText(
                f"Indices loaded: {n_priv} private, {n_pub} public "
                f"(dim {self._private_vectors.shape[1]})"
            )
            self.status_label.setStyleSheet("color: green;")
            self.viz_btn.setEnabled(True)
        except Exception as exc:
            import traceback
            self.log.append(traceback.format_exc())
            QMessageBox.critical(self, "Load failed", str(exc))

    def _reduce_dimensions(self):
        """Project the joint distance matrix to 2D, cached by reducer choice."""
        reducer_name = self.reducer_combo.currentText()
        if self._coords_2d is not None and self._coords_reducer == reducer_name:
            return self._coords_2d

        random_state = int(self.viz_random_state_input.text() or "42")
        all_vectors = np.vstack([self._private_vectors, self._public_vectors])

        self.log.append(f"Projecting with {reducer_name}…")
        if reducer_name == "MDS":
            reducer = MDS(
                n_components=2, dissimilarity="precomputed", random_state=random_state,
                normalized_stress="auto",
            )
            coords = reducer.fit_transform(self._distance_matrix)
        elif reducer_name == "PCA":
            from sklearn.decomposition import PCA
            coords = PCA(n_components=2, random_state=random_state).fit_transform(all_vectors)
        elif reducer_name == "t-SNE":
            from sklearn.manifold import TSNE
            perplexity = max(5.0, min(30.0, (len(all_vectors) - 1) / 3.0))
            coords = TSNE(
                n_components=2, metric="precomputed", init="random",
                perplexity=perplexity, random_state=random_state,
            ).fit_transform(self._distance_matrix)
        elif reducer_name == "UMAP":
            try:
                import umap  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "UMAP not installed. Run: pip install umap-learn"
                ) from exc
            coords = umap.UMAP(
                n_components=2, metric="precomputed", random_state=random_state,
            ).fit_transform(self._distance_matrix)
        else:
            raise ValueError(f"Unknown reducer: {reducer_name}")

        self._coords_2d = coords
        self._coords_reducer = reducer_name
        return coords

    # -----------------------------------------------------------------------
    # Plot dispatch
    # -----------------------------------------------------------------------

    def generate_visualization(self):
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Matplotlib not available")
            return
        if self._private_vectors is None or self._public_vectors is None:
            QMessageBox.warning(self, "No data", "Click 'Load Indices' first.")
            return

        plot_type = self.plot_type_combo.currentText()
        self.figure.clear()
        try:
            if plot_type == self.PLOT_TYPES[0]:        # Scatter
                self._plot_scatter()
            elif plot_type == self.PLOT_TYPES[1]:      # Distance histogram
                self._plot_histogram()
            elif plot_type == self.PLOT_TYPES[2]:      # CDF
                self._plot_cdf()
            elif plot_type == self.PLOT_TYPES[3]:      # Density contours
                self._plot_density_contours()
            elif plot_type == self.PLOT_TYPES[4]:      # Cross-distance heatmap
                self._plot_cross_heatmap()
            elif plot_type == self.PLOT_TYPES[5]:      # Pairwise distance matrix
                self._plot_distance_matrix()
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
            self.figure.tight_layout()
            self.canvas.draw()
            self.viz_export_btn.setEnabled(True)
            self.log.append(f"Rendered {plot_type}")
        except Exception as exc:
            import traceback
            self.log.append(traceback.format_exc())
            QMessageBox.critical(self, "Plot failed", str(exc))

    # ---- Individual plotters ----------------------------------------------

    def _cross_distances(self):
        n_priv = len(self._private_vectors)
        return self._distance_matrix[:n_priv, n_priv:]

    def _plot_scatter(self):
        coords = self._reduce_dimensions()
        n_priv = len(self._private_vectors)
        private_coords = coords[:n_priv]
        public_coords = coords[n_priv:]
        ax = self.figure.add_subplot(111)

        # Optional clustering
        if self.show_clustering_checkbox.isChecked():
            try:
                from sklearn.cluster import KMeans
                cross = self._cross_distances()
                priv_avg = np.mean(cross, axis=1)
                pub_avg = np.mean(cross, axis=0)
                feats = np.concatenate([priv_avg, pub_avg]).reshape(-1, 1)
                n_clusters = max(2, int(self.num_clusters_input.text() or "5"))
                labels = KMeans(n_clusters=n_clusters, n_init="auto",
                                random_state=42).fit_predict(feats)
                priv_labels = labels[:n_priv]
                pub_labels = labels[n_priv:]
                colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
                for k in range(n_clusters):
                    mask = priv_labels == k
                    if np.any(mask):
                        ax.scatter(private_coords[mask, 0], private_coords[mask, 1],
                                   c=[colors[k]], marker="o", s=50, alpha=0.7,
                                   label=f"Private c{k+1}")
                    mask = pub_labels == k
                    if np.any(mask):
                        ax.scatter(public_coords[mask, 0], public_coords[mask, 1],
                                   c=[colors[k]], marker="s", s=50, alpha=0.7,
                                   label=f"Public c{k+1}")
            except Exception as exc:
                self.log.append(f"Clustering skipped: {exc}")
                self._plot_scatter_plain(ax, private_coords, public_coords)
        else:
            self._plot_scatter_plain(ax, private_coords, public_coords)

        # Optional nearest-neighbour lines
        if self.show_distance_lines_checkbox.isChecked():
            try:
                threshold = float(self.distance_threshold_input.text() or "0.5")
                cross = self._cross_distances()
                for i, p in enumerate(private_coords):
                    j = int(np.argmin(cross[i]))
                    d = float(cross[i, j])
                    if d <= threshold:
                        q = public_coords[j]
                        ax.plot([p[0], q[0]], [p[1], q[1]],
                                "g-", alpha=max(0.2, 1.0 - d * 2.0),
                                linewidth=max(0.5, 3.0 - d * 5.0))
            except Exception as exc:
                self.log.append(f"Lines skipped: {exc}")

        ax.set_title(f"2D Scatter ({self._coords_reducer})")
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    def _plot_scatter_plain(self, ax, private_coords, public_coords):
        if len(private_coords) > 0:
            ax.scatter(private_coords[:, 0], private_coords[:, 1],
                       c="tab:blue", marker="o", s=50, alpha=0.7, label="Private")
        if len(public_coords) > 0:
            ax.scatter(public_coords[:, 0], public_coords[:, 1],
                       c="tab:red", marker="s", s=50, alpha=0.7, label="Public")

    def _plot_histogram(self):
        cross = self._cross_distances()
        bins = max(5, int(self.bins_input.text() or "40"))
        ax = self.figure.add_subplot(111)
        nn = cross.min(axis=1)
        ax.hist(cross.ravel(), bins=bins, alpha=0.4,
                label="All public×private", color="tab:gray")
        ax.hist(nn, bins=bins, alpha=0.8,
                label="Per-private nearest", color="tab:orange")
        ax.axvline(nn.mean(), linestyle="--", color="black",
                   label=f"mean NN = {nn.mean():.3f}")
        ax.set_xlabel("Cosine distance (lower = more similar)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distance distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_cdf(self):
        cross = self._cross_distances()
        nn = np.sort(cross.min(axis=1))
        ax = self.figure.add_subplot(111)
        ax.plot(nn, np.linspace(0, 1, len(nn)), color="tab:orange",
                label="Per-private nearest CDF")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
        ax.set_xlabel("Cosine distance")
        ax.set_ylabel("Fraction of private chunks ≤ x")
        ax.set_title("Nearest-neighbour CDF (leak risk curve)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_density_contours(self):
        coords = self._reduce_dimensions()
        n_priv = len(self._private_vectors)
        priv = coords[:n_priv]
        pub = coords[n_priv:]
        ax = self.figure.add_subplot(111)

        try:
            from scipy.stats import gaussian_kde
            self._draw_kde_contours(ax, priv, "tab:blue", "Private")
            self._draw_kde_contours(ax, pub, "tab:red", "Public")
        except ImportError:
            self.log.append("scipy not available — falling back to scatter")
            self._plot_scatter_plain(ax, priv, pub)

        ax.scatter(priv[:, 0], priv[:, 1], c="tab:blue", marker="o", s=18, alpha=0.5)
        ax.scatter(pub[:, 0], pub[:, 1], c="tab:red", marker="s", s=18, alpha=0.5)
        ax.set_title(f"Density contours ({self._coords_reducer})")
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _draw_kde_contours(self, ax, points, color, label):
        from scipy.stats import gaussian_kde
        if len(points) < 3:
            return
        kde = gaussian_kde(points.T)
        x = np.linspace(points[:, 0].min(), points[:, 0].max(), 120)
        y = np.linspace(points[:, 1].min(), points[:, 1].max(), 120)
        xx, yy = np.meshgrid(x, y)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        cs = ax.contour(xx, yy, zz, levels=5, colors=color, linewidths=1.2, alpha=0.8)
        if cs.collections:
            cs.collections[0].set_label(label)

    def _plot_cross_heatmap(self):
        coords = self._reduce_dimensions()
        n_priv = len(self._private_vectors)
        priv = coords[:n_priv]
        pub = coords[n_priv:]
        cross = self._cross_distances()
        bins = max(10, int(self.bins_input.text() or "40"))
        ax = self.figure.add_subplot(111)

        xs, ys, ds = [], [], []
        for i, p in enumerate(priv):
            for j, q in enumerate(pub):
                xs.append((p[0] + q[0]) / 2.0)
                ys.append((p[1] + q[1]) / 2.0)
                ds.append(cross[i, j])
        xs, ys, ds = map(np.array, (xs, ys, ds))

        x_bins = np.linspace(xs.min(), xs.max(), bins)
        y_bins = np.linspace(ys.min(), ys.max(), bins)
        h, xe, ye = np.histogram2d(xs, ys, bins=[x_bins, y_bins], weights=ds)
        c, _, _ = np.histogram2d(xs, ys, bins=[x_bins, y_bins])
        h = np.divide(h, c, out=np.zeros_like(h), where=c > 0)
        im = ax.imshow(h.T, origin="lower",
                       extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       cmap="hot", aspect="auto")
        cbar = self.figure.colorbar(im, ax=ax)
        cbar.set_label("Average cross distance")
        ax.set_title(f"Cross-distance heatmap ({self._coords_reducer})")
        ax.set_xticklabels([]); ax.set_yticklabels([])

    def _plot_distance_matrix(self):
        n_priv = len(self._private_vectors)
        # Show ordered: private first, then public; vmin/vmax for colour scale
        ax = self.figure.add_subplot(111)
        im = ax.imshow(self._distance_matrix, cmap="viridis", aspect="auto",
                       vmin=0.0, vmax=min(1.0, float(self._distance_matrix.max())))
        ax.axhline(n_priv - 0.5, color="white", linewidth=1)
        ax.axvline(n_priv - 0.5, color="white", linewidth=1)
        ax.set_title("Pairwise cosine distance (private | public)")
        ax.set_xlabel("Chunk index")
        ax.set_ylabel("Chunk index")
        cbar = self.figure.colorbar(im, ax=ax)
        cbar.set_label("Cosine distance")

    # -----------------------------------------------------------------------

    def export_visualization(self):
        if not hasattr(self, "figure"):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "moyo_plot.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=300, bbox_inches="tight")
            self.log.append(f"Saved plot to {path}")
            QMessageBox.information(self, "Success", f"Saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))


class MoyoGUI(QMainWindow):
    """Main moyo GUI application."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("moyo GUI")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(DataInputTab(), "Private Data Input")
        self.tab_widget.addTab(FAISSIndexTab(), "Create Private Index")
        self.tab_widget.addTab(GatherPublicSourcesTab(), "Gather Public Sources")
        self.tab_widget.addTab(BuildPublicCorpusTab(), "Build Public Corpus")
        self.tab_widget.addTab(BarrierProbeTab(), "Barrier Probe")
        self.tab_widget.addTab(FuzzerTab(), "LLM Fuzzer")
        self.tab_widget.addTab(VisualizationTab(), "Visualize Indices")

        layout.addWidget(self.tab_widget)
        self.statusBar().showMessage("Ready")

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Exit", "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        event.accept() if reply == QMessageBox.Yes else event.ignore()


def main():
    """Entry point for the moyo GUI (moyo-gui console script)."""
    app = QApplication(sys.argv)
    app.setApplicationName("moyo GUI")
    app.setApplicationVersion("1.0.0")

    window = MoyoGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
