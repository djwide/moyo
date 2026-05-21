#!/usr/bin/env python3
"""
moyo GUI - A comprehensive GUI for the moyo project
Provides tabs for data input, FAISS index creation, and 2D visualization
"""

import sys
import os
import threading
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QTextEdit, QFileDialog, QProgressBar,
    QGroupBox, QScrollArea, QComboBox, QLineEdit, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon

# Add the project root to the path to import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import numpy as np
    from sklearn.manifold import MDS
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some functionality may be limited")
    MATPLOTLIB_AVAILABLE = False


class WorkerThread(QThread):
    """Worker thread for running operations"""
    output = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, operation, args=None):
        super().__init__()
        self.operation = operation
        self.args = args or []
        
    def run(self):
        try:
            # Capture stdout/stderr
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            
            output_buffer = io.StringIO()
            
            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                if self.operation == "process_data":
                    self.process_data_input()
                elif self.operation == "create_index":
                    self.create_faiss_index()
                elif self.operation == "visualize":
                    self.create_visualization()
                    
            output = output_buffer.getvalue()
            self.output.emit(output)
            self.finished.emit(True, "Operation completed successfully")
            
        except Exception as e:
            self.finished.emit(False, str(e))
    
    def process_data_input(self):
        """Process data input and save to corpus"""
        # This would integrate with the actual moyo data processing
        pass
    
    def create_faiss_index(self):
        """Create FAISS index from corpus"""
        # This would integrate with the actual FAISS index creation
        pass
    
    def create_visualization(self):
        """Create 2D visualization"""
        # This would integrate with the actual visualization
        pass


class DataInputTab(QWidget):
    """Tab for data input functionality - text, file, or folder to corpus"""
    
    def __init__(self):
        super().__init__()
        self.corpus_data = []
        self.default_corpus_path = Path(__file__).parent.parent / "data" / "private" / "corpus.txt"
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Data Input - Text, File, or Folder to Corpus")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Input method selection
        method_group = QGroupBox("Input Method")
        method_layout = QVBoxLayout()
        
        self.text_radio = QCheckBox("Direct Text Input")
        self.file_radio = QCheckBox("Single File")
        self.folder_radio = QCheckBox("Folder")
        
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
        
        # File type selection
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
        
        # File extensions
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
        
        # Default location checkbox
        self.use_default_checkbox = QCheckBox("Use default location (moyo/data/private/corpus.txt)")
        self.use_default_checkbox.setChecked(True)
        self.use_default_checkbox.toggled.connect(self.on_output_method_changed)
        output_layout.addWidget(self.use_default_checkbox)
        
        # Custom location
        custom_layout = QHBoxLayout()
        self.custom_path_label = QLabel("No custom path selected")
        select_output_btn = QPushButton("Select Custom Output File")
        select_output_btn.clicked.connect(self.select_output_file)
        custom_layout.addWidget(self.custom_path_label)
        custom_layout.addWidget(select_output_btn)
        output_layout.addLayout(custom_layout)
        

        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Process button
        self.process_btn = QPushButton("Process Data and Save to Corpus")
        self.process_btn.clicked.connect(self.process_data)
        layout.addWidget(self.process_btn)
        
        # Progress and output
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)
        
        self.setLayout(layout)
        
    def on_input_method_changed(self):
        """Handle input method selection"""
        self.text_group.setVisible(self.text_radio.isChecked())
        self.file_group.setVisible(self.file_radio.isChecked())
        self.folder_group.setVisible(self.folder_radio.isChecked())
        
    def on_output_method_changed(self):
        """Handle output method selection"""
        use_default = self.use_default_checkbox.isChecked()
        self.custom_path_label.setEnabled(not use_default)
        
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", "", 
            "Text Files (*.txt);;All Files (*);;PDF Files (*.pdf);;Word Files (*.docx);;CSV Files (*.csv);;JSON Files (*.json)"
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
        """Process data input and save to corpus"""
        self.output_text.append("Processing data input...")
        self.progress_bar.setValue(0)
        
        try:
            # Determine input data
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
            
            # Determine output path
            if self.use_default_checkbox.isChecked():
                output_path = self.default_corpus_path
                # Ensure directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_path = Path(self.custom_path_label.text())
            
            # Save corpus
            self.save_corpus(data, output_path)
            
            self.progress_bar.setValue(100)
            self.output_text.append(f"Corpus saved to: {output_path}")
            self.output_text.append(f"Processed {len(data)} items")
            
        except Exception as e:
            self.output_text.append(f"Error processing data: {str(e)}")
            self.progress_bar.setValue(0)
            
    def process_text_input(self):
        """Process direct text input - each line as separate item"""
        text_content = self.text_input.toPlainText()
        if not text_content.strip():
            return []
        
        lines = text_content.strip().split('\n')
        items = []
        
        for i, line in enumerate(lines):
            if line.strip():  # Only process non-empty lines
                items.append({
                    "id": f"text_line_{i}",
                    "text": line.strip(),
                    "source": "direct_text",
                    "chunk_id": i
                })
        
        return items
    
    def process_file_input(self):
        """Process single file input - each line as separate item"""
        file_path = self.file_path_label.text()
        if file_path == "No file selected":
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            items = []
            for i, line in enumerate(lines):
                if line.strip():  # Only process non-empty lines
                    items.append({
                        "id": f"file_line_{i}",
                        "text": line.strip(),
                        "source": Path(file_path).name,
                        "chunk_id": i
                    })
            
            return items
            
        except Exception as e:
            self.output_text.append(f"Error reading file: {str(e)}")
            return []
    
    def process_folder_input(self):
        """Process folder input - each line as separate item"""
        folder_path = self.folder_path_label.text()
        if folder_path == "No folder selected":
            return []
        
        try:
            folder = Path(folder_path)
            extensions = [ext.strip() for ext in self.file_extensions.text().split(',')]
            
            all_items = []
            item_counter = 0
            
            for ext in extensions:
                for file_path in folder.glob(ext):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        for i, line in enumerate(lines):
                            if line.strip():  # Only process non-empty lines
                                all_items.append({
                                    "id": f"{file_path.stem}_line_{item_counter}",
                                    "text": line.strip(),
                                    "source": file_path.name,
                                    "chunk_id": item_counter
                                })
                                item_counter += 1
                    except Exception as e:
                        self.output_text.append(f"Error reading {file_path}: {str(e)}")
            
            return all_items
            
        except Exception as e:
            self.output_text.append(f"Error processing folder: {str(e)}")
            return []
    
    def save_corpus(self, data, output_path):
        """Save corpus to file in TXT format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(f"ID: {item['id']}\n")
                f.write(f"Source: {item['source']}\n")
                f.write(f"Text: {item['text']}\n")
                f.write("-" * 50 + "\n")


class FAISSIndexTab(QWidget):
    """Tab for creating FAISS index from corpus"""
    
    # Embedding model mapping (same as sente)
    EMBEDDING_MODELS = {
        "mini": "all-MiniLM-L6-v2",
        "multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "openai-large": "text-embedding-3-large",
        "openai-small": "text-embedding-3-small",
    }
    
    # Model dimensions
    MODEL_DIMENSIONS = {
        "mini": 384,
        "multilingual": 768,
        "openai-large": 3072,
        "openai-small": 1536,
    }
    
    def __init__(self):
        super().__init__()
        self.default_corpus_path = Path(__file__).parent.parent / "data" / "private" / "corpus.txt"
        self.default_private_index_path = Path(__file__).parent.parent / "data" / "private_faiss_index"
        self.default_public_index_path = Path(__file__).parent.parent / "data" / "public_faiss_index"
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Create FAISS Index from Corpus")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Corpus selection
        corpus_group = QGroupBox("Corpus Selection")
        corpus_layout = QVBoxLayout()
        
        # Default corpus checkbox
        self.use_default_corpus_checkbox = QCheckBox("Use default corpus (moyo/data/private/corpus.txt)")
        self.use_default_corpus_checkbox.setChecked(True)
        self.use_default_corpus_checkbox.toggled.connect(self.on_corpus_method_changed)
        corpus_layout.addWidget(self.use_default_corpus_checkbox)
        
        # Custom corpus selection
        corpus_btn_layout = QHBoxLayout()
        self.corpus_path_label = QLabel("No custom corpus selected")
        select_corpus_btn = QPushButton("Select Custom Corpus File")
        select_corpus_btn.clicked.connect(self.select_corpus)
        corpus_btn_layout.addWidget(self.corpus_path_label)
        corpus_btn_layout.addWidget(select_corpus_btn)
        corpus_layout.addLayout(corpus_btn_layout)
        
        corpus_group.setLayout(corpus_layout)
        layout.addWidget(corpus_group)
        
        # Index options
        index_group = QGroupBox("Index Options")
        index_layout = QVBoxLayout()
        
        # Index type selection
        index_type_layout = QHBoxLayout()
        index_type_layout.addWidget(QLabel("Index Type:"))
        self.index_type_combo = QComboBox()
        self.index_type_combo.addItems(["IndexFlatL2", "IndexIVFFlat", "IndexHNSW"])
        index_type_layout.addWidget(self.index_type_combo)
        index_layout.addLayout(index_type_layout)
        
        # Embedding model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Embedding Model:"))
        self.embedding_model_combo = QComboBox()
        self.embedding_model_combo.addItems(["mini", "multilingual", "openai-large", "openai-small"])
        self.embedding_model_combo.setCurrentText("mini")
        self.embedding_model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.embedding_model_combo)
        index_layout.addLayout(model_layout)
        
        # Index parameters
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Dimension:"))
        self.dimension_label = QLabel("384")
        params_layout.addWidget(self.dimension_label)
        index_layout.addLayout(params_layout)
        
        index_group.setLayout(index_layout)
        layout.addWidget(index_group)
        
        # Output options
        output_group = QGroupBox("Index Output")
        output_layout = QVBoxLayout()
        
        # Index type selection
        index_type_layout = QHBoxLayout()
        index_type_layout.addWidget(QLabel("Create Index For:"))
        self.index_type_radio = QComboBox()
        self.index_type_radio.addItems(["Private", "Public"])
        self.index_type_radio.currentTextChanged.connect(self.on_index_type_changed)
        index_type_layout.addWidget(self.index_type_radio)
        output_layout.addLayout(index_type_layout)
        
        # Default location checkbox
        self.use_default_index_checkbox = QCheckBox("Use default location")
        self.use_default_index_checkbox.setChecked(True)
        self.use_default_index_checkbox.toggled.connect(self.on_index_output_changed)
        output_layout.addWidget(self.use_default_index_checkbox)
        
        # Custom location
        index_output_layout = QHBoxLayout()
        self.index_output_label = QLabel("No custom index location selected")
        select_index_output_btn = QPushButton("Select Custom Index Location")
        select_index_output_btn.clicked.connect(self.select_index_output)
        index_output_layout.addWidget(self.index_output_label)
        index_output_layout.addWidget(select_index_output_btn)
        output_layout.addLayout(index_output_layout)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Create index button
        self.create_index_btn = QPushButton("Create FAISS Index")
        self.create_index_btn.clicked.connect(self.create_index)
        layout.addWidget(self.create_index_btn)
        
        # Progress and output
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)
        
        self.setLayout(layout)
        
    def on_corpus_method_changed(self):
        """Handle corpus method selection"""
        use_default = self.use_default_corpus_checkbox.isChecked()
        self.corpus_path_label.setEnabled(not use_default)
        
    def on_index_output_changed(self):
        """Handle index output method selection"""
        use_default = self.use_default_index_checkbox.isChecked()
        self.index_output_label.setEnabled(not use_default)
        
    def on_model_changed(self):
        """Handle embedding model selection"""
        model_key = self.embedding_model_combo.currentText()
        dimension = self.MODEL_DIMENSIONS.get(model_key, 384)
        self.dimension_label.setText(str(dimension))
        
    def on_index_type_changed(self):
        """Handle index type selection"""
        # Update checkbox text based on index type
        index_type = self.index_type_radio.currentText()
        if index_type == "Private":
            self.use_default_index_checkbox.setText(f"Use default location (moyo/data/private_faiss_index)")
        else:
            self.use_default_index_checkbox.setText(f"Use default location (moyo/data/public_faiss_index)")
        
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
        """Create FAISS index from corpus"""
        self.output_text.append("Creating FAISS index...")
        self.progress_bar.setValue(0)
        
        try:
            # Determine corpus path
            if self.use_default_corpus_checkbox.isChecked():
                corpus_path = self.default_corpus_path
            else:
                corpus_path = Path(self.corpus_path_label.text())
            
            if not corpus_path.exists():
                self.output_text.append(f"Corpus file not found: {corpus_path}")
                return
            
            self.progress_bar.setValue(10)
            
            # Determine index output path
            index_type = self.index_type_radio.currentText()
            if self.use_default_index_checkbox.isChecked():
                if index_type == "Private":
                    index_path = self.default_private_index_path
                else:
                    index_path = self.default_public_index_path
            else:
                index_path = Path(self.index_output_label.text())
            
            # Ensure output directory exists
            index_path.mkdir(parents=True, exist_ok=True)
            
            self.progress_bar.setValue(20)
            
            # Load corpus from TXT file
            corpus_data = []
            current_item = {}
            with open(corpus_path, 'r', encoding='utf-8') as f:
                content = f.read()
                sections = content.split('-' * 50)
                
                for section in sections:
                    if section.strip():
                        lines = section.strip().split('\n')
                        item = {}
                        for line in lines:
                            if line.startswith('ID: '):
                                item['id'] = line[4:].strip()
                            elif line.startswith('Source: '):
                                item['source'] = line[8:].strip()
                            elif line.startswith('Text: '):
                                item['text'] = line[6:].strip()
                        
                        if item.get('text', '').strip():
                            corpus_data.append(item)
            
            self.output_text.append(f"Loaded corpus with {len(corpus_data)} items")
            self.progress_bar.setValue(30)
            
            # Extract text data from corpus
            texts = [item.get('text', '') for item in corpus_data if item.get('text', '').strip()]
            
            if not texts:
                self.output_text.append("No valid text data found in corpus")
                return
            
            self.output_text.append(f"Processing {len(texts)} text items...")
            self.progress_bar.setValue(40)
            
            # Get embedding model name
            model_key = self.embedding_model_combo.currentText()
            model_name = self.EMBEDDING_MODELS.get(model_key, "all-MiniLM-L6-v2")
            
            # Import shared_utils functions
            try:
                from shared_utils.embeddings import embed
                from shared_utils.faiss_index import FAISSIndex
                import faiss
                # Set GPU as default
                if hasattr(faiss, 'GpuIndexFlatIP'):
                    self.output_text.append("Using FAISS GPU backend")
                else:
                    self.output_text.append("FAISS GPU not available, using CPU")
            except ImportError as e:
                self.output_text.append(f"Error importing shared_utils: {str(e)}")
                return
            
            # Generate embeddings
            self.output_text.append(f"Generating embeddings using {model_name}...")
            self.progress_bar.setValue(50)
            embeddings = embed(texts, model_name=model_name, batch_size=32, normalize=True)
            
            if not embeddings:
                self.output_text.append("Failed to generate embeddings")
                return
            
            self.output_text.append(f"Generated {len(embeddings)} embeddings")
            self.progress_bar.setValue(70)
            
            # Get index type and parameters
            index_type = self.index_type_combo.currentText().lower()
            if "flat" in index_type:
                index_type = "flat"
            elif "ivf" in index_type:
                index_type = "ivf"
            elif "hnsw" in index_type:
                index_type = "hnsw"
            else:
                index_type = "flat"
            
            dimension = self.MODEL_DIMENSIONS.get(model_key, 384)
            
            # Create FAISS index
            self.output_text.append(f"Creating {index_type.upper()} index with dimension {dimension}...")
            self.progress_bar.setValue(80)
            faiss_index = FAISSIndex(dimension=dimension, index_type=index_type)
            
            # Add vectors with metadata
            metadata = []
            for i, item in enumerate(corpus_data):
                if item.get('text', '').strip():
                    metadata.append({
                        "id": item.get('id', f"item_{i}"),
                        "source": item.get('source', 'unknown'),
                        "chunk_id": item.get('chunk_id', i),
                        "text": item.get('text', '')[:100] + "..." if len(item.get('text', '')) > 100 else item.get('text', '')
                    })
            
            faiss_index.add_vectors(embeddings, metadata)
            self.progress_bar.setValue(90)
            
            # Save index
            self.output_text.append(f"Saving index to {index_path}...")
            faiss_index.save(index_path)
            
            self.output_text.append(f"FAISS index created successfully!")
            self.output_text.append(f"Index saved to: {index_path}")
            self.output_text.append(f"Total vectors: {faiss_index.get_vector_count()}")
            self.output_text.append(f"Index type: {index_type.upper()}")
            self.output_text.append(f"Dimension: {dimension}")
            self.progress_bar.setValue(100)
            
        except Exception as e:
            self.output_text.append(f"Error creating index: {str(e)}")
            import traceback
            self.output_text.append(f"Traceback: {traceback.format_exc()}")
            self.progress_bar.setValue(0)


class GatherPublicSourcesTab(QWidget):
    """Tab for gathering public sources"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Gather Public Sources")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel("This tab allows you to gather public sources for building public FAISS indices.")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Placeholder content
        placeholder = QLabel("Public source gathering functionality will be implemented here.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(placeholder)
        
        self.setLayout(layout)


class BarrierProbeTab(QWidget):
    """Tab for barrier probing"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Barrier Probe")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel("This tab allows you to perform barrier probing analysis.")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Placeholder content
        placeholder = QLabel("Barrier probing functionality will be implemented here.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(placeholder)
        
        self.setLayout(layout)


class BuildPublicCorpusTab(QWidget):
    """Tab for building public indices"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Build Public Index")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel("This tab allows you to gather public sources and build public FAISS indices.")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Placeholder content
        placeholder = QLabel("Public index building functionality will be implemented here.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(placeholder)
        
        self.setLayout(layout)


class VisualizationTab(QWidget):
    """Tab for 2D FAISS index visualization"""
    
    def __init__(self):
        super().__init__()
        self.default_private_index_path = Path(__file__).parent.parent / "data" / "private_faiss_index"
        self.default_public_index_path = Path(__file__).parent.parent / "data" / "public_faiss_index"
        self.private_index = None
        self.public_index = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("2D FAISS Index Visualization")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Data loading section
        data_group = QGroupBox("Load FAISS Indices")
        data_layout = QVBoxLayout()
        
        # Private index selection
        private_layout = QHBoxLayout()
        self.private_index_label = QLabel("No private index selected")
        select_private_btn = QPushButton("Load Private Index")
        select_private_btn.clicked.connect(self.load_private_index)
        private_layout.addWidget(self.private_index_label)
        private_layout.addWidget(select_private_btn)
        data_layout.addLayout(private_layout)
        
        # Use default private index checkbox
        self.use_default_private_checkbox = QCheckBox("Use default private index (moyo/data/private_faiss_index)")
        self.use_default_private_checkbox.setChecked(True)
        self.use_default_private_checkbox.toggled.connect(self.on_private_method_changed)
        data_layout.addWidget(self.use_default_private_checkbox)
        
        # Public index selection
        public_layout = QHBoxLayout()
        self.public_index_label = QLabel("No public index selected")
        select_public_btn = QPushButton("Load Public Index")
        select_public_btn.clicked.connect(self.load_public_index)
        public_layout.addWidget(self.public_index_label)
        public_layout.addWidget(select_public_btn)
        data_layout.addLayout(public_layout)
        
        # Use default public index checkbox
        self.use_default_public_checkbox = QCheckBox("Use default public index (moyo/data/public_faiss_index)")
        self.use_default_public_checkbox.setChecked(True)
        self.use_default_public_checkbox.toggled.connect(self.on_public_method_changed)
        data_layout.addWidget(self.use_default_public_checkbox)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Visualization options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QVBoxLayout()
        
        # Algorithm selection
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel("Dimensionality Reduction:"))
        self.viz_algo_combo = QComboBox()
        self.viz_algo_combo.addItems(["mds"])
        algo_layout.addWidget(self.viz_algo_combo)
        viz_layout.addLayout(algo_layout)

        # Parameters
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Random State:"))
        self.viz_random_state_input = QLineEdit("42")
        params_layout.addWidget(self.viz_random_state_input)
        viz_layout.addLayout(params_layout)

        # Semantic distance explanation
        self.distance_info_label = QLabel(
            "Points are spaced so that cosine distance reflects semantic similarity."
        )
        self.distance_info_label.setWordWrap(True)
        viz_layout.addWidget(self.distance_info_label)
        
        # Distance visualization options
        distance_group = QGroupBox("Distance Visualization")
        distance_layout = QVBoxLayout()
        
        # Show distance lines checkbox
        self.show_distance_lines_checkbox = QCheckBox("Show distance lines between closest points")
        self.show_distance_lines_checkbox.setChecked(True)
        distance_layout.addWidget(self.show_distance_lines_checkbox)
        
        # Distance threshold for lines
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Distance threshold for lines:"))
        self.distance_threshold_input = QLineEdit("0.5")
        threshold_layout.addWidget(self.distance_threshold_input)
        distance_layout.addLayout(threshold_layout)
        
        # Show clustering checkbox
        self.show_clustering_checkbox = QCheckBox("Show distance-based clustering")
        self.show_clustering_checkbox.setChecked(False)
        distance_layout.addWidget(self.show_clustering_checkbox)
        
        # Number of clusters
        clusters_layout = QHBoxLayout()
        clusters_layout.addWidget(QLabel("Number of clusters:"))
        self.num_clusters_input = QLineEdit("5")
        clusters_layout.addWidget(self.num_clusters_input)
        distance_layout.addLayout(clusters_layout)
        
        distance_group.setLayout(distance_layout)
        viz_layout.addWidget(distance_group)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Visualization buttons
        viz_btn_layout = QHBoxLayout()
        
        self.viz_btn = QPushButton("Generate 2D Visualization")
        self.viz_btn.clicked.connect(self.generate_visualization)
        viz_btn_layout.addWidget(self.viz_btn)
        
        self.heatmap_btn = QPushButton("Generate Distance Heatmap")
        self.heatmap_btn.clicked.connect(self.generate_distance_heatmap)
        viz_btn_layout.addWidget(self.heatmap_btn)
        
        layout.addLayout(viz_btn_layout)
        
        # Export section
        export_group = QGroupBox("Export Visualization")
        export_layout = QVBoxLayout()
        
        export_btn_layout = QHBoxLayout()
        self.viz_export_path_label = QLabel("No export path selected")
        select_viz_export_btn = QPushButton("Select Export File")
        select_viz_export_btn.clicked.connect(self.select_viz_export_file)
        export_btn_layout.addWidget(self.viz_export_path_label)
        export_btn_layout.addWidget(select_viz_export_btn)
        export_layout.addLayout(export_btn_layout)
        
        # Export buttons
        export_btn_layout = QHBoxLayout()
        
        self.viz_export_btn = QPushButton("Export Main Visualization")
        self.viz_export_btn.clicked.connect(self.export_visualization)
        self.viz_export_btn.setEnabled(False)
        export_btn_layout.addWidget(self.viz_export_btn)
        
        self.heatmap_export_btn = QPushButton("Export Heatmap")
        self.heatmap_export_btn.clicked.connect(self.export_heatmap)
        self.heatmap_export_btn.setEnabled(False)
        export_btn_layout.addWidget(self.heatmap_export_btn)
        
        export_layout.addLayout(export_btn_layout)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Visualization canvases with scroll area
        if MATPLOTLIB_AVAILABLE:
            # Create scroll area for visualizations
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setMinimumHeight(600)
            
            # Create container widget for visualizations
            viz_container = QWidget()
            viz_container_layout = QVBoxLayout()
            
            # Main visualization canvas
            viz_canvas_group = QGroupBox("Main Visualization")
            viz_canvas_layout = QVBoxLayout()
            
            # Add expand/collapse button
            viz_header_layout = QHBoxLayout()
            viz_header_layout.addWidget(QLabel("Main Plot with Distance Analysis"))
            self.viz_expand_btn = QPushButton("Expand/Collapse")
            self.viz_expand_btn.clicked.connect(lambda: self.toggle_plot_size(self.canvas, viz_canvas_group))
            viz_header_layout.addWidget(self.viz_expand_btn)
            viz_canvas_layout.addLayout(viz_header_layout)
            
            self.figure = Figure(figsize=(10, 8))
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumSize(400, 300)
            viz_canvas_layout.addWidget(self.canvas)
            viz_canvas_group.setLayout(viz_canvas_layout)
            viz_container_layout.addWidget(viz_canvas_group)
            
            # Heatmap canvas (completely separate)
            heatmap_canvas_group = QGroupBox("Distance Heatmap (Standalone)")
            heatmap_canvas_layout = QVBoxLayout()
            
            # Add expand/collapse button for heatmap
            heatmap_header_layout = QHBoxLayout()
            heatmap_header_layout.addWidget(QLabel("Distance Density Heatmap"))
            self.heatmap_expand_btn = QPushButton("Expand/Collapse")
            self.heatmap_expand_btn.clicked.connect(lambda: self.toggle_plot_size(self.heatmap_canvas, heatmap_canvas_group))
            heatmap_header_layout.addWidget(self.heatmap_expand_btn)
            heatmap_canvas_layout.addLayout(heatmap_header_layout)
            
            self.heatmap_figure = Figure(figsize=(10, 8))
            self.heatmap_canvas = FigureCanvas(self.heatmap_figure)
            self.heatmap_canvas.setMinimumSize(400, 300)
            heatmap_canvas_layout.addWidget(self.heatmap_canvas)
            heatmap_canvas_group.setLayout(heatmap_canvas_layout)
            viz_container_layout.addWidget(heatmap_canvas_group)
            
            # Set container as scroll area widget
            viz_container.setLayout(viz_container_layout)
            scroll_area.setWidget(viz_container)
            layout.addWidget(scroll_area)
        else:
            no_viz_label = QLabel("Matplotlib not available. Please install matplotlib and scikit-learn for visualization.")
            no_viz_label.setStyleSheet("color: red;")
            layout.addWidget(no_viz_label)
        
        # Progress and output
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)
        
        self.setLayout(layout)
        
    def on_private_method_changed(self):
        """Handle private index method selection"""
        use_default = self.use_default_private_checkbox.isChecked()
        self.private_index_label.setEnabled(not use_default)
        
    def on_public_method_changed(self):
        """Handle public index method selection"""
        use_default = self.use_default_public_checkbox.isChecked()
        self.public_index_label.setEnabled(not use_default)
        
    def load_private_index(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Load Private FAISS Index")
        if folder_path:
            self.private_index_label.setText(folder_path)
            
    def load_public_index(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Load Public FAISS Index")
        if folder_path:
            self.public_index_label.setText(folder_path)
            
    def select_viz_export_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Export File", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        if file_path:
            self.viz_export_path_label.setText(file_path)
            self.viz_export_btn.setEnabled(True)
            
    def generate_visualization(self):
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Matplotlib not available for visualization")
            return
            
        try:
            self.output_text.append("Generating 2D FAISS index visualization...")
            
            # Load private index
            if self.use_default_private_checkbox.isChecked():
                private_index_path = self.default_private_index_path
            else:
                private_index_path = Path(self.private_index_label.text())
            
            if not private_index_path.exists():
                self.output_text.append(f"Private index not found: {private_index_path}")
                return
            
            # Load public index
            if self.use_default_public_checkbox.isChecked():
                public_index_path = self.default_public_index_path
            else:
                public_index_path = Path(self.public_index_label.text())
            
            if not public_index_path.exists():
                self.output_text.append(f"Public index not found: {public_index_path}")
                return
            
            # Import shared_utils functions
            try:
                from shared_utils.faiss_index import FAISSIndex
            except ImportError as e:
                self.output_text.append(f"Error importing shared_utils: {str(e)}")
                return
            
            # Load indices
            self.output_text.append("Loading private index...")
            self.private_index = FAISSIndex.load(private_index_path)
            private_vectors = self.private_index.index.reconstruct_n(0, self.private_index.get_vector_count())
            
            self.output_text.append("Loading public index...")
            self.public_index = FAISSIndex.load(public_index_path)
            public_vectors = self.public_index.index.reconstruct_n(0, self.public_index.get_vector_count())
            
            self.output_text.append(f"Loaded {len(private_vectors)} private vectors and {len(public_vectors)} public vectors")
            
            # Combine vectors and compute cosine distances
            all_vectors = np.vstack([private_vectors, public_vectors])
            from sklearn.metrics.pairwise import cosine_distances
            distance_matrix = cosine_distances(all_vectors)

            # Dimensionality reduction preserving distances
            algorithm = self.viz_algo_combo.currentText()
            random_state = int(self.viz_random_state_input.text())
            reducer = MDS(n_components=2, dissimilarity='precomputed', random_state=random_state)
            self.output_text.append("Applying MDS dimensionality reduction with cosine distances...")
            coords_2d = reducer.fit_transform(distance_matrix)

            # Split coordinates back to private and public
            private_coords = coords_2d[:len(private_vectors)]
            public_coords = coords_2d[len(private_vectors):]
            cross_distances = distance_matrix[:len(private_vectors), len(private_vectors):]
            
            # Create visualization
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Option 3: Distance lines between closest points
            if self.show_distance_lines_checkbox.isChecked() and len(private_coords) > 0 and len(public_coords) > 0:
                try:
                    threshold = float(self.distance_threshold_input.text())

                    for i, private_point in enumerate(private_coords):
                        closest_public_idx = np.argmin(cross_distances[i])
                        distance = cross_distances[i][closest_public_idx]

                        if distance <= threshold:
                            closest_public_point = public_coords[closest_public_idx]

                            line_width = max(0.5, 3.0 - (distance * 5.0))
                            alpha = max(0.2, 1.0 - (distance * 2.0))

                            ax.plot([
                                private_point[0],
                                closest_public_point[0],
                            ], [
                                private_point[1],
                                closest_public_point[1],
                            ], 'g-', alpha=alpha, linewidth=line_width)

                    self.output_text.append(
                        f"Drew distance lines for points within cosine distance {threshold}"
                    )

                except Exception as e:
                    self.output_text.append(f"Error drawing distance lines: {str(e)}")
            
            # Option 4: Distance-based clustering
            if self.show_clustering_checkbox.isChecked() and len(private_coords) > 0 and len(public_coords) > 0:
                try:
                    from sklearn.cluster import KMeans

                    # Calculate average cosine distance to other corpus for each point
                    private_avg_distances = np.mean(cross_distances, axis=1)
                    public_avg_distances = np.mean(cross_distances, axis=0)
                    
                    # Combine distances for clustering
                    all_distances = np.concatenate([private_avg_distances, public_avg_distances])
                    
                    # Reshape for clustering
                    distance_features = all_distances.reshape(-1, 1)
                    
                    # Perform clustering
                    n_clusters = int(self.num_clusters_input.text())
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(distance_features)
                    
                    # Split labels back to private and public
                    private_clusters = cluster_labels[:len(private_coords)]
                    public_clusters = cluster_labels[len(private_coords):]
                    
                    # Create color map for clusters
                    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
                    
                    # Clear the plot and redraw with clustering
                    ax.clear()
                    
                    # Plot private points with cluster colors
                    for i in range(n_clusters):
                        mask = private_clusters == i
                        if np.any(mask):
                            ax.scatter(private_coords[mask, 0], private_coords[mask, 1], 
                                      c=[colors[i]], label=f'Private Cluster {i+1}', 
                                      alpha=0.7, s=50, marker='o')
                    
                    # Plot public points with cluster colors
                    for i in range(n_clusters):
                        mask = public_clusters == i
                        if np.any(mask):
                            ax.scatter(public_coords[mask, 0], public_coords[mask, 1], 
                                      c=[colors[i]], label=f'Public Cluster {i+1}', 
                                      alpha=0.7, s=50, marker='s')
                    
                    # Redraw distance lines if enabled
                    if self.show_distance_lines_checkbox.isChecked():
                        for i, private_point in enumerate(private_coords):
                            closest_public_idx = np.argmin(cross_distances[i])
                            distance = cross_distances[i][closest_public_idx]

                            threshold = float(self.distance_threshold_input.text())
                            if distance <= threshold:
                                closest_public_point = public_coords[closest_public_idx]
                                line_width = max(0.5, 3.0 - (distance * 5.0))
                                alpha = max(0.2, 1.0 - (distance * 2.0))

                                ax.plot([
                                    private_point[0],
                                    closest_public_point[0],
                                ], [
                                    private_point[1],
                                    closest_public_point[1],
                                ], 'g-', alpha=alpha, linewidth=line_width)
                    
                    # Add cluster statistics
                    for i in range(n_clusters):
                        cluster_mask = cluster_labels == i
                        cluster_distances = all_distances[cluster_mask]
                        avg_distance = np.mean(cluster_distances)
                        self.output_text.append(f"Cluster {i+1}: {np.sum(cluster_mask)} points, avg distance: {avg_distance:.3f}")
                    
                    self.output_text.append(f"Applied distance-based clustering with {n_clusters} clusters")
                    
                except ImportError:
                    self.output_text.append("Warning: sklearn not available for clustering")
                except Exception as e:
                    self.output_text.append(f"Error applying clustering: {str(e)}")
            else:
                # If clustering is not enabled, draw the original scatter plot
                # Plot private vectors in blue
                if len(private_coords) > 0:
                    ax.scatter(private_coords[:, 0], private_coords[:, 1], 
                              c='blue', label='Private', alpha=0.7, s=50)
                
                # Plot public vectors in red
                if len(public_coords) > 0:
                    ax.scatter(public_coords[:, 0], public_coords[:, 1], 
                              c='red', label='Public', alpha=0.7, s=50)
            
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_title(f'2D FAISS Index Visualization ({algorithm.upper()})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Remove tick labels (numbers) from axes
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            self.canvas.draw()
            self.output_text.append(f"Visualization generated with {algorithm.upper()}")
            self.output_text.append(f"Private vectors: {len(private_coords)} (blue)")
            self.output_text.append(f"Public vectors: {len(public_coords)} (red)")
            self.output_text.append(
                "Semantic distances (cosine) are reflected in point spacing; closer points are more similar."
            )
            self.viz_export_btn.setEnabled(True)
            
        except Exception as e:
            self.output_text.append(f"Error generating visualization: {str(e)}")
            import traceback
            self.output_text.append(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to generate visualization: {str(e)}")
            
    def export_visualization(self):
        if not hasattr(self, 'figure'):
            QMessageBox.warning(self, "Warning", "No visualization to export")
            return
            
        export_path = self.viz_export_path_label.text()
        if export_path == "No export path selected":
            QMessageBox.warning(self, "Warning", "Please select an export file")
            return
            
        try:
            self.figure.savefig(export_path, dpi=300, bbox_inches='tight')
            self.output_text.append(f"Visualization exported to {export_path}")
            QMessageBox.information(self, "Success", f"Visualization exported successfully to {export_path}")
            
        except Exception as e:
            self.output_text.append(f"Error exporting visualization: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to export visualization: {str(e)}")
    
    def generate_distance_heatmap(self):
        """Generate distance heatmap overlay (Option 1)"""
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Matplotlib not available for visualization")
            return
            
        try:
            self.output_text.append("Generating distance heatmap...")
            
            # Check if we have loaded indices
            if self.private_index is None or self.public_index is None:
                QMessageBox.warning(self, "Warning", "Please load both private and public indices first")
                return
            
            # Get vectors from loaded indices
            private_vectors = self.private_index.index.reconstruct_n(0, self.private_index.get_vector_count())
            public_vectors = self.public_index.index.reconstruct_n(0, self.public_index.get_vector_count())
            
            # Combine vectors and compute cosine distances
            all_vectors = np.vstack([private_vectors, public_vectors])
            from sklearn.metrics.pairwise import cosine_distances
            distance_matrix = cosine_distances(all_vectors)

            # Dimensionality reduction using MDS
            algorithm = self.viz_algo_combo.currentText()
            random_state = int(self.viz_random_state_input.text())
            reducer = MDS(n_components=2, dissimilarity='precomputed', random_state=random_state)
            coords_2d = reducer.fit_transform(distance_matrix)

            # Split coordinates
            private_coords = coords_2d[:len(private_vectors)]
            public_coords = coords_2d[len(private_vectors):]
            cross_distances = distance_matrix[:len(private_vectors), len(private_vectors):]
            
            if len(private_coords) == 0 or len(public_coords) == 0:
                self.output_text.append("Need both private and public points for heatmap")
                return
            
            # Use cosine distances between original vectors
            distances = cross_distances
            
            # Create standalone heatmap
            self.heatmap_figure.clear()
            ax = self.heatmap_figure.add_subplot(111)
            
            # Create 2D histogram of distances
            x_coords = []
            y_coords = []
            distance_values = []
            
            for i, private_point in enumerate(private_coords):
                for j, public_point in enumerate(public_coords):
                    x_coords.append((private_point[0] + public_point[0]) / 2)  # Midpoint
                    y_coords.append((private_point[1] + public_point[1]) / 2)  # Midpoint
                    distance_values.append(distances[i, j])
            
            # Create 2D histogram
            x_coords = np.array(x_coords)
            y_coords = np.array(y_coords)
            distance_values = np.array(distance_values)
            
            # Create grid for heatmap
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add some padding
            x_pad = (x_max - x_min) * 0.1
            y_pad = (y_max - y_min) * 0.1
            
            x_bins = np.linspace(x_min - x_pad, x_max + x_pad, 50)
            y_bins = np.linspace(y_min - y_pad, y_max + y_pad, 50)
            
            # Create 2D histogram
            heatmap, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins], 
                                                      weights=distance_values)
            
            # Normalize by count
            count_hist, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])
            heatmap = np.divide(heatmap, count_hist, out=np.zeros_like(heatmap), where=count_hist > 0)
            
            # Plot standalone heatmap (no overlay points)
            im = ax.imshow(heatmap.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], 
                          cmap='hot', aspect='auto')
            
            # Add colorbar
            cbar = self.heatmap_figure.colorbar(im, ax=ax)
            cbar.set_label('Average Distance')
            
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_title(f'Distance Density Heatmap ({algorithm.upper()}) - Red areas show high private-public proximity')
            ax.grid(True, alpha=0.3)
            
            # Remove tick labels
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            self.heatmap_canvas.draw()
            
            # Add statistics
            min_distance = np.min(distances)
            max_distance = np.max(distances)
            mean_distance = np.mean(distances)
            
            self.output_text.append(f"Distance heatmap generated")
            self.output_text.append(f"Distance range: {min_distance:.3f} - {max_distance:.3f}")
            self.output_text.append(f"Average distance: {mean_distance:.3f}")
            self.output_text.append(f"Hot spots (red areas) indicate regions where private and public points are close together")
            
            # Enable heatmap export button
            self.heatmap_export_btn.setEnabled(True)
            
        except Exception as e:
            self.output_text.append(f"Error generating distance heatmap: {str(e)}")
            import traceback
            self.output_text.append(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to generate distance heatmap: {str(e)}")
    
    def export_heatmap(self):
        """Export the distance heatmap"""
        if not hasattr(self, 'heatmap_figure'):
            QMessageBox.warning(self, "Warning", "No heatmap to export")
            return
            
        export_path = self.viz_export_path_label.text()
        if export_path == "No export path selected":
            QMessageBox.warning(self, "Warning", "Please select an export file")
            return
            
        try:
            # Modify filename for heatmap
            heatmap_path = export_path.replace('.png', '_heatmap.png').replace('.pdf', '_heatmap.pdf').replace('.svg', '_heatmap.svg')
            self.heatmap_figure.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            self.output_text.append(f"Distance heatmap exported to {heatmap_path}")
            QMessageBox.information(self, "Success", f"Distance heatmap exported successfully to {heatmap_path}")
            
        except Exception as e:
            self.output_text.append(f"Error exporting heatmap: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to export heatmap: {str(e)}")
    
    def toggle_plot_size(self, canvas, group_box):
        """Toggle between normal and expanded plot size"""
        current_size = canvas.size()
        
        if current_size.width() <= 600:  # Currently normal size
            # Expand
            canvas.setMinimumSize(1200, 800)
            canvas.resize(1200, 800)
            group_box.setTitle(group_box.title() + " (Expanded)")
        else:
            # Collapse back to normal
            canvas.setMinimumSize(400, 300)
            canvas.resize(400, 300)
            group_box.setTitle(group_box.title().replace(" (Expanded)", ""))


class MoyoGUI(QMainWindow):
    """Main moyo GUI application"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("moyo GUI")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Add tabs
        self.tab_widget.addTab(DataInputTab(), "Private Data Input")
        self.tab_widget.addTab(BuildPublicCorpusTab(), "Build Public Corpus")
        self.tab_widget.addTab(FAISSIndexTab(), "Create FAISS Indices")
        self.tab_widget.addTab(BarrierProbeTab(), "Barrier Probe")
        self.tab_widget.addTab(VisualizationTab(), "2D FAISS Indices Visualization")
        
        layout.addWidget(self.tab_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def closeEvent(self, event):
        """Handle application close event"""
        reply = QMessageBox.question(
            self, 'Exit', 'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    """Main entry point for the moyo GUI"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("moyo GUI")
    app.setApplicationVersion("1.0.0")
    
    # Create and show the main window
    window = MoyoGUI()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
