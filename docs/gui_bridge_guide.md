# moyo GUI Bridge Guide

The GUI Bridge provides a clean interface for GUI applications to process text and files, build FAISS indexes, and perform semantic searches.

## Overview

The `GUIBridge` class is the main interface for processing data and building indexes. It handles:

- Text validation and chunking
- File loading and processing
- Embedding generation
- FAISS index creation and management
- Search functionality
- Progress tracking and statistics

## Quick Start

### Basic Text Processing

```python
from moyo.privateside.datainput.gui_bridge import GUIBridge, ProcessingConfig

# Create configuration
config = ProcessingConfig(
    chunk_size=512,
    chunk_overlap=50,
    embedding_model="all-MiniLM-L6-v2",
    index_type="flat",
    save_index=True,
    output_dir="indexes/private"
)

# Create bridge
bridge = GUIBridge(config)

# Process text
text = "Your text content here..."
result = bridge.process_text(text, "my_document")

if result.success:
    print(f"Created {result.chunks_created} chunks")
    print(f"Index saved to: {result.index_path}")
else:
    print(f"Error: {result.message}")
```

### File Processing

```python
# Process a single file
result = bridge.process_file("path/to/document.txt")

# Process multiple files
file_paths = ["doc1.txt", "doc2.md", "doc3.json"]
results = bridge.process_files(file_paths)

for result in results:
    if result.success:
        print(f"✅ {result.chunks_created} chunks created")
    else:
        print(f"❌ {result.message}")
```

### Searching

```python
# Search the index
search_result = bridge.search_index("your search query", k=10)

if search_result["success"]:
    for result in search_result["results"]:
        print(f"Rank {result['rank']}: {result['distance']:.4f}")
        print(f"Preview: {result['metadata']['text_preview']}")
```

## Configuration

### ProcessingConfig

The `ProcessingConfig` class controls how data is processed:

```python
config = ProcessingConfig(
    chunk_size=512,              # Size of text chunks
    chunk_overlap=50,            # Overlap between chunks
    embedding_model="all-MiniLM-L6-v2",  # Embedding model
    batch_size=32,               # Batch size for embeddings
    index_type="flat",           # FAISS index type: "flat", "ivf", "hnsw"
    save_index=True,             # Whether to save index to disk
    output_dir="indexes/private" # Output directory
)
```

### Index Types

- **flat**: Simple exact search, fastest for small datasets
- **ivf**: Inverted file index, good for large datasets
- **hnsw**: Hierarchical navigable small world, good balance of speed/accuracy

## Supported File Types

The bridge supports multiple file formats:

- **Text files**: `.txt`, `.text`
- **Markdown**: `.md`, `.markdown`
- **JSON**: `.json`
- **CSV**: `.csv`

Files are automatically converted to text for processing.

## CLI Usage

The bridge also provides a command-line interface:

### Process Text
```bash
python -m moyo.privateside.datainput.cli process "Your text content"
```

### Process Files
```bash
# Single file
python -m moyo.privateside.datainput.cli process --file document.txt

# Multiple files
python -m moyo.privateside.datainput.cli process --files doc1.txt doc2.md doc3.json
```

### Search Index
```bash
python -m moyo.privateside.datainput.cli search indexes/private --query "search term"
```

### Get Index Info
```bash
python -m moyo.privateside.datainput.cli info indexes/private
```

## GUI Integration

### Web Application Example

```python
from flask import Flask, request, jsonify
from moyo.privateside.datainput.gui_bridge import GUIBridge, ProcessingConfig

app = Flask(__name__)
bridge = GUIBridge()

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text', '')
    
    result = bridge.process_text(text, "web_input")
    return jsonify(result.to_dict())

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 10)
    
    search_result = bridge.search_index(query, k)
    return jsonify(search_result)

@app.route('/stats', methods=['GET'])
def get_stats():
    stats = bridge.get_processing_stats()
    return jsonify(stats)
```

### Desktop Application Example

```python
import tkinter as tk
from tkinter import filedialog, messagebox
from moyo.privateside.datainput.gui_bridge import GUIBridge

class MoyoGUI:
    def __init__(self):
        self.bridge = GUIBridge()
        self.setup_ui()
    
    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("moyo Data Processor")
        
        # Text input
        tk.Label(self.root, text="Text Input:").pack()
        self.text_input = tk.Text(self.root, height=10, width=50)
        self.text_input.pack()
        
        # File selection
        tk.Button(self.root, text="Select Files", command=self.select_files).pack()
        
        # Process button
        tk.Button(self.root, text="Process", command=self.process_data).pack()
        
        # Search
        tk.Label(self.root, text="Search:").pack()
        self.search_input = tk.Entry(self.root, width=50)
        self.search_input.pack()
        tk.Button(self.root, text="Search", command=self.search).pack()
        
        # Results
        self.results_text = tk.Text(self.root, height=10, width=50)
        self.results_text.pack()
    
    def select_files(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        self.selected_files = files
    
    def process_data(self):
        # Process text input
        text = self.text_input.get("1.0", tk.END).strip()
        if text:
            result = self.bridge.process_text(text, "gui_input")
            if result.success:
                messagebox.showinfo("Success", f"Processed {result.chunks_created} chunks")
            else:
                messagebox.showerror("Error", result.message)
        
        # Process files
        if hasattr(self, 'selected_files') and self.selected_files:
            results = self.bridge.process_files(self.selected_files)
            successful = sum(1 for r in results if r.success)
            messagebox.showinfo("Files", f"Processed {successful}/{len(results)} files")
    
    def search(self):
        query = self.search_input.get()
        if query:
            result = self.bridge.search_index(query, k=5)
            if result["success"]:
                self.results_text.delete("1.0", tk.END)
                for r in result["results"]:
                    self.results_text.insert(tk.END, 
                        f"Rank {r['rank']}: {r['metadata']['text_preview']}\n")
            else:
                messagebox.showerror("Search Error", result["message"])
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MoyoGUI()
    app.run()
```

## Error Handling

The bridge provides comprehensive error handling:

```python
result = bridge.process_text(text, "source")

if not result.success:
    print(f"Error: {result.message}")
    for error in result.errors:
        print(f"  - {error}")
```

Common error scenarios:
- Invalid text (too short, too long, no content)
- File not found or unreadable
- Unsupported file format
- Embedding model not available
- FAISS not installed

## Performance Considerations

### Embedding Models
- **all-MiniLM-L6-v2**: Fast, good quality, 384 dimensions
- **all-mpnet-base-v2**: Higher quality, slower, 768 dimensions
- **text-embedding-3-large**: OpenAI model, requires API key

### Index Types
- **flat**: Best for < 1M vectors
- **ivf**: Good for 1M-100M vectors
- **hnsw**: Good for 100K-10M vectors

### Batch Processing
For large datasets, process files in batches:

```python
# Process files in batches
batch_size = 10
for i in range(0, len(files), batch_size):
    batch = files[i:i+batch_size]
    results = bridge.process_files(batch)
    # Update progress bar
```

## Advanced Features

### Custom Metadata
```python
# Add custom metadata to chunks
metadata = [{
    "source": "my_document",
    "chunk_index": i,
    "custom_field": "custom_value",
    "timestamp": "2024-01-01T00:00:00Z"
} for i in range(len(chunks))]

bridge.current_index.add_vectors(embeddings, metadata)
```

### Index Management
```python
# Save index
bridge.current_index.save(Path("my_index"))

# Load index
bridge.load_index("my_index")

# Clear index
bridge.clear_index()

# Get index info
info = bridge.get_index_info()
print(f"Vector count: {info['vector_count']}")
```

### Progress Tracking
```python
# Get processing statistics
stats = bridge.get_processing_stats()
print(f"Total chunks: {stats['total_chunks_created']}")
print(f"Total vectors: {stats['total_vectors_created']}")
print(f"Processing time: {stats['total_processing_time']:.2f}s")
```

## Troubleshooting

### Common Issues

1. **"FAISS not available"**
   - Install FAISS: `pip install faiss-cpu` or `pip install faiss-gpu`

2. **"sentence-transformers not available"**
   - Install: `pip install sentence-transformers`
   - Falls back to stub embeddings if not available

3. **"File too large"**
   - Increase file size limit in validators or split large files

4. **"Invalid file format"**
   - Check supported extensions: `.txt`, `.md`, `.json`, `.csv`

### Debug Mode
Enable debug logging to see detailed processing information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

See `examples/gui_bridge_example.py` for complete working examples.

Run the example:
```bash
cd moyo
python examples/gui_bridge_example.py
```
