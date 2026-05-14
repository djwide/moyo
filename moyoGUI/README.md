# Moyo GUI

A comprehensive graphical user interface for the Moyo project, providing user-friendly access to all major functionality through a tabbed interface.

## Features

### 🔒 Private Side Data Input Tab
- **File Upload**: Upload various file types (text, PDF, DOCX, CSV)
- **Direct Text Input**: Paste text directly into the interface
- **Processing Options**: Configure chunking parameters (size, overlap)
- **Real-time Processing**: Process data with progress tracking

### 🗂️ Private Side Corpus Mapping Tab
- **Corpus Selection**: Choose corpus directories for mapping
- **Schema Configuration**: Select mapping schemas (default or custom)
- **Deduplication Options**: Enable/disable deduplication
- **Build Management**: Build corpus maps with progress tracking

### 🌐 Public Side Gather Sources Tab
- **Multiple Sources**: Support for conferences, git commits, arXiv, PubMed, and custom sources
- **Search Configuration**: Configure search queries and date ranges
- **Source Selection**: Choose which data sources to gather from
- **Progress Tracking**: Monitor gathering progress

### 🔍 Public Side Barrier Probe Tab
- **Probe Types**: Support for iterative LLM search, LLM fuzzer, and barrier analyzer
- **Target Configuration**: Specify targets (URLs, identifiers)
- **Parameter Tuning**: Configure iterations, timeouts, and other parameters
- **Results Display**: View probe results in a table format

## Installation

### Prerequisites

- Python 3.8 or higher
- PyQt5 (GUI framework)
- Moyo project dependencies

### Installation Steps

1. **Install PyQt5**:
   ```bash
   pip install PyQt5
   ```

2. **Install Moyo dependencies**:
   ```bash
   cd moyo
   pip install -e .
   ```
   The vendored `shared_utils` package is installed automatically as part of moyo.

## Usage

### Running the GUI

#### Method 1: Using the launcher script (recommended)
```bash
cd moyo/moyoGUI
python run_moyo_gui.py
```

#### Method 2: Direct execution
```bash
cd moyo/moyoGUI
python moyo_gui.py
```

### Workflow

1. **Private Side Data Input**:
   - Select input files or paste text directly
   - Configure processing options
   - Click "Process Data" to begin processing

2. **Private Side Corpus Mapping**:
   - Select corpus directory
   - Choose mapping schema and options
   - Click "Build Corpus Map" to create mappings

3. **Public Side Gather Sources**:
   - Select data sources to gather from
   - Configure search queries and date ranges
   - Click "Gather Sources" to collect data

4. **Public Side Barrier Probe**:
   - Choose probe type and target
   - Configure parameters
   - Click "Start Probe" to begin analysis

## Architecture

The Moyo GUI is built with PyQt5 and provides:

- **Tabbed Interface**: Organized access to different functionality
- **Threaded Operations**: Background processing to keep UI responsive
- **Progress Tracking**: Real-time progress bars and status updates
- **Error Handling**: Comprehensive error reporting and recovery
- **Cross-platform Support**: Works on Windows, macOS, and Linux

## Dependencies

- `PyQt5`: GUI framework
- `moyo`: Core Moyo functionality
- `shared_utils`: Shared utilities from the SenTe project

## Development

To modify the GUI:

1. Edit the appropriate tab class in `moyo_gui.py`
2. Test changes by running the GUI
3. Update documentation as needed

### Adding New Tabs

To add a new tab:

1. Create a new tab class inheriting from `QWidget`
2. Implement the `init_ui()` method
3. Add the tab to the main window in `MoyoGUI.__init__()`

## Troubleshooting

### Common Issues

1. **Import errors**:
   - Ensure all dependencies are installed
   - Check that the project paths are correctly set
   - Verify Python environment

2. **GUI won't start**:
   - Ensure PyQt5 is installed: `python -c "from PyQt5.QtWidgets import QApplication"`
   - Check for missing dependencies

3. **Functionality not working**:
   - Verify that the underlying Moyo modules are properly installed
   - Check the console output for error messages

## Configuration

The GUI uses the same configuration as the underlying Moyo modules. Configuration files should be placed in the appropriate Moyo data directories.

## License

This GUI is part of the Moyo project and follows the same licensing terms.
