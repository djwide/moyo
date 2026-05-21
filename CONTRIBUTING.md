# Contributing to moyo

Thank you for contributing!

## Naming Conventions

These rules exist so names stay predictable across the codebase, docs, and
release artefacts.

| Context | Style | Examples |
|---------|-------|---------|
| **Repo / PyPI package name** | lowercase | `moyo` |
| **Python package directory** | lowercase | `moyo/`, `moyo/gui/` |
| **Console scripts** | lowercase with hyphens | `moyo`, `moyo-gather`, `moyo-gui` |
| **Python classes** | PascalCase | `MoyoGUI`, `CorpusBuilder`, `BarrierAnalyzer` |
| **Environment variables** | `MOYO_` prefix, SCREAMING_SNAKE | `MOYO_LOG_LEVEL`, `OPENAI_API_KEY` |
| **Prometheus metric namespace** | lowercase | `moyo_` prefix (`moyo_requests_total`) |
| **Prose / README H1 / PR titles** | "Moyo" (title case) | "Moyo v0.2 released" |

> **Rule of thumb:** if a human reads it as a word ("Moyo is fast"), capitalise
> the M.  If a computer parses it as an identifier (file path, package import,
> shell command, env var key), keep it lowercase or use the convention for that
> language.

## Development Setup

```bash
git clone https://github.com/<org>/moyo.git
cd moyo

# Base install (no GUI or monitoring deps)
pip install -e .

# With monitoring extras
pip install -e ".[monitoring]"

# With GUI extras
pip install -e ".[gui]"

# With everything
pip install -e ".[monitoring,gui]"
```

## Running Tests

```bash
pytest tests/ -v
```

The CI workflow (`.github/workflows/ci.yml`) runs on every push and pull
request: it installs the package with `pip install -e .`, smoke-tests the
public imports, and then runs `pytest`.

## Code Style

- Python 3.10+, type hints on public functions.
- `black` for formatting, `ruff` for linting (no enforced CI gate yet — just
  be consistent with surrounding code).
- Docstrings on all public classes and functions.
- Do **not** add `*.egg-info/` to the repository; it is already in `.gitignore`.

## Project Structure

```
moyo/               # Main package
  cli.py            # moyo entry point
  gui/              # Desktop GUI (moyo-gui)
  privateside/      # Data ingestion + corpus building
  publicside/       # Public source gathering + barrier probing
  redteam/          # Red-team utilities
shared_utils/       # Vendored shared utilities (included in the wheel)
docs/               # Operational runbook and guides
tests/              # Test suite
moyoGUI/            # Legacy launcher (kept for backwards compatibility)
```
