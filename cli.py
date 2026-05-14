import click
from pathlib import Path

from .config.settings import get_settings
from .logging import setup_logging


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--config', '-c', help='Configuration file path')
def cli(verbose: bool, debug: bool, config: str) -> None:
    """
    Moyo - Experimental tooling for corpus mapping and barrier probing.
    
    Moyo provides tools for building knowledge corpora and assessing information barriers
    between private and public data sources.
    
    \b
    Key Components:
    • Private Side: Ingest local data and map into FAISS-backed corpus
    • Public Side: Gather open-source information and probe barriers between corpora
    
    \b
    Main Commands:
    • datainput: Process text/files and build FAISS indexes
    • corpus: Build and manage private corpora
    • probe: LLM-assisted fuzzing and barrier analysis
    • metrics: Prometheus metrics and monitoring
    
    For detailed help on any command, use: moyo <command> --help
    """
    # Load configuration
    settings = get_settings()
    
    # Setup logging
    if verbose:
        settings.logging.level = "INFO"
    if debug:
        settings.logging.level = "DEBUG"
    
    setup_logging(settings)


@cli.command(name="version")
def version_cmd() -> None:
    """Print version information."""
    try:
        from importlib.metadata import version
        print(f"moyo {version('moyo')}")
    except Exception:
        print("moyo (dev)")


@cli.command(name="info")
def info_cmd() -> None:
    """Display system information and configuration."""
    try:
        from importlib.metadata import version
        moyo_version = version('moyo')
    except Exception:
        moyo_version = "dev"
    
    try:
        from shared_utils import get_embedding_dimension
        embedding_dim = get_embedding_dimension()
        embedding_available = True
    except Exception:
        embedding_dim = "N/A"
        embedding_available = False
    
    click.echo("Moyo System Information")
    click.echo("=" * 40)
    click.echo(f"Version: {moyo_version}")
    click.echo(f"Python: {click.get_current_context().meta.get('python_version', 'Unknown')}")
    click.echo(f"Embedding Model: {'Available' if embedding_available else 'Not Available'}")
    if embedding_available:
        click.echo(f"Embedding Dimension: {embedding_dim}")
    
    # Check for data directories
    data_dirs = ['indexes/private', 'indexes/public', 'data/private', 'data/public']
    click.echo("\nData Directories:")
    for dir_path in data_dirs:
        if Path(dir_path).exists():
            click.echo(f"  ✅ {dir_path}")
        else:
            click.echo(f"  ❌ {dir_path} (not found)")


@cli.command(name="setup")
@click.option('--force', is_flag=True, help='Force recreation of directories')
def setup_cmd(force: bool) -> None:
    """Set up initial directory structure and configuration."""
    dirs_to_create = [
        'indexes/private',
        'indexes/public', 
        'data/private',
        'data/public',
        'logs'
    ]
    
    click.echo("Setting up Moyo directory structure...")
    
    for dir_path in dirs_to_create:
        path = Path(dir_path)
        if path.exists() and not force:
            click.echo(f"  ⚠️  {dir_path} (already exists)")
        else:
            path.mkdir(parents=True, exist_ok=True)
            click.echo(f"  ✅ {dir_path}")
    
    click.echo("\nSetup complete! You can now:")
    click.echo("  • Use 'moyo-datainput' to process private data")
    click.echo("  • Use 'moyo-corpus' to build corpora")
    click.echo("  • Use 'moyo-probe' for barrier analysis")
    click.echo("  • Use 'moyo metrics' for monitoring and metrics")


# Import and register subcommands
try:
    from .cli_metrics import metrics
    cli.add_command(metrics)
except ImportError:
    pass

try:
    from .redteam.cli import cli as redteam_cli
    cli.add_command(redteam_cli, name="redteam")
except ImportError:
    pass


if __name__ == "__main__":
    cli()
