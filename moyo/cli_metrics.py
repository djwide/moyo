"""CLI commands for metrics and monitoring."""

import click
import json
import requests
from pathlib import Path
from typing import Optional

from .config.settings import get_settings, reload_settings
from .metrics_server import start_metrics_server, stop_metrics_server, get_metrics_server
from .metrics import get_metrics_registry
from .logging import get_logger


@click.group()
def metrics():
    """Metrics and monitoring commands."""
    pass


@metrics.command()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--daemon/--no-daemon', default=True, help='Run as daemon')
@click.option('--port', '-p', type=int, help='Port to run metrics server on')
def start(config: Optional[str], daemon: bool, port: Optional[int]):
    """Start the Prometheus metrics server."""
    settings = get_settings()
    
    if config:
        settings = reload_settings(config)
    
    if port:
        settings.prometheus.port = port
    
    logger = get_logger("metrics-server")
    logger.info("Starting metrics server", 
               port=settings.prometheus.port,
               path=settings.prometheus.path)
    
    try:
        server = start_metrics_server(daemon=daemon)
        
        if not daemon:
            # Keep the main thread alive
            import time
            try:
                while server.is_running():
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping server")
                server.stop()
        
        logger.info("Metrics server started successfully")
        
    except Exception as e:
        logger.error("Failed to start metrics server", error=str(e))
        raise click.ClickException(f"Failed to start metrics server: {e}")


@metrics.command()
def stop():
    """Stop the Prometheus metrics server."""
    logger = get_logger("metrics-server")
    logger.info("Stopping metrics server")
    
    try:
        stop_metrics_server()
        logger.info("Metrics server stopped successfully")
    except Exception as e:
        logger.error("Failed to stop metrics server", error=str(e))
        raise click.ClickException(f"Failed to stop metrics server: {e}")


@metrics.command()
@click.option('--format', '-f', 'output_format', 
              type=click.Choice(['prometheus', 'json', 'summary']), 
              default='summary', help='Output format')
@click.option('--url', '-u', help='Metrics server URL')
def show(output_format: str, url: Optional[str]):
    """Show current metrics."""
    settings = get_settings()
    
    if url:
        metrics_url = f"{url.rstrip('/')}{settings.prometheus.path}"
    else:
        metrics_url = f"http://localhost:{settings.prometheus.port}{settings.prometheus.path}"
    
    try:
        response = requests.get(metrics_url, timeout=10)
        response.raise_for_status()
        metrics_data = response.text
        
        if output_format == 'prometheus':
            click.echo(metrics_data)
        elif output_format == 'json':
            # Convert Prometheus format to JSON summary
            summary = _parse_prometheus_metrics(metrics_data)
            click.echo(json.dumps(summary, indent=2))
        else:  # summary
            summary = _parse_prometheus_metrics(metrics_data)
            _display_metrics_summary(summary)
            
    except requests.exceptions.RequestException as e:
        raise click.ClickException(f"Failed to fetch metrics from {metrics_url}: {e}")
    except Exception as e:
        raise click.ClickException(f"Failed to process metrics: {e}")


@metrics.command()
@click.option('--url', '-u', help='Metrics server URL')
def health(url: Optional[str]):
    """Check metrics server health."""
    settings = get_settings()
    
    if url:
        health_url = f"{url.rstrip('/')}/health"
    else:
        health_url = f"http://localhost:{settings.prometheus.port}/health"
    
    try:
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        health_data = response.json()
        
        if health_data.get('status') == 'healthy':
            click.echo("✅ Metrics server is healthy")
            click.echo(f"Service: {health_data.get('service', 'unknown')}")
            click.echo(f"Timestamp: {health_data.get('timestamp', 'unknown')}")
        else:
            click.echo("❌ Metrics server is unhealthy")
            click.echo(f"Status: {health_data.get('status', 'unknown')}")
            
    except requests.exceptions.RequestException as e:
        click.echo(f"❌ Cannot connect to metrics server: {e}")
        raise click.ClickException(f"Health check failed: {e}")
    except Exception as e:
        raise click.ClickException(f"Failed to check health: {e}")


@metrics.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', type=click.Choice(['prometheus', 'json']), 
              default='prometheus', help='Export format')
def export(output: Optional[str], format: str):
    """Export current metrics to file."""
    try:
        metrics_registry = get_metrics_registry()
        metrics_data = metrics_registry.get_metrics()
        
        if format == 'json':
            # Convert to JSON format
            summary = _parse_prometheus_metrics(metrics_data)
            export_data = json.dumps(summary, indent=2)
        else:
            export_data = metrics_data
        
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(export_data)
            click.echo(f"Metrics exported to {output_path}")
        else:
            click.echo(export_data)
            
    except Exception as e:
        raise click.ClickException(f"Failed to export metrics: {e}")


def _parse_prometheus_metrics(metrics_data: str) -> dict:
    """Parse Prometheus metrics format into a structured summary."""
    summary = {
        'counters': {},
        'gauges': {},
        'histograms': {},
        'summaries': {}
    }
    
    for line in metrics_data.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        try:
            # Parse metric line (name{labels} value)
            if '{' in line:
                # Metric with labels
                name_part, value_part = line.rsplit('}', 1)
                name_with_labels = name_part + '}'
                value = float(value_part.strip())
                
                # Extract name and labels
                name_end = name_with_labels.find('{')
                name = name_with_labels[:name_end]
                labels_str = name_with_labels[name_end+1:-1]
                
                # Parse labels
                labels = {}
                if labels_str:
                    for label_pair in labels_str.split(','):
                        if '=' in label_pair:
                            key, val = label_pair.split('=', 1)
                            labels[key.strip()] = val.strip().strip('"')
                
            else:
                # Simple metric without labels
                name, value = line.split(' ', 1)
                value = float(value)
                labels = {}
            
            # Categorize by metric type
            if name.endswith('_total'):
                summary['counters'][name] = {'value': value, 'labels': labels}
            elif name.endswith('_seconds') or name.endswith('_bytes'):
                if 'bucket' in name or 'quantile' in name:
                    summary['histograms'][name] = {'value': value, 'labels': labels}
                else:
                    summary['gauges'][name] = {'value': value, 'labels': labels}
            else:
                summary['gauges'][name] = {'value': value, 'labels': labels}
                
        except Exception:
            # Skip malformed lines
            continue
    
    return summary


def _display_metrics_summary(summary: dict):
    """Display a human-readable metrics summary."""
    click.echo("📊 Metrics Summary")
    click.echo("=" * 50)
    
    # Counters
    if summary['counters']:
        click.echo("\n🔢 Counters:")
        for name, data in summary['counters'].items():
            labels_str = ', '.join([f"{k}={v}" for k, v in data['labels'].items()])
            if labels_str:
                click.echo(f"  {name}{{{labels_str}}} = {data['value']}")
            else:
                click.echo(f"  {name} = {data['value']}")
    
    # Gauges
    if summary['gauges']:
        click.echo("\n📈 Gauges:")
        for name, data in summary['gauges'].items():
            labels_str = ', '.join([f"{k}={v}" for k, v in data['labels'].items()])
            if labels_str:
                click.echo(f"  {name}{{{labels_str}}} = {data['value']}")
            else:
                click.echo(f"  {name} = {data['value']}")
    
    # Histograms
    if summary['histograms']:
        click.echo("\n📊 Histograms:")
        for name, data in summary['histograms'].items():
            labels_str = ', '.join([f"{k}={v}" for k, v in data['labels'].items()])
            if labels_str:
                click.echo(f"  {name}{{{labels_str}}} = {data['value']}")
            else:
                click.echo(f"  {name} = {data['value']}")


if __name__ == "__main__":
    metrics()
